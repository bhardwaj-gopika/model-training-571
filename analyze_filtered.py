"""Filtered correlation plots: show pred vs true only for 'acceptable beam' samples.

Beam quality thresholds (from accelerator physics group):
  - Beam size (sigma_x, sigma_y) < 5 mm
  - Relative energy spread std(E)/E < 5e-3
  - Normalized projected emittance (x, y) < 20 um
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from train import build_model, chol_vectors_to_covariance

# M-normalization used during target creation
M_DIAG = np.array([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6])
M_INV = np.diag(1.0 / M_DIAG)

# Beam quality thresholds
DEFAULT_THRESHOLDS = {
    "sigma_x_mm": 5.0,
    "sigma_y_mm": 5.0,
    "rel_energy_spread": 5e-3,
    "emit_x_norm_um": 20.0,
    "emit_y_norm_um": 20.0,
}

MEAN_TARGET_COLS = ["mean_energy", "mean_time"]


def compute_beam_properties(cov_norm: np.ndarray, mean_energy: np.ndarray) -> dict:
    """Compute physical beam properties from M-normalized covariance matrices."""
    # Denormalize: C_phys = M_inv @ C_norm @ M_inv^T
    cov_phys = np.einsum("ij,njk,lk->nil", M_INV, cov_norm, M_INV)

    sigma_x_mm = np.sqrt(np.abs(cov_phys[:, 0, 0])) * 1000
    sigma_y_mm = np.sqrt(np.abs(cov_phys[:, 2, 2])) * 1000
    std_pz = np.sqrt(np.abs(cov_phys[:, 5, 5]))
    rel_espread = std_pz / mean_energy

    # Normalized projected emittance: sqrt(det(2x2 subblock)) / (m_e * c)
    m_e_c = 511e3  # eV/c
    det_x = np.abs(cov_phys[:, 0, 0] * cov_phys[:, 1, 1] - cov_phys[:, 0, 1] ** 2)
    det_y = np.abs(cov_phys[:, 2, 2] * cov_phys[:, 3, 3] - cov_phys[:, 2, 3] ** 2)
    emit_x_norm_um = np.sqrt(det_x) / m_e_c * 1e6
    emit_y_norm_um = np.sqrt(det_y) / m_e_c * 1e6

    return {
        "sigma_x_mm": sigma_x_mm,
        "sigma_y_mm": sigma_y_mm,
        "rel_energy_spread": rel_espread,
        "emit_x_norm_um": emit_x_norm_um,
        "emit_y_norm_um": emit_y_norm_um,
    }


def apply_thresholds(props: dict, thresholds: dict) -> np.ndarray:
    """Return boolean mask: True where ALL thresholds are satisfied."""
    mask = np.ones(len(props["sigma_x_mm"]), dtype=bool)
    for key, limit in thresholds.items():
        mask &= props[key] < limit
    return mask


def covariance_labels():
    return [f"cov_{row}{col}" for row in range(6) for col in range(6)]


def load_model_and_predict(model_dir, test_csv, batch_size=256):
    """Load model and return predictions + targets for test set."""
    model_dir = Path(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tr = torch.load(model_dir / "input_transformers.pt", map_location="cpu")
    output_tr = torch.load(model_dir / "output_transformers.pt", map_location="cpu")

    feature_cols = list(input_tr["feature_cols"])
    chol_cols = list(output_tr["target_cols"])
    n_inputs = len(feature_cols)
    n_chol_outputs = len(chol_cols)

    # Check for mean outputs
    mean_tr_path = model_dir / "mean_transformers.pt"
    has_mean = mean_tr_path.exists()
    if has_mean:
        mean_tr = torch.load(mean_tr_path, map_location="cpu")
        mean_cols = mean_tr["mean_cols"]
        n_mean_outputs = len(mean_cols)
        mean_y_mean = mean_tr["mean_y_mean"]
        mean_y_std = mean_tr["mean_y_std"]
    else:
        mean_cols = []
        n_mean_outputs = 0
        mean_y_mean = None
        mean_y_std = None

    model = build_model(
        n_inputs, n_chol_outputs,
        n_mean_outputs=n_mean_outputs,
        mean_y_mean=mean_y_mean,
        mean_y_std=mean_y_std,
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True, map_location="cpu"))
    model.eval().to(device)

    # Load test data
    df = pd.read_csv(test_csv, low_memory=False)
    x_mean = input_tr["x_mean"].numpy()
    x_std = input_tr["x_std"].numpy()
    y_mean = output_tr["y_mean"].numpy()
    y_std = output_tr["y_std"].numpy()

    X_test = (df[feature_cols].values.astype(np.float32) - x_mean) / x_std
    y_test = (df[chol_cols].values.astype(np.float32) - y_mean) / y_std

    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Run inference
    all_cov_preds = []
    all_mean_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            output = model(X_batch.to(device))
            if has_mean:
                all_cov_preds.append(output[0].cpu().numpy())
                all_mean_preds.append(output[1].cpu().numpy())
            else:
                all_cov_preds.append(output.cpu().numpy())

    preds_cov = np.concatenate(all_cov_preds)  # (N, 6, 6) covariance matrices

    # Reconstruct target covariance from raw Cholesky vectors
    y_raw = df[chol_cols].values.astype(np.float32)
    targets_cov = chol_vectors_to_covariance(torch.from_numpy(y_raw)).numpy()

    # Mean beam predictions (raw scale)
    preds_mean = np.concatenate(all_mean_preds) if has_mean else None
    targets_mean = df[mean_cols].values.astype(np.float32) if has_mean else None

    return {
        "preds_cov": preds_cov,
        "targets_cov": targets_cov,
        "preds_mean": preds_mean,
        "targets_mean": targets_mean,
        "mean_cols": mean_cols,
        "has_mean": has_mean,
        "df": df,
    }


def plot_filtered_scatter(preds_cov, targets_cov, mask, out_path, title_suffix=""):
    """6x6 scatter: pred vs true for covariance elements, filtered samples only."""
    labels = covariance_labels()
    preds_flat = preds_cov[mask].reshape(-1, 36)
    targets_flat = targets_cov[mask].reshape(-1, 36)
    n_pass = mask.sum()

    fig, axes = plt.subplots(6, 6, figsize=(22, 22))
    fig.suptitle(
        f"Predicted vs True covariance (acceptable beam, n={n_pass}){title_suffix}",
        fontsize=14, y=1.01,
    )
    r_squared = []
    for k in range(36):
        r, c = divmod(k, 6)
        ax = axes[r, c]
        tgt = targets_flat[:, k]
        prd = preds_flat[:, k]
        ax.scatter(tgt, prd, s=8, color="steelblue", alpha=0.5)
        all_vals = np.concatenate([tgt, prd])
        lo, hi = np.nanpercentile(all_vals, [1, 99])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4)
        ss_res = np.sum((prd - tgt) ** 2)
        ss_tot = np.sum((tgt - tgt.mean()) ** 2)
        r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared.append(r_sq)
        ax.set_title(f"{labels[k]} (R²={r_sq:.3f})", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("True", fontsize=6)
        ax.set_ylabel("Predicted", fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return r_squared


def plot_filtered_mean_scatter(preds_mean, targets_mean, mean_cols, mask, out_path):
    """Scatter for mean beam outputs, filtered samples only."""
    n_pass = mask.sum()
    fig, axes = plt.subplots(1, len(mean_cols), figsize=(7 * len(mean_cols), 6))
    if len(mean_cols) == 1:
        axes = [axes]
    for idx, (col, ax) in enumerate(zip(mean_cols, axes)):
        tgt = targets_mean[mask, idx]
        prd = preds_mean[mask, idx]
        ax.scatter(tgt, prd, s=8, color="steelblue", alpha=0.5)
        lo, hi = np.nanpercentile(np.concatenate([tgt, prd]), [1, 99])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4)
        ss_res = np.sum((prd - tgt) ** 2)
        ss_tot = np.sum((tgt - tgt.mean()) ** 2)
        r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        ax.set_title(f"{col} (R²={r_sq:.4f}, n={n_pass})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_distributions(props, thresholds, out_path):
    """Histogram of each beam property with threshold lines."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (key, values) in enumerate(props.items()):
        ax = axes[idx]
        # Clip display for readability
        display_max = thresholds[key] * 5
        clipped = np.clip(values, 0, display_max)
        ax.hist(clipped, bins=60, color="steelblue", alpha=0.7, edgecolor="none")
        ax.axvline(thresholds[key], color="red", linestyle="--", linewidth=2,
                   label=f"threshold = {thresholds[key]}")
        n_pass = (values < thresholds[key]).sum()
        ax.set_title(f"{key}\n({n_pass}/{len(values)} pass = {n_pass/len(values)*100:.1f}%)")
        ax.set_xlabel(key)
        ax.legend(fontsize=9)
    # Hide extra subplot
    axes[5].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate filtered correlation plots using beam quality thresholds."
    )
    parser.add_argument("--model-dir", default="model-output-571")
    parser.add_argument("--test-csv", default="dataset-test.csv")
    parser.add_argument("--output-dir", default="analysis-filtered-571")
    parser.add_argument("--sigma-x-mm", type=float, default=DEFAULT_THRESHOLDS["sigma_x_mm"])
    parser.add_argument("--sigma-y-mm", type=float, default=DEFAULT_THRESHOLDS["sigma_y_mm"])
    parser.add_argument("--rel-energy-spread", type=float, default=DEFAULT_THRESHOLDS["rel_energy_spread"])
    parser.add_argument("--emit-x-um", type=float, default=DEFAULT_THRESHOLDS["emit_x_norm_um"])
    parser.add_argument("--emit-y-um", type=float, default=DEFAULT_THRESHOLDS["emit_y_norm_um"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "sigma_x_mm": args.sigma_x_mm,
        "sigma_y_mm": args.sigma_y_mm,
        "rel_energy_spread": args.rel_energy_spread,
        "emit_x_norm_um": args.emit_x_um,
        "emit_y_norm_um": args.emit_y_um,
    }

    print("[run] Loading model and running inference ...", flush=True)
    results = load_model_and_predict(args.model_dir, args.test_csv, args.batch_size)

    targets_cov = results["targets_cov"]  # (N, 6, 6) M-normalized
    preds_cov = results["preds_cov"]
    df = results["df"]

    # Compute beam properties from TRUE covariance
    mean_energy = df["mean_energy"].values if "mean_energy" in df.columns else None
    if mean_energy is None:
        raise SystemExit("mean_energy column required in test CSV for energy spread threshold")

    print("[run] Computing beam properties from true covariance ...", flush=True)
    props = compute_beam_properties(targets_cov, mean_energy)
    mask = apply_thresholds(props, thresholds)

    n_total = len(mask)
    n_pass = mask.sum()
    print(f"\n[filter] Samples passing beam quality thresholds: {n_pass}/{n_total} ({n_pass/n_total*100:.1f}%)")
    for key, limit in thresholds.items():
        n_key = (props[key] < limit).sum()
        print(f"  {key} < {limit}: {n_key}/{n_total} ({n_key/n_total*100:.1f}%)")

    # Plot threshold distributions
    print("\n[plot] Beam property distributions with thresholds ...", flush=True)
    plot_threshold_distributions(props, thresholds, output_dir / "beam_property_distributions.png")

    # Filtered scatter: covariance elements
    print("[plot] Filtered covariance scatter (pred vs true) ...", flush=True)
    r_squared = plot_filtered_scatter(
        preds_cov, targets_cov, mask, output_dir / "scatter_filtered.png"
    )

    # Also generate unfiltered for comparison
    mask_all = np.ones(n_total, dtype=bool)
    r_squared_all = plot_filtered_scatter(
        preds_cov, targets_cov, mask_all, output_dir / "scatter_all.png",
        title_suffix=" (all samples)",
    )

    # Save metrics comparison
    labels = covariance_labels()
    metrics_df = pd.DataFrame({
        "element": labels,
        "r_squared_all": r_squared_all,
        "r_squared_filtered": r_squared,
    })
    metrics_df["r_squared_improvement"] = metrics_df["r_squared_filtered"] - metrics_df["r_squared_all"]
    metrics_df.to_csv(output_dir / "filtered_metrics.csv", index=False)

    # Filtered MAPE
    preds_flat = preds_cov[mask].reshape(-1, 36)
    targets_flat = targets_cov[mask].reshape(-1, 36)
    abs_error = np.abs(preds_flat - targets_flat)
    abs_true = np.abs(targets_flat)
    target_scale = abs_true.std(axis=0)
    pct_mask = abs_true > 0.01 * target_scale[None, :]
    pct_errors = np.where(pct_mask, abs_error / abs_true * 100, np.nan)
    mape_filtered = np.nanmean(pct_errors)

    # Unfiltered MAPE
    preds_flat_all = preds_cov.reshape(-1, 36)
    targets_flat_all = targets_cov.reshape(-1, 36)
    abs_error_all = np.abs(preds_flat_all - targets_flat_all)
    abs_true_all = np.abs(targets_flat_all)
    target_scale_all = abs_true_all.std(axis=0)
    pct_mask_all = abs_true_all > 0.01 * target_scale_all[None, :]
    pct_errors_all = np.where(pct_mask_all, abs_error_all / abs_true_all * 100, np.nan)
    mape_all = np.nanmean(pct_errors_all)

    print(f"\n[results] MAPE (all samples):      {mape_all:.2f}%")
    print(f"[results] MAPE (filtered samples): {mape_filtered:.2f}%")
    print(f"[results] Mean R² (all):           {np.mean(r_squared_all):.4f}")
    print(f"[results] Mean R² (filtered):      {np.mean(r_squared):.4f}")

    # Mean beam filtered scatter
    if results["has_mean"]:
        print("[plot] Filtered mean beam scatter ...", flush=True)
        plot_filtered_mean_scatter(
            results["preds_mean"], results["targets_mean"],
            results["mean_cols"], mask,
            output_dir / "mean_beam_scatter_filtered.png",
        )

    # Summary text file
    with open(output_dir / "filter_summary.txt", "w") as f:
        f.write("Beam Quality Filter Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write("Thresholds:\n")
        for key, val in thresholds.items():
            f.write(f"  {key}: {val}\n")
        f.write(f"\nSamples: {n_pass}/{n_total} pass ({n_pass/n_total*100:.1f}%)\n")
        f.write(f"\nMAPE (all samples):      {mape_all:.2f}%\n")
        f.write(f"MAPE (filtered samples): {mape_filtered:.2f}%\n")
        f.write(f"Mean R² (all):           {np.mean(r_squared_all):.4f}\n")
        f.write(f"Mean R² (filtered):      {np.mean(r_squared):.4f}\n")

    print(f"\n[run] Done. Results in {output_dir}/")


if __name__ == "__main__":
    main()
