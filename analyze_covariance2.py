"""Analyze training results for covariance-output models (571)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from train2 import build_model, chol_vectors_to_covariance

PHASE_SPACE_VARS = ["x", "px", "y", "py", "t", "pz"]
MEAN_TARGET_COLS = [f"mean_{v}" for v in PHASE_SPACE_VARS]

# M-normalization diagonal (from AGENTS.md / lume_model_utils.py)
M_DIAG = np.array([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6], dtype=np.float64)

# Display-unit scale factors: SI -> display
# x,y: m -> mm (1e3), px,py,pz: eV/c -> MeV/c (1e-6), t: s -> ps (1e12)
DISPLAY_SCALE = np.array([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6], dtype=np.float64)
DISPLAY_UNITS = ["mm", "MeV/c", "mm", "MeV/c", "ps", "MeV/c"]

# Combined per-element scale: M-denorm then display-unit conversion
# cov_display[i,j] = cov_mnorm[i,j] / (M[i]*M[j]) * display[i]*display[j]
_ELEMENT_SCALE = (DISPLAY_SCALE / M_DIAG)  # per-axis factor
COV_ELEMENT_SCALE = np.outer(_ELEMENT_SCALE, _ELEMENT_SCALE).ravel()  # (36,)


def covariance_display_labels():
    """Return 6x6 covariance labels with display units."""
    labels = []
    for i in range(6):
        for j in range(6):
            u_i, u_j = DISPLAY_UNITS[i], DISPLAY_UNITS[j]
            unit = f"{u_i}·{u_j}" if u_i != u_j else f"{u_i}²"
            labels.append(f"cov_{i}{j} [{unit}]")
    return labels


def to_display_units(cov_array):
    """Convert (N,6,6) or (N,36) M-normalized covariances to display units."""
    shape = cov_array.shape
    flat = cov_array.reshape(-1, 36)
    flat_display = flat * COV_ELEMENT_SCALE[None, :]
    return flat_display.reshape(shape)


def load_model_and_transformers(model_dir: Path, dropout: float = 0.05, activation: str = "elu"):
    """Load trained covariance-output model and transformation dictionaries."""
    input_tr = torch.load(model_dir / "input_transformers.pt", map_location="cpu")
    output_tr = torch.load(model_dir / "output_transformers.pt", map_location="cpu")
    cov_tr = torch.load(model_dir / "covariance_transformers.pt", map_location="cpu")

    feature_cols = list(input_tr["feature_cols"])
    chol_cols = list(output_tr["target_cols"])
    n_inputs = len(feature_cols)
    n_chol_outputs = len(chol_cols)

    # Check for mean beam transformers
    mean_tr_path = model_dir / "mean_transformers.pt"
    if mean_tr_path.exists():
        mean_tr = torch.load(mean_tr_path, map_location="cpu")
        mean_cols = mean_tr["mean_cols"]
        n_mean_outputs = len(mean_cols)
        mean_y_mean = mean_tr["mean_y_mean"]
        mean_y_std = mean_tr["mean_y_std"]
    else:
        mean_tr = None
        mean_cols = []
        n_mean_outputs = 0
        mean_y_mean = None
        mean_y_std = None

    model = build_model(
        n_inputs, n_chol_outputs, n_mean_outputs=n_mean_outputs,
        mean_y_mean=mean_y_mean, mean_y_std=mean_y_std,
        dropout=dropout, activation=activation,
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True, map_location="cpu"))
    model.eval()

    return model, input_tr, output_tr, cov_tr, mean_tr, mean_cols


def covariance_labels():
    """Return flattened 6x6 covariance element labels."""
    return [f"cov_{row}{col}" for row in range(6) for col in range(6)]


def evaluate_on_test(model, test_loader, output_tr, cov_tr, device, has_mean=False):
    """Evaluate covariance-output model on the test set."""
    y_mean = output_tr["y_mean"].cpu().numpy()
    y_std = output_tr["y_std"].cpu().numpy()
    cov_mean = cov_tr["cov_mean"].cpu().numpy().reshape(1, 36)
    cov_std = cov_tr["cov_std"].cpu().numpy().reshape(1, 36)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch.to(device))
            pred_cov = output[0].cpu().numpy() if has_mean else output.cpu().numpy()
            all_preds.append(pred_cov)
            all_targets.append(y_batch.cpu().numpy())

    preds_cov = np.concatenate(all_preds)
    targets_norm = np.concatenate(all_targets)
    targets_chol_raw = targets_norm * y_std + y_mean
    targets_cov = chol_vectors_to_covariance(torch.from_numpy(targets_chol_raw)).cpu().numpy()

    preds_cov_flat = preds_cov.reshape(preds_cov.shape[0], -1)
    targets_cov_flat = targets_cov.reshape(targets_cov.shape[0], -1)

    abs_error = np.abs(preds_cov_flat - targets_cov_flat)
    mae_per_element = abs_error.mean(axis=0)
    mse_per_element = ((preds_cov_flat - targets_cov_flat) ** 2).mean(axis=0)
    rmse_per_element = np.sqrt(mse_per_element)

    # Objective-space metrics (normalized covariance elements).
    preds_cov_norm = (preds_cov_flat - cov_mean) / cov_std
    targets_cov_norm = (targets_cov_flat - cov_mean) / cov_std
    norm_mse_per_element = ((preds_cov_norm - targets_cov_norm) ** 2).mean(axis=0)
    norm_rmse_per_element = np.sqrt(norm_mse_per_element)

    abs_true = np.abs(targets_cov_flat)
    target_scale = abs_true.std(axis=0)
    mask = abs_true > 0.01 * target_scale[None, :]
    pct_errors = np.where(mask, abs_error / abs_true * 100, np.nan)
    mape_per_element = np.nanmean(pct_errors, axis=0)

    return {
        "preds_cov": preds_cov,
        "targets_cov": targets_cov,
        "labels": covariance_labels(),
        "mae_per_element": mae_per_element,
        "rmse_per_element": rmse_per_element,
        "norm_rmse_per_element": norm_rmse_per_element,
        "mape_per_element": mape_per_element,
        "mae_overall": mae_per_element.mean(),
        "rmse_overall": rmse_per_element.mean(),
        "norm_rmse_overall": norm_rmse_per_element.mean(),
        "mape_overall": np.nanmean(pct_errors),
    }


def predict_covariances(model, loader, output_tr, device, has_mean=False):
    """Return predicted and target covariance matrices for a given loader."""
    y_mean = output_tr["y_mean"].cpu().numpy()
    y_std = output_tr["y_std"].cpu().numpy()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            output = model(X_batch.to(device))
            pred_cov = output[0].cpu().numpy() if has_mean else output.cpu().numpy()
            all_preds.append(pred_cov)
            all_targets.append(y_batch.cpu().numpy())

    preds_cov = np.concatenate(all_preds)
    targets_norm = np.concatenate(all_targets)
    targets_chol_raw = targets_norm * y_std + y_mean
    targets_cov = chol_vectors_to_covariance(torch.from_numpy(targets_chol_raw)).cpu().numpy()
    return preds_cov, targets_cov


def plot_per_sample_overlay(preds_cov, targets_cov, labels, out_path: Path, max_samples: int = 1000):
    """Plot per-sample predicted values over true values for each covariance element."""
    preds_flat = preds_cov.reshape(preds_cov.shape[0], -1)
    targets_flat = targets_cov.reshape(targets_cov.shape[0], -1)

    n = preds_flat.shape[0]
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=int)
        preds_flat = preds_flat[idx]
        targets_flat = targets_flat[idx]

    x = np.arange(preds_flat.shape[0])
    fig, axes = plt.subplots(6, 6, figsize=(22, 22), sharex=True)
    fig.suptitle("Per-sample agreement: Target vs Predicted covariance values", fontsize=14, y=1.01)
    for k in range(36):
        r, c = divmod(k, 6)
        ax = axes[r, c]
        ax.plot(x, targets_flat[:, k], color="steelblue", linewidth=0.8, alpha=0.6, label="Target")
        ax.plot(x, preds_flat[:, k], color="tomato", linewidth=0.8, alpha=0.6, label="Predicted")
        ax.set_title(labels[k], fontsize=8)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("Sample index", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample_zoomed_dots(
    preds_cov,
    targets_cov,
    labels,
    out_path: Path,
    max_samples: int = 1000,
    low_q: float = 5.0,
    high_q: float = 95.0,
):
    """Plot zoomed per-sample agreement using dots and percentile y-limits."""
    preds_flat = preds_cov.reshape(preds_cov.shape[0], -1)
    targets_flat = targets_cov.reshape(targets_cov.shape[0], -1)

    n = preds_flat.shape[0]
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=int)
        preds_flat = preds_flat[idx]
        targets_flat = targets_flat[idx]

    x = np.arange(preds_flat.shape[0])
    fig, axes = plt.subplots(6, 6, figsize=(22, 22), sharex=True)
    fig.suptitle(
        f"Per-sample agreement (zoomed {low_q:.0f}-{high_q:.0f} percentile): Target vs Predicted",
        fontsize=14,
        y=1.01,
    )
    for k in range(36):
        r, c = divmod(k, 6)
        ax = axes[r, c]
        tgt = targets_flat[:, k]
        prd = preds_flat[:, k]
        ax.scatter(x, tgt, s=6, color="steelblue", alpha=0.5, label="Target")
        ax.scatter(x, prd, s=6, color="tomato", alpha=0.5, label="Predicted")
        y_all = np.concatenate([tgt, prd])
        y_low, y_high = np.nanpercentile(y_all, [low_q, high_q])
        if np.isfinite(y_low) and np.isfinite(y_high) and y_high > y_low:
            pad = 0.08 * (y_high - y_low)
            ax.set_ylim(y_low - pad, y_high + pad)
        ax.set_title(labels[k], fontsize=8)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("Sample index", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_pred_vs_true(preds_cov, targets_cov, labels, out_path: Path):
    """Scatter plot: predicted vs true for each covariance element with R² metric."""
    preds_flat = preds_cov.reshape(preds_cov.shape[0], -1)
    targets_flat = targets_cov.reshape(targets_cov.shape[0], -1)

    fig, axes = plt.subplots(6, 6, figsize=(22, 22))
    fig.suptitle("Scatter: Predicted vs True covariance values", fontsize=14, y=1.01)
    for k in range(36):
        r, c = divmod(k, 6)
        ax = axes[r, c]
        tgt = targets_flat[:, k]
        prd = preds_flat[:, k]
        # Clip axes to 1st/99th percentile to avoid outlier-dominated scaling
        all_vals = np.concatenate([tgt, prd])
        lo, hi = np.nanpercentile(all_vals, [1, 99])
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        ax_lo, ax_hi = lo - pad, hi + pad
        # Mask for R² computation on inliers only
        inlier = (tgt >= lo) & (tgt <= hi) & (prd >= lo) & (prd <= hi)
        ax.scatter(tgt, prd, s=8, color="steelblue", alpha=0.5)
        ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", linewidth=1, alpha=0.4, label="Perfect agreement")
        ax.set_xlim(ax_lo, ax_hi)
        ax.set_ylim(ax_lo, ax_hi)
        # Compute R² on inlier samples
        ss_res = np.sum((prd[inlier] - tgt[inlier]) ** 2)
        ss_tot = np.sum((tgt[inlier] - tgt[inlier].mean()) ** 2)
        r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        ax.set_title(f"{labels[k]} (R²={r_sq:.3f})", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("True", fontsize=6)
        ax.set_ylabel("Predicted", fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sorted_by_magnitude_overlay(
    preds_cov,
    targets_cov,
    labels,
    out_path: Path,
    max_samples: int = 1000,
):
    """Sort by target magnitude, then overlay predicted; helps visualize systematic bias across range."""
    preds_flat = preds_cov.reshape(preds_cov.shape[0], -1)
    targets_flat = targets_cov.reshape(targets_cov.shape[0], -1)

    n = preds_flat.shape[0]
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=int)
        preds_flat = preds_flat[idx]
        targets_flat = targets_flat[idx]

    fig, axes = plt.subplots(6, 6, figsize=(22, 22), sharex=True)
    fig.suptitle("Sorted by magnitude (true values): Target vs Predicted", fontsize=14, y=1.01)
    for k in range(36):
        r, c = divmod(k, 6)
        ax = axes[r, c]
        tgt = targets_flat[:, k]
        prd = preds_flat[:, k]
        # Sort by true magnitude
        sort_idx = np.argsort(np.abs(tgt))
        tgt_sorted = tgt[sort_idx]
        prd_sorted = prd[sort_idx]
        x = np.arange(len(tgt_sorted))
        ax.plot(x, tgt_sorted, color="steelblue", linewidth=0.8, alpha=0.6, label="Target (sorted)")
        ax.plot(x, prd_sorted, color="tomato", linewidth=0.8, alpha=0.6, label="Predicted")
        ax.set_title(labels[k], fontsize=8)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("Sample index (sorted by |true| magnitude)", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze covariance-output training results (571).")
    parser.add_argument(
        "--model-dir",
        default="model-output-571",
        help="Model output directory (default: model-output-571)",
    )
    parser.add_argument(
        "--test-csv",
        default="dataset-test.csv",
        help="Test CSV path (default: dataset-test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis-output-covariance-571",
        help="Analysis output directory (default: analysis-output-covariance-571)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "--agreement-csv",
        default="dataset-train.csv",
        help="CSV used for per-sample agreement overlay plot (default: dataset-train.csv)",
    )
    parser.add_argument(
        "--agreement-max-samples",
        type=int,
        default=1000,
        help="Max number of samples to show in agreement plot (default: 1000)",
    )
    parser.add_argument(
        "--agreement-zoom-low-q",
        type=float,
        default=5.0,
        help="Lower percentile for zoomed dot plot y-limits (default: 5)",
    )
    parser.add_argument(
        "--agreement-zoom-high-q",
        type=float,
        default=95.0,
        help="Upper percentile for zoomed dot plot y-limits (default: 95)",
    )
    parser.add_argument(
        "--skip-scatter",
        action="store_true",
        help="Skip scatter (pred vs true) plots to save time (default: False)",
    )
    parser.add_argument(
        "--skip-sorted",
        action="store_true",
        help="Skip sorted-by-magnitude plots to save time (default: False)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout rate used during training (default: 0.05; 0 disables)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="elu",
        choices=["elu", "tanh", "relu", "silu"],
        help="Activation function used during training (default: elu)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    print(f"[run] Loading training history from {model_dir}/training_history.csv", flush=True)
    history_df = pd.read_csv(model_dir / "training_history.csv")
    print(f"[run] Loaded {len(history_df)} epochs of training history", flush=True)

    print(f"[run] Loading model and transformers from {model_dir}", flush=True)
    model, input_tr, output_tr, cov_tr, mean_tr, mean_cols = load_model_and_transformers(
        model_dir, dropout=args.dropout, activation=args.activation,
    )
    has_mean = len(mean_cols) > 0
    model.to(device)

    print(f"[run] Loading test data from {args.test_csv}", flush=True)
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    feature_cols = input_tr["feature_cols"]
    target_cols = output_tr["target_cols"]  # chol cols only
    x_mean = input_tr["x_mean"].numpy()
    x_std = input_tr["x_std"].numpy()
    y_mean = output_tr["y_mean"].numpy()
    y_std = output_tr["y_std"].numpy()

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_cols].values.astype(np.float32)
    X_test = (X_test - x_mean) / x_std
    y_test = (y_test - y_mean) / y_std

    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    print("\n[run] Evaluating on test set ...", flush=True)
    metrics = evaluate_on_test(model, test_loader, output_tr, cov_tr, device, has_mean=has_mean)

    print(f"\n[results] Test MAE (overall): {metrics['mae_overall']:.6e}", flush=True)
    print(f"[results] Test RMSE (overall): {metrics['rmse_overall']:.6e}", flush=True)
    print(f"[results] Test RMSE (overall, normalized covariance): {metrics['norm_rmse_overall']:.6f}", flush=True)
    print(f"[results] Test MAPE (overall): {metrics['mape_overall']:.2f}%", flush=True)
    print("\n[results] Per-element MAE / RMSE / MAPE (test set):", flush=True)
    for label, mae, rmse, mape in zip(
        metrics["labels"],
        metrics["mae_per_element"],
        metrics["rmse_per_element"],
        metrics["mape_per_element"],
    ):
        print(f"  {label}: MAE={mae:.6e}  RMSE={rmse:.6e}  MAPE={mape:.2f}%", flush=True)

    metrics_df = pd.DataFrame(
        {
            "target": metrics["labels"],
            "mae": metrics["mae_per_element"],
            "rmse": metrics["rmse_per_element"],
            "rmse_norm": metrics["norm_rmse_per_element"],
            "mape_pct": metrics["mape_per_element"],
        }
    )
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    print(f"\n[run] Test metrics saved to {output_dir}/test_metrics.csv", flush=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(1, len(history_df) + 1)
    ax.plot(epochs, history_df["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, history_df["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized covariance elements)")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=150)
    print(f"[run] Training curve saved to {output_dir}/training_curve.png", flush=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    x_pos = np.arange(len(metrics["labels"]))
    ax.bar(x_pos, metrics["mae_per_element"], alpha=0.7, color="steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics["labels"], rotation=45, ha="right")
    ax.set_ylabel("MAE (covariance units)")
    ax.set_title("Test MAE Per Covariance Element")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "mae_per_element.png", dpi=150)
    print(f"[run] Per-element MAE plot saved to {output_dir}/mae_per_element.png", flush=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    mape_vals = metrics["mape_per_element"]
    colors = ["tomato" if np.isfinite(val) and val > 50 else "steelblue" for val in mape_vals]
    ax.bar(x_pos, mape_vals, alpha=0.8, color=colors)
    ax.axhline(
        y=metrics["mape_overall"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall MAPE = {metrics['mape_overall']:.1f}%",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics["labels"], rotation=45, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Test Percentage Error Per Covariance Element")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "mape_per_element.png", dpi=150)
    print(f"[run] Per-element MAPE plot saved to {output_dir}/mape_per_element.png", flush=True)

    print(f"[run] Loading agreement data from {args.agreement_csv}", flush=True)
    agree_df = pd.read_csv(args.agreement_csv, low_memory=False)
    X_agree = agree_df[feature_cols].values.astype(np.float32)
    y_agree = agree_df[target_cols].values.astype(np.float32)
    X_agree = (X_agree - x_mean) / x_std
    y_agree = (y_agree - y_mean) / y_std
    agree_ds = TensorDataset(torch.from_numpy(X_agree), torch.from_numpy(y_agree))
    agree_loader = DataLoader(agree_ds, batch_size=args.batch_size)

    preds_cov_agree, targets_cov_agree = predict_covariances(model, agree_loader, output_tr, device, has_mean=has_mean)

    # Convert to display units (mm, MeV/c, ps) for all plots
    display_labels = covariance_display_labels()
    preds_display = to_display_units(preds_cov_agree)
    targets_display = to_display_units(targets_cov_agree)

    overlay_path = output_dir / "per_sample_agreement_overlay.png"
    plot_per_sample_overlay(
        preds_display,
        targets_display,
        display_labels,
        overlay_path,
        max_samples=args.agreement_max_samples,
    )
    print(f"[run] Per-sample agreement overlay saved to {overlay_path}", flush=True)

    zoomed_path = output_dir / "per_sample_agreement_zoomed_dots.png"
    plot_per_sample_zoomed_dots(
        preds_display,
        targets_display,
        display_labels,
        zoomed_path,
        max_samples=args.agreement_max_samples,
        low_q=args.agreement_zoom_low_q,
        high_q=args.agreement_zoom_high_q,
    )
    print(f"[run] Per-sample zoomed dot overlay saved to {zoomed_path}", flush=True)

    if not args.skip_scatter:
        scatter_path = output_dir / "scatter_pred_vs_true.png"
        plot_scatter_pred_vs_true(preds_display, targets_display, display_labels, scatter_path)
        print(f"[run] Scatter plot (pred vs true) saved to {scatter_path}", flush=True)

    if not args.skip_sorted:
        sorted_path = output_dir / "sorted_by_magnitude_overlay.png"
        plot_sorted_by_magnitude_overlay(
            preds_display,
            targets_display,
            display_labels,
            sorted_path,
            max_samples=args.agreement_max_samples,
        )
        print(f"[run] Sorted-by-magnitude overlay saved to {sorted_path}", flush=True)

    # ── Covariance element histograms (6×6 grid, predicted vs target) ──────────
    preds_flat = to_display_units(metrics["preds_cov"]).reshape(-1, 36)
    targets_flat = to_display_units(metrics["targets_cov"]).reshape(-1, 36)
    labels = covariance_display_labels()

    fig, axes = plt.subplots(6, 6, figsize=(22, 22))
    fig.suptitle("Covariance Element Distributions: Target vs Predicted (test set)", fontsize=14, y=1.01)
    for idx in range(36):
        row, col = divmod(idx, 6)
        ax = axes[row, col]
        tgt = targets_flat[:, idx]
        prd = preds_flat[:, idx]
        all_vals = np.concatenate([tgt, prd])
        lo, hi = np.nanpercentile(all_vals, 1), np.nanpercentile(all_vals, 99)
        bins = np.linspace(lo, hi, 40)
        ax.hist(tgt, bins=bins, alpha=0.55, color="steelblue", density=True, label="Target")
        ax.hist(prd, bins=bins, alpha=0.55, color="tomato", density=True, label="Predicted")
        ax.set_title(labels[idx], fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "covariance_histograms.png", dpi=120, bbox_inches="tight")
    print(f"[run] Covariance histograms saved to {output_dir}/covariance_histograms.png", flush=True)

    # Also save a version showing only the target distributions as a 6×6 heatmap of std devs
    # (useful to visually confirm block-diagonal structure)
    target_std = targets_flat.std(axis=0).reshape(6, 6)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log10(target_std + 1e-30), cmap="viridis")
    ax.set_title("log10(std) of covariance elements\n(training set target distribution)")
    ax.set_xlabel("Column index j")
    ax.set_ylabel("Row index i")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{target_std[i, j]:.1e}", ha="center", va="center", fontsize=7,
                    color="white" if target_std[i, j] < target_std.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax, label="log10(std)")
    plt.tight_layout()
    plt.savefig(output_dir / "covariance_std_heatmap.png", dpi=150)
    print(f"[run] Covariance std heatmap saved to {output_dir}/covariance_std_heatmap.png", flush=True)

    # ── Mean beam output evaluation (energy, time) ─────────────────────────────
    if has_mean:
        mean_y_mean = mean_tr["mean_y_mean"].numpy()
        mean_y_std = mean_tr["mean_y_std"].numpy()

        all_mean_preds, all_mean_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = model(X_batch.to(device))
                _, pred_mean = output
                all_mean_preds.append(pred_mean.cpu().numpy())
                # Get raw targets from CSV
        test_mean_raw = test_df[mean_cols].values.astype(np.float32)
        preds_mean = np.concatenate(all_mean_preds)

        print("\n[results] Mean beam output metrics (test set):", flush=True)
        mean_metrics_rows = []
        for idx, col in enumerate(mean_cols):
            tgt = test_mean_raw[:, idx]
            prd = preds_mean[:, idx]
            mae = np.abs(prd - tgt).mean()
            rmse = np.sqrt(((prd - tgt) ** 2).mean())
            mape = np.nanmean(np.abs((prd - tgt) / tgt) * 100)
            ss_res = np.sum((prd - tgt) ** 2)
            ss_tot = np.sum((tgt - tgt.mean()) ** 2)
            r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            print(f"  {col}: MAE={mae:.6e}  RMSE={rmse:.6e}  MAPE={mape:.2f}%  R²={r_sq:.4f}", flush=True)
            mean_metrics_rows.append({"target": col, "mae": mae, "rmse": rmse, "mape_pct": mape, "r_squared": r_sq})

        mean_metrics_df = pd.DataFrame(mean_metrics_rows)
        mean_metrics_df.to_csv(output_dir / "mean_beam_metrics.csv", index=False)
        print(f"[run] Mean beam metrics saved to {output_dir}/mean_beam_metrics.csv", flush=True)

        # Scatter plots for mean beam outputs
        n_mean = len(mean_cols)
        ncols = min(n_mean, 3)
        nrows = (n_mean + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()
        for idx, (col, ax) in enumerate(zip(mean_cols, axes_flat)):
            tgt = test_mean_raw[:, idx]
            prd = preds_mean[:, idx]
            ax.scatter(tgt, prd, s=8, color="steelblue", alpha=0.5)
            lo, hi = np.nanpercentile(np.concatenate([tgt, prd]), [1, 99])
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4)
            ss_res = np.sum((prd - tgt) ** 2)
            ss_tot = np.sum((tgt - tgt.mean()) ** 2)
            r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            ax.set_title(f"{col} (R²={r_sq:.4f})")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[n_mean:]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / "mean_beam_scatter.png", dpi=150)
        print(f"[run] Mean beam scatter plot saved to {output_dir}/mean_beam_scatter.png", flush=True)

    print(f"\n[run] Analysis complete. Results in {output_dir}/", flush=True)


if __name__ == "__main__":
    main()