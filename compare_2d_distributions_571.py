"""2D distribution plot comparison for the 571 model in machine input units.

Converts sim parameters to machine PV units, evaluates the covariance matrix
via lume-torch, and compares against ground-truth covariance from the
bmad_final_particles OpenPMD beam distributions.

Usage:
    python compare_2d_distributions_571.py
    python compare_2d_distributions_571.py --dump-csv particles-571.csv --max-samples 200
    python compare_2d_distributions_571.py --sample-index 42
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from beamphysics import ParticleGroup
from lume_torch.models import TorchModel, TorchModule

from pv_mapping import (
    sim_to_machine_array,
    machine_input_names,
    ordered_pv_mapping,
)


PHASE_SPACE_LABELS = ["x", "px", "y", "py", "t", "pz"]
PHASE_SPACE_UNITS = ["m", "rad", "m", "rad", "s", "eV/c"]

# M-normalization matrix: brings (x, px, y, py, t, pz) to comparable scales
M_DIAG = np.array([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6])
M = np.diag(M_DIAG)


def load_lume_model(yaml_path: str) -> TorchModule:
    torch_model = TorchModel(yaml_path)
    return TorchModule(model=torch_model)


def gt_covariance_from_h5(h5_path: str, normalize: bool = False) -> np.ndarray:
    beam = ParticleGroup(h5=h5_path)
    cov = np.asarray(beam.cov("x", "px", "y", "py", "t", "pz"), dtype=float)
    if normalize:
        cov = M @ cov @ M.T
    return cov


def draw_2d_ellipse(ax, cov_2x2, mean=(0, 0), n_std=2.0, **kwargs):
    """Draw a 2-sigma ellipse from a 2x2 covariance sub-matrix."""
    vals, vecs = np.linalg.eigh(cov_2x2)
    vals = np.maximum(vals, 0)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def plot_2d_distribution_comparison(
    true_cov: np.ndarray,
    pred_cov: np.ndarray,
    sample_label: str,
    machine_inputs: dict,
    output_path: Path,
    n_std: float = 2.0,
):
    """Plot 2D phase-space ellipse comparisons for all unique pairs."""
    n = 6
    fig, axes = plt.subplots(n, n, figsize=(22, 20))
    fig.suptitle(
        f"2D Phase-Space Distribution Comparison (2σ ellipses)\n{sample_label}",
        fontsize=13,
        y=0.98,
    )

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Diagonal: show 1D variance comparison
                true_sigma = np.sqrt(max(true_cov[i, i], 0))
                pred_sigma = np.sqrt(max(pred_cov[i, i], 0))
                ax.barh(
                    ["Predicted", "True"],
                    [pred_sigma, true_sigma],
                    color=["tab:orange", "tab:blue"],
                    alpha=0.7,
                )
                ax.set_title(f"σ({PHASE_SPACE_LABELS[i]})", fontsize=8)
                ax.tick_params(labelsize=6)
            elif i > j:
                # Lower triangle: 2D ellipses
                idx = [j, i]
                true_sub = true_cov[np.ix_(idx, idx)]
                pred_sub = pred_cov[np.ix_(idx, idx)]

                draw_2d_ellipse(
                    ax, true_sub, n_std=n_std,
                    fill=False, edgecolor="tab:blue", linewidth=1.5, label="True",
                )
                draw_2d_ellipse(
                    ax, pred_sub, n_std=n_std,
                    fill=False, edgecolor="tab:orange", linewidth=1.5,
                    linestyle="--", label="Predicted",
                )

                # Auto-scale
                max_extent = max(
                    n_std * np.sqrt(max(true_sub[0, 0], 0)),
                    n_std * np.sqrt(max(true_sub[1, 1], 0)),
                    n_std * np.sqrt(max(pred_sub[0, 0], 0)),
                    n_std * np.sqrt(max(pred_sub[1, 1], 0)),
                ) * 1.3
                if max_extent > 0:
                    ax.set_xlim(-max_extent, max_extent)
                    ax.set_ylim(-max_extent, max_extent)
                ax.set_aspect("equal")
                ax.axhline(0, color="gray", linewidth=0.3)
                ax.axvline(0, color="gray", linewidth=0.3)

                if i == n - 1:
                    ax.set_xlabel(
                        f"{PHASE_SPACE_LABELS[j]} [{PHASE_SPACE_UNITS[j]}]", fontsize=7,
                    )
                if j == 0:
                    ax.set_ylabel(
                        f"{PHASE_SPACE_LABELS[i]} [{PHASE_SPACE_UNITS[i]}]", fontsize=7,
                    )
                ax.tick_params(labelsize=5)
                if i == 1 and j == 0:
                    ax.legend(fontsize=6, loc="upper right")
            else:
                # Upper triangle: covariance element value comparison
                true_val = true_cov[i, j]
                pred_val = pred_cov[i, j]
                rel_err = (
                    abs(pred_val - true_val) / abs(true_val) * 100
                    if abs(true_val) > 1e-30
                    else float("nan")
                )
                text = (
                    f"cov({PHASE_SPACE_LABELS[i]},{PHASE_SPACE_LABELS[j]})\n"
                    f"True:  {true_val:.4e}\n"
                    f"Pred:  {pred_val:.4e}\n"
                    f"Err:   {rel_err:.1f}%"
                )
                ax.text(
                    0.5, 0.5, text,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=6, family="monospace",
                )
                ax.set_xticks([])
                ax.set_yticks([])

    # Add machine input annotation
    input_text = "Machine Inputs:\n" + "\n".join(
        f"  {k}: {v:.6g}" for k, v in machine_inputs.items()
    )
    fig.text(
        0.01, 0.01, input_text,
        fontsize=6, family="monospace", verticalalignment="bottom",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_covariance_scatter_grid(
    true_covs: np.ndarray,
    pred_covs: np.ndarray,
    output_path: Path,
):
    """Scatter plots of true vs predicted for all 36 covariance elements."""
    true_flat = true_covs.reshape(-1, 36)
    pred_flat = pred_covs.reshape(-1, 36)

    fig, axes = plt.subplots(6, 6, figsize=(24, 22))
    fig.suptitle(
        "True vs Predicted Covariance Elements (Machine-Unit Inputs, 571 Model)",
        fontsize=14,
    )

    for idx in range(36):
        row, col = idx // 6, idx % 6
        ax = axes[row][col]
        t = true_flat[:, idx]
        p = pred_flat[:, idx]

        ax.scatter(t, p, s=4, alpha=0.4, rasterized=True)

        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8)

        r2 = float(np.corrcoef(t, p)[0, 1] ** 2) if t.std() > 0 else float("nan")
        label = f"cov({PHASE_SPACE_LABELS[row]},{PHASE_SPACE_LABELS[col]})"
        ax.set_title(f"{label}\n$R^2$={r2:.3f}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("True", fontsize=5)
        ax.set_ylabel("Predicted", fontsize=5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_covariance_heatmap_comparison(
    true_cov: np.ndarray,
    pred_cov: np.ndarray,
    sample_label: str,
    output_path: Path,
):
    """Side-by-side heatmaps of true vs predicted covariance matrices."""
    diff = pred_cov - true_cov
    vmax = max(np.abs(true_cov).max(), np.abs(pred_cov).max())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Covariance Matrix Comparison — {sample_label}", fontsize=12)

    for ax, data, title, cmap in [
        (ax1, true_cov, "True (Particles)", "RdBu_r"),
        (ax2, pred_cov, "Predicted (Model)", "RdBu_r"),
        (ax3, diff, "Difference (Pred - True)", "coolwarm"),
    ]:
        if title.startswith("Diff"):
            im = ax.imshow(data, cmap=cmap, vmin=-vmax * 0.3, vmax=vmax * 0.3)
        else:
            im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(PHASE_SPACE_LABELS, fontsize=8)
        ax.set_yticklabels(PHASE_SPACE_LABELS, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Compare 2D phase-space distributions from the 571 model "
            "(machine-unit inputs via lume-torch) against particle ground truth."
        )
    )
    p.add_argument(
        "--lume-yaml", default="lumetorchyaml-machine/injector_machine.yaml",
        help="Path to lume-torch YAML config (default: lumetorchyaml-machine/injector_machine.yaml)",
    )
    p.add_argument(
        "--dump-csv", default="particles-571.csv",
        help="CSV with bmad_final_particles h5 paths and sim input columns (default: particles-571.csv)",
    )
    p.add_argument(
        "--particles-column", default="bmad_final_particles",
        help="Column containing OpenPMD .h5 file paths (default: bmad_final_particles)",
    )
    p.add_argument(
        "--output-dir", default="compare-2d-571",
        help="Output directory for plots (default: compare-2d-571)",
    )
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="Inference batch size (default: 256)",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples (default: all valid rows)",
    )
    p.add_argument(
        "--sample-indices", type=int, nargs="+", default=None,
        help="Specific sample indices for per-sample ellipse plots (default: first 5)",
    )
    p.add_argument(
        "--n-ellipse-samples", type=int, default=5,
        help="Number of per-sample ellipse plots when --sample-indices is not set (default: 5)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subsampling (default: 42)",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Apply M-normalization C_norm = M @ C @ M^T with "
            "M = diag(1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6) to ground-truth covariance. "
            "Must match the normalization used during training."
        ),
    )
    return p


def main():
    args = build_parser().parse_args()

    dump_csv = Path(args.dump_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    particles_col = args.particles_column

    # ── Load lume-torch model ──────────────────────────────────────────────────
    print(f"[run] Loading lume-torch model from {args.lume_yaml}", flush=True)
    lume_model = load_lume_model(args.lume_yaml)

    # Extract input PV names from the lume-torch model config
    lume_torch_model = TorchModel(args.lume_yaml)
    pv_cols = lume_torch_model.input_names

    # Map PV names back to sim feature columns via the pv_mapping
    from pv_mapping import PV_MAPPING_BY_SIM_PARAM
    pv_to_sim = {}
    for sim_param, spec in PV_MAPPING_BY_SIM_PARAM.items():
        pv_name = spec["experimental_pv"] or sim_param
        pv_to_sim[pv_name] = sim_param
    feature_cols = [pv_to_sim[pv] for pv in pv_cols]

    print(f"[run] Machine PV inputs ({len(pv_cols)}): {pv_cols}", flush=True)
    print(f"[run] Sim feature columns: {feature_cols}", flush=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"[run] Reading dump CSV: {dump_csv}", flush=True)
    df = pd.read_csv(dump_csv, low_memory=False)

    has_h5 = (
        df[particles_col].notna()
        & df[particles_col].astype(str).str.endswith(".h5")
    )
    has_inputs = df[feature_cols].notna().all(axis=1)
    if "impact_error" in df.columns:
        no_error = df["impact_error"].isna() | (
            df["impact_error"].astype(str).str.strip() == ""
        )
        valid_mask = has_h5 & has_inputs & no_error
    else:
        valid_mask = has_h5 & has_inputs

    df_valid = df[valid_mask].reset_index(drop=True)
    print(f"[run] {len(df_valid)} valid rows (out of {len(df)} total)", flush=True)

    if len(df_valid) == 0:
        raise SystemExit("No valid rows found.")

    if args.max_samples is not None and args.max_samples < len(df_valid):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(df_valid), size=args.max_samples, replace=False)
        df_valid = df_valid.iloc[sorted(idx)].reset_index(drop=True)
        print(f"[run] Subsampled to {len(df_valid)} rows", flush=True)

    # ── Convert sim → machine units ────────────────────────────────────────────
    X_sim = df_valid[feature_cols].values.astype(np.float32)
    X_machine = sim_to_machine_array(X_sim, feature_cols)
    print("[run] Converted sim parameters → machine PV units", flush=True)

    machine_df = pd.DataFrame(X_machine, columns=pv_cols)
    machine_df.to_csv(output_dir / "machine_inputs.csv", index=False)

    # ── Lume-torch inference (machine inputs) ──────────────────────────────────
    print(f"[run] Running lume-torch inference on {len(df_valid)} rows ...", flush=True)
    X_machine_tensor = torch.from_numpy(X_machine)
    loader = DataLoader(
        TensorDataset(X_machine_tensor), batch_size=args.batch_size,
    )
    pred_batches = []
    with torch.no_grad():
        for (X_batch,) in loader:
            pred_cov = lume_model(X_batch.to(device)).cpu().numpy()
            pred_batches.append(pred_cov)

    pred_covs = np.concatenate(pred_batches, axis=0)  # (N, 6, 6)
    print(f"[run] Predicted covariance shape: {pred_covs.shape}", flush=True)

    # ── Ground truth from bmad_final_particles h5 files ───────────────────────
    print(f"[run] Loading ground-truth covariances from {particles_col} h5 files ...", flush=True)
    h5_paths = df_valid[particles_col].tolist()
    true_covs_list = []
    valid_indices = []
    for i, h5_path in enumerate(h5_paths):
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(h5_paths)}", flush=True)
        try:
            cov = gt_covariance_from_h5(h5_path, normalize=args.normalize)
            true_covs_list.append(cov)
            valid_indices.append(i)
        except Exception as exc:
            print(f"  [warn] Failed {h5_path}: {exc}")

    print(
        f"[run] Loaded {len(valid_indices)} ground-truth covariances "
        f"({len(h5_paths) - len(valid_indices)} failed)",
        flush=True,
    )

    true_covs = np.stack(true_covs_list)  # (N_valid, 6, 6)
    pred_covs_valid = pred_covs[valid_indices]
    X_machine_valid = X_machine[valid_indices]

    # ── Save arrays ────────────────────────────────────────────────────────────
    np.save(output_dir / "true_covariances.npy", true_covs)
    np.save(output_dir / "pred_covariances.npy", pred_covs_valid)

    # ── Scatter grid (all samples) ─────────────────────────────────────────────
    print("[run] Generating scatter grid ...", flush=True)
    scatter_path = output_dir / "scatter_grid.png"
    plot_covariance_scatter_grid(true_covs, pred_covs_valid, scatter_path)
    print(f"[plot] Saved: {scatter_path}", flush=True)

    # ── Per-sample 2D ellipse comparisons ──────────────────────────────────────
    if args.sample_indices is not None:
        ellipse_indices = [
            i for i in args.sample_indices if i < len(valid_indices)
        ]
    else:
        n_plot = min(args.n_ellipse_samples, len(valid_indices))
        rng = np.random.default_rng(args.seed + 1)
        ellipse_indices = sorted(rng.choice(len(valid_indices), size=n_plot, replace=False))

    ellipse_dir = output_dir / "ellipse_comparisons"
    ellipse_dir.mkdir(exist_ok=True)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    for local_idx in ellipse_indices:
        machine_inputs = {
            pv_cols[k]: float(X_machine_valid[local_idx, k])
            for k in range(len(pv_cols))
        }
        sample_label = f"Sample {local_idx}"

        plot_2d_distribution_comparison(
            true_covs[local_idx],
            pred_covs_valid[local_idx],
            sample_label,
            machine_inputs,
            ellipse_dir / f"ellipse_sample_{local_idx:04d}.png",
        )

        plot_covariance_heatmap_comparison(
            true_covs[local_idx],
            pred_covs_valid[local_idx],
            sample_label,
            heatmap_dir / f"heatmap_sample_{local_idx:04d}.png",
        )

    print(f"[plot] Saved {len(ellipse_indices)} ellipse plots to {ellipse_dir}", flush=True)
    print(f"[plot] Saved {len(ellipse_indices)} heatmap plots to {heatmap_dir}", flush=True)

    # ── Summary statistics ─────────────────────────────────────────────────────
    true_flat = true_covs.reshape(-1, 36)
    pred_flat = pred_covs_valid.reshape(-1, 36)

    print("\n[summary] Per-element metrics:")
    header = f"{'Element':<20}  {'R²':>8}  {'MAE':>14}  {'RMSE':>14}"
    print(header)
    print("-" * len(header))
    for idx in range(36):
        row, col = idx // 6, idx % 6
        label = f"cov({PHASE_SPACE_LABELS[row]},{PHASE_SPACE_LABELS[col]})"
        t = true_flat[:, idx]
        p = pred_flat[:, idx]
        r2 = float(np.corrcoef(t, p)[0, 1] ** 2) if t.std() > 0 else float("nan")
        mae = float(np.mean(np.abs(t - p)))
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        print(f"{label:<20}  {r2:>8.4f}  {mae:>14.4e}  {rmse:>14.4e}")

    print(f"\n[run] All outputs saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
