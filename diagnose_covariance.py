"""Per-element covariance diagnostics: R², median relative error, and worst-sample analysis.

Usage:
    python diagnose_covariance.py
    python diagnose_covariance.py --true-cov compare-4/true_covariances.npy --pred-cov compare-4/pred_covariances.npy
    python diagnose_covariance.py --output-dir diag-output
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = ["x", "px", "y", "py", "t", "pz"]
N = len(LABELS)


def build_parser():
    p = argparse.ArgumentParser(description="Per-element covariance diagnostics.")
    p.add_argument("--true-cov", default="compare-4/true_covariances.npy",
                   help="Path to true covariances .npy (N, 6, 6)")
    p.add_argument("--pred-cov", default="compare-4/pred_covariances.npy",
                   help="Path to predicted covariances .npy (N, 6, 6)")
    p.add_argument("--output-dir", default="diag-output",
                   help="Output directory for plots and summary (default: diag-output)")
    p.add_argument("--worst-k", type=int, default=5,
                   help="Number of worst samples to list per element (default: 5)")
    return p


def r_squared(true, pred):
    """Compute R² for a 1D array of values."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def median_relative_error(true, pred):
    """Median absolute relative error, skipping near-zero true values."""
    mask = np.abs(true) > 0
    if not np.any(mask):
        return float("nan")
    return float(np.median(np.abs((pred[mask] - true[mask]) / true[mask])))


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_cov = np.load(args.true_cov)  # (N_samples, 6, 6)
    pred_cov = np.load(args.pred_cov)
    n_samples = true_cov.shape[0]
    print(f"[run] Loaded {n_samples} samples", flush=True)

    # ── Per-element R² and median relative error (upper triangle + diagonal) ──
    r2_matrix = np.full((N, N), np.nan)
    mre_matrix = np.full((N, N), np.nan)

    print()
    print(f"{'Element':<12s} {'R²':>8s} {'Med Rel Err':>12s} {'Mean True':>12s} {'Std True':>12s}")
    print("-" * 60)

    for i in range(N):
        for j in range(i, N):
            true_ij = true_cov[:, i, j]
            pred_ij = pred_cov[:, i, j]
            r2 = r_squared(true_ij, pred_ij)
            mre = median_relative_error(true_ij, pred_ij)
            r2_matrix[i, j] = r2
            r2_matrix[j, i] = r2
            mre_matrix[i, j] = mre
            mre_matrix[j, i] = mre

            label = f"C({LABELS[i]},{LABELS[j]})"
            print(f"{label:<12s} {r2:>8.4f} {mre:>12.4f} {np.mean(true_ij):>12.4e} {np.std(true_ij):>12.4e}")

    # ── Heatmap: R² matrix ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    im1 = ax1.imshow(r2_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax1.set_xticks(range(N))
    ax1.set_xticklabels(LABELS)
    ax1.set_yticks(range(N))
    ax1.set_yticklabels(LABELS)
    ax1.set_title("Per-Element R²")
    for i in range(N):
        for j in range(N):
            val = r2_matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax1.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # Heatmap: median relative error
    mre_display = np.clip(mre_matrix, 0, 10)  # cap display at 1000%
    im2 = ax2.imshow(mre_display, cmap="Reds", vmin=0, vmax=2)
    ax2.set_xticks(range(N))
    ax2.set_xticklabels(LABELS)
    ax2.set_yticks(range(N))
    ax2.set_yticklabels(LABELS)
    ax2.set_title("Per-Element Median |Relative Error|")
    for i in range(N):
        for j in range(N):
            val = mre_matrix[i, j]
            color = "white" if val > 1.0 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Relative error (clipped at 2)")

    plt.tight_layout()
    fig.savefig(output_dir / "element_r2_mre_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[run] Saved: {output_dir / 'element_r2_mre_heatmaps.png'}", flush=True)

    # ── Per-element scatter plots (upper triangle) ──
    n_elements = N * (N + 1) // 2  # 21
    ncols = 7
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    idx = 0
    for i in range(N):
        for j in range(i, N):
            ax = axes_flat[idx]
            true_ij = true_cov[:, i, j]
            pred_ij = pred_cov[:, i, j]
            ax.scatter(true_ij, pred_ij, s=1, alpha=0.3, rasterized=True)
            lo = min(true_ij.min(), pred_ij.min())
            hi = max(true_ij.max(), pred_ij.max())
            margin = (hi - lo) * 0.05 if hi != lo else 1.0
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "r--", linewidth=0.8, alpha=0.7)
            ax.set_title(f"C({LABELS[i]},{LABELS[j]})  R²={r2_matrix[i,j]:.3f}", fontsize=8)
            ax.tick_params(labelsize=6)
            idx += 1

    # Hide unused axes
    for k in range(idx, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle("Per-Element Covariance: True vs Predicted", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "element_scatter_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[run] Saved: {output_dir / 'element_scatter_grid.png'}", flush=True)

    # ── Worst samples per diagonal element ──
    print(f"\n{'='*70}")
    print(f"Worst {args.worst_k} samples per diagonal element (by |relative error|):")
    print(f"{'='*70}")

    for i in range(N):
        true_diag = true_cov[:, i, i]
        pred_diag = pred_cov[:, i, i]
        mask = np.abs(true_diag) > 0
        rel_err = np.full(n_samples, np.nan)
        rel_err[mask] = np.abs((pred_diag[mask] - true_diag[mask]) / true_diag[mask])
        worst_idx = np.argsort(rel_err)[::-1][:args.worst_k]

        print(f"\n  σ²_{LABELS[i]}:")
        for rank, si in enumerate(worst_idx):
            print(f"    #{rank+1} sample {si:5d}: true={true_diag[si]:.4e}  "
                  f"pred={pred_diag[si]:.4e}  rel_err={rel_err[si]:.2%}")

    # ── Summary: diagonal variance ratio distributions ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for k, (i, ax) in enumerate(zip(range(N), axes.flatten())):
        true_diag = true_cov[:, i, i]
        pred_diag = pred_cov[:, i, i]
        mask = true_diag > 0
        ratios = pred_diag[mask] / true_diag[mask]
        ax.hist(ratios, bins=50, edgecolor="black", linewidth=0.5, alpha=0.7)
        ax.axvline(1.0, color="red", linewidth=1.5, linestyle="--", label="Perfect")
        ax.axvline(np.median(ratios), color="blue", linewidth=1.2, linestyle="-",
                   label=f"Median={np.median(ratios):.3f}")
        ax.set_xlabel(f"pred/true ratio")
        ax.set_title(f"σ²_{LABELS[i]}  (median ratio={np.median(ratios):.3f})")
        ax.legend(fontsize=8)

    plt.suptitle("Diagonal Variance Ratio Distributions (pred/true)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "variance_ratio_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[run] Saved: {output_dir / 'variance_ratio_histograms.png'}", flush=True)

    print(f"\n[run] Done. All outputs in {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
