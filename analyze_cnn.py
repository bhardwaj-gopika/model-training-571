"""Evaluate CNN projection predictions: overlay plots, SSIM, covariance recovery.

Loads the trained ProjectionMLP, runs inference on the test set, and produces:
  1. Side-by-side overlay plots (true vs predicted 2D histograms)
  2. Per-projection SSIM scores
  3. Covariance recovery: compute 2x2 covariance from predicted histograms and
     compare to the true covariance sub-blocks (apples-to-apples with the MLP model)

Example:
    python analyze_cnn.py \\
        --test-h5 projections_test.h5 \\
        --model-dir model-output-cnn \\
        --output-dir analysis-cnn \\
        --n-examples 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from train_cnn import ProjectionMLP, ProjectionDataset


PROJECTION_LABELS = ["x-px", "y-py", "t-pz"]
AXIS_LABELS = {
    "x-px": ("x [m]", "px [eV/c]"),
    "y-py": ("y [m]", "py [eV/c]"),
    "t-pz": ("t [s]", "pz [eV/c]"),
}
PHASE_SPACE_VARS = ["x", "px", "y", "py", "t", "pz"]


def compute_ssim_2d(pred: np.ndarray, target: np.ndarray,
                    C1=1e-8, C2=1e-8) -> float:
    """Simple SSIM between two 2D arrays (no windowing, global stats)."""
    mu_p = pred.mean()
    mu_t = target.mean()
    sigma_p = pred.std()
    sigma_t = target.std()
    sigma_pt = ((pred - mu_p) * (target - mu_t)).mean()

    ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
           ((mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2))
    return float(ssim)


def recover_covariance_2x2(hist: np.ndarray, edges_a: np.ndarray,
                           edges_b: np.ndarray) -> np.ndarray:
    """Compute 2x2 covariance from a 2D histogram (weighted by bin centers).

    hist: (N_bins, N_bins), normalized to sum=1
    edges_a: (N_bins+1,) bin edges for axis 0
    edges_b: (N_bins+1,) bin edges for axis 1

    Returns (2, 2) covariance matrix.
    """
    centers_a = 0.5 * (edges_a[:-1] + edges_a[1:])
    centers_b = 0.5 * (edges_b[:-1] + edges_b[1:])

    # Meshgrid of bin centers
    A, B = np.meshgrid(centers_a, centers_b, indexing="ij")

    # Weighted mean
    mean_a = (A * hist).sum()
    mean_b = (B * hist).sum()

    # Weighted covariance
    var_a = ((A - mean_a)**2 * hist).sum()
    var_b = ((B - mean_b)**2 * hist).sum()
    cov_ab = ((A - mean_a) * (B - mean_b) * hist).sum()

    return np.array([[var_a, cov_ab], [cov_ab, var_b]])


def plot_overlay_grid(true_projs, pred_projs, bin_edges, output_path,
                      n_examples=16):
    """Plot grid of true vs predicted projections."""
    n_examples = min(n_examples, len(true_projs))
    n_cols = 3  # one per projection pair
    n_rows = n_examples

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(4 * n_cols * 2, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_rows):
        for col, (proj_name, (var_a, var_b)) in enumerate(
                zip(PROJECTION_LABELS, [("x", "px"), ("y", "py"), ("t", "pz")])):
            edges_a = bin_edges[var_a]
            edges_b = bin_edges[var_b]

            true_img = true_projs[row, col]
            pred_img = pred_projs[row, col]

            # True
            ax_true = axes[row, col * 2]
            ax_true.imshow(true_img.T, origin="lower", aspect="auto",
                           extent=[edges_a[0], edges_a[-1], edges_b[0], edges_b[-1]],
                           cmap="viridis")
            if row == 0:
                ax_true.set_title(f"True {proj_name}")
            ax_true.set_xlabel(AXIS_LABELS[proj_name][0])
            ax_true.set_ylabel(AXIS_LABELS[proj_name][1])

            # Predicted
            ax_pred = axes[row, col * 2 + 1]
            ax_pred.imshow(pred_img.T, origin="lower", aspect="auto",
                           extent=[edges_a[0], edges_a[-1], edges_b[0], edges_b[-1]],
                           cmap="viridis")
            if row == 0:
                ax_pred.set_title(f"Pred {proj_name}")
            ax_pred.set_xlabel(AXIS_LABELS[proj_name][0])
            ax_pred.set_ylabel(AXIS_LABELS[proj_name][1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Overlay grid saved to {output_path}", flush=True)


def plot_contour_overlay(true_projs, pred_projs, bin_edges, output_path,
                         n_examples=8):
    """Plot contour overlays (true=filled, pred=contour lines)."""
    n_examples = min(n_examples, len(true_projs))

    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_examples):
        for col, (proj_name, (var_a, var_b)) in enumerate(
                zip(PROJECTION_LABELS, [("x", "px"), ("y", "py"), ("t", "pz")])):
            ax = axes[row, col]
            edges_a = bin_edges[var_a]
            edges_b = bin_edges[var_b]
            centers_a = 0.5 * (edges_a[:-1] + edges_a[1:])
            centers_b = 0.5 * (edges_b[:-1] + edges_b[1:])

            true_img = true_projs[row, col]
            pred_img = pred_projs[row, col]

            # True as filled contours
            ax.contourf(centers_a, centers_b, true_img.T, levels=10,
                        cmap="Blues", alpha=0.6)
            # Predicted as contour lines
            ax.contour(centers_a, centers_b, pred_img.T, levels=10,
                       colors="red", linewidths=0.8, alpha=0.8)

            if row == 0:
                ax.set_title(f"{proj_name} (blue=true, red=pred)")
            ax.set_xlabel(AXIS_LABELS[proj_name][0])
            ax.set_ylabel(AXIS_LABELS[proj_name][1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Contour overlay saved to {output_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-h5", required=True, help="Test HDF5 dataset")
    ap.add_argument("--model-dir", required=True, help="Directory with model_cnn.pt")
    ap.add_argument("--output-dir", default="analysis-cnn")
    ap.add_argument("--n-examples", type=int, default=16,
                    help="Number of example shots for overlay plots")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    input_tf = torch.load(model_dir / "input_transformers.pt", weights_only=False)
    x_mean = input_tf["x_mean"].numpy()
    x_std = input_tf["x_std"].numpy()
    n_inputs = len(x_mean)

    bin_edges = torch.load(model_dir / "bin_edges.pt", weights_only=False)

    model = ProjectionMLP(n_inputs=n_inputs)
    model.load_state_dict(torch.load(model_dir / "model_cnn.pt", weights_only=True))
    model.to(device)
    model.eval()

    # Load test data
    test_ds = ProjectionDataset(args.test_h5, x_mean, x_std)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)

    print(f"[run] Test set: {len(test_ds)} samples", flush=True)
    print(f"[run] Running inference ...", flush=True)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds)     # (N, 3, 64, 64)
    targets = np.concatenate(all_targets)  # (N, 3, 64, 64)
    n_test = preds.shape[0]

    # ── SSIM per projection ───────────────────────────────────────────────────
    print("\n[run] Computing SSIM per projection ...", flush=True)
    ssim_scores = {proj: [] for proj in PROJECTION_LABELS}
    for i in range(n_test):
        for j, proj_name in enumerate(PROJECTION_LABELS):
            s = compute_ssim_2d(preds[i, j], targets[i, j])
            ssim_scores[proj_name].append(s)

    print("[results] SSIM (mean +/- std):")
    for proj_name in PROJECTION_LABELS:
        scores = np.array(ssim_scores[proj_name])
        print(f"  {proj_name}: {scores.mean():.4f} +/- {scores.std():.4f}"
              f"  (median={np.median(scores):.4f})")

    # ── MSE per projection ────────────────────────────────────────────────────
    print("\n[run] Computing MSE per projection ...", flush=True)
    for j, proj_name in enumerate(PROJECTION_LABELS):
        mse = ((preds[:, j] - targets[:, j])**2).mean()
        print(f"  {proj_name}: MSE={mse:.8f}")

    # ── Covariance recovery ───────────────────────────────────────────────────
    print("\n[run] Computing covariance recovery ...", flush=True)
    proj_pairs = [("x", "px"), ("y", "py"), ("t", "pz")]
    cov_results = {proj: {"true": [], "pred": []} for proj in PROJECTION_LABELS}

    for i in range(n_test):
        for j, (proj_name, (var_a, var_b)) in enumerate(
                zip(PROJECTION_LABELS, proj_pairs)):
            edges_a = bin_edges[var_a]
            edges_b = bin_edges[var_b]
            cov_true = recover_covariance_2x2(targets[i, j], edges_a, edges_b)
            cov_pred = recover_covariance_2x2(preds[i, j], edges_a, edges_b)
            cov_results[proj_name]["true"].append(cov_true)
            cov_results[proj_name]["pred"].append(cov_pred)

    print("[results] Covariance recovery (MAE of 2x2 sub-blocks):")
    for proj_name in PROJECTION_LABELS:
        true_arr = np.array(cov_results[proj_name]["true"])  # (N, 2, 2)
        pred_arr = np.array(cov_results[proj_name]["pred"])
        mae = np.abs(true_arr - pred_arr).mean(axis=0)
        print(f"  {proj_name}:")
        print(f"    var_a  MAE: {mae[0,0]:.6e}")
        print(f"    var_b  MAE: {mae[1,1]:.6e}")
        print(f"    cov_ab MAE: {mae[0,1]:.6e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[run] Generating plots ...", flush=True)

    # Random subset for overlay plots
    rng = np.random.default_rng(42)
    example_idx = rng.choice(n_test, size=min(args.n_examples, n_test), replace=False)
    example_idx.sort()

    plot_overlay_grid(
        targets[example_idx], preds[example_idx], bin_edges,
        output_dir / "projection_grid.png", n_examples=len(example_idx))

    plot_contour_overlay(
        targets[example_idx], preds[example_idx], bin_edges,
        output_dir / "contour_overlay.png", n_examples=min(8, len(example_idx)))

    # SSIM histogram
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for j, proj_name in enumerate(PROJECTION_LABELS):
        axes[j].hist(ssim_scores[proj_name], bins=50, alpha=0.7)
        axes[j].set_title(f"SSIM: {proj_name}")
        axes[j].set_xlabel("SSIM")
        axes[j].axvline(np.mean(ssim_scores[proj_name]), color="red",
                        linestyle="--", label=f"mean={np.mean(ssim_scores[proj_name]):.3f}")
        axes[j].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ssim_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Covariance scatter (recovered from CNN vs true)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, proj_name in enumerate(PROJECTION_LABELS):
        true_diag = np.array([c[0, 0] for c in cov_results[proj_name]["true"]])
        pred_diag = np.array([c[0, 0] for c in cov_results[proj_name]["pred"]])
        axes[j].scatter(true_diag, pred_diag, alpha=0.3, s=5)
        lims = [min(true_diag.min(), pred_diag.min()),
                max(true_diag.max(), pred_diag.max())]
        axes[j].plot(lims, lims, "r--", linewidth=1)
        axes[j].set_xlabel(f"True var ({proj_name.split('-')[0]})")
        axes[j].set_ylabel(f"Pred var ({proj_name.split('-')[0]})")
        axes[j].set_title(f"Covariance recovery: {proj_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "covariance_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save metrics to CSV
    import pandas as pd
    metrics_rows = []
    for proj_name in PROJECTION_LABELS:
        scores = np.array(ssim_scores[proj_name])
        true_arr = np.array(cov_results[proj_name]["true"])
        pred_arr = np.array(cov_results[proj_name]["pred"])
        mae = np.abs(true_arr - pred_arr).mean(axis=0)
        metrics_rows.append({
            "projection": proj_name,
            "ssim_mean": scores.mean(),
            "ssim_std": scores.std(),
            "ssim_median": np.median(scores),
            "mse": ((preds[:, PROJECTION_LABELS.index(proj_name)] -
                     targets[:, PROJECTION_LABELS.index(proj_name)])**2).mean(),
            "cov_var_a_mae": mae[0, 0],
            "cov_var_b_mae": mae[1, 1],
            "cov_ab_mae": mae[0, 1],
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    print(f"\n[done] Metrics saved to {output_dir}/test_metrics.csv", flush=True)
    print(f"[done] All outputs in {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
