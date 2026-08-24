"""Evaluate the trained normalizing flow: sample beams, compute covariance, overlay plots.

For each test shot:
  1. Sample N particles from the flow conditioned on the 18 inputs
  2. Compute 6x6 covariance of samples and compare to true
  3. Compute 2D projections and overlay true vs sampled
  4. Report per-element MAE/MAPE (same format as analyze_covariance2.py)

Example:
    python analyze_flow.py \
        --test-h5 particles_test.h5 \
        --model-dir model-output-flow \
        --output-dir analysis-flow \
        --n-samples 10000 --n-examples 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import zuko
except ImportError:
    raise ImportError("zuko is required: pip install zuko")


PHASE_SPACE_VARS = ["x", "px", "y", "py", "t", "pz"]
PROJECTION_PAIRS = [("x", "px"), ("y", "py"), ("t", "pz")]
PROJECTION_LABELS = ["x-px", "y-py", "t-pz"]


def load_flow(model_dir: Path, device):
    """Load flow model and transformers."""
    with open(model_dir / "flow_config.json") as f:
        config = json.load(f)

    flow = zuko.flows.NSF(
        features=config["features"],
        context=config["context"],
        transforms=config["transforms"],
        hidden_features=config["hidden_features"],
        bins=config["bins"],
    )
    flow.load_state_dict(torch.load(model_dir / "flow_model.pt", weights_only=True))
    flow.to(device)
    flow.eval()

    input_tf = torch.load(model_dir / "input_transformers.pt", weights_only=False)
    particle_tf = torch.load(model_dir / "particle_transformers.pt", weights_only=False)

    return flow, config, input_tf, particle_tf


@torch.no_grad()
def sample_from_flow(flow, inputs_norm, n_samples, device, particle_tf):
    """Sample particles from the flow for a batch of inputs.

    Args:
        inputs_norm: (B, 18) normalized input tensor
        n_samples: particles to sample per shot

    Returns:
        (B, n_samples, 6) numpy array in original (unnormalized) coordinates
    """
    B = inputs_norm.shape[0]
    # Expand context: (B, 18) -> (B * n_samples, 18)
    context = inputs_norm.to(device).unsqueeze(1).expand(B, n_samples, -1)
    context = context.reshape(B * n_samples, -1)

    # Sample
    samples_norm = flow(context).sample()  # (B * n_samples, 6)
    samples_norm = samples_norm.reshape(B, n_samples, 6).cpu().numpy()

    # Denormalize
    p_mean = particle_tf["p_mean"].numpy()
    p_std = particle_tf["p_std"].numpy()
    samples = samples_norm * p_std + p_mean

    return samples


def compute_covariance_6x6(particles: np.ndarray) -> np.ndarray:
    """Compute 6x6 covariance from particles (N, 6)."""
    valid = ~np.isnan(particles).any(axis=1)
    particles = particles[valid]
    if len(particles) < 10:
        return np.full((6, 6), np.nan)
    return np.cov(particles.T)


def compute_means(particles: np.ndarray) -> np.ndarray:
    """Compute 6 means from particles (N, 6)."""
    valid = ~np.isnan(particles).any(axis=1)
    particles = particles[valid]
    if len(particles) < 10:
        return np.full(6, np.nan)
    return particles.mean(axis=0)


def plot_projection_overlays(true_particles_list, sampled_particles_list,
                             output_path, n_examples=8):
    """Plot 2D projection overlays: true (blue) vs flow-sampled (red)."""
    n_examples = min(n_examples, len(true_particles_list))

    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_examples):
        true_p = true_particles_list[row]
        sampled_p = sampled_particles_list[row]

        # Filter NaN
        true_valid = ~np.isnan(true_p).any(axis=1)
        true_p = true_p[true_valid]

        for col, (var_a, var_b) in enumerate(PROJECTION_PAIRS):
            ax = axes[row, col]
            ia = PHASE_SPACE_VARS.index(var_a)
            ib = PHASE_SPACE_VARS.index(var_b)

            # Subsample for plotting speed
            n_plot = min(2000, len(true_p), len(sampled_p))
            rng = np.random.default_rng(42)

            if len(true_p) > n_plot:
                idx_t = rng.choice(len(true_p), n_plot, replace=False)
            else:
                idx_t = np.arange(len(true_p))
            if len(sampled_p) > n_plot:
                idx_s = rng.choice(len(sampled_p), n_plot, replace=False)
            else:
                idx_s = np.arange(len(sampled_p))

            ax.scatter(true_p[idx_t, ia], true_p[idx_t, ib],
                       alpha=0.3, s=2, c="blue", label="True")
            ax.scatter(sampled_p[idx_s, ia], sampled_p[idx_s, ib],
                       alpha=0.3, s=2, c="red", label="Flow")

            if row == 0:
                ax.set_title(f"{var_a}-{var_b}")
                ax.legend(markerscale=5, fontsize=8)
            ax.set_xlabel(var_a)
            ax.set_ylabel(var_b)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Projection overlays saved to {output_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-h5", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", default="analysis-flow")
    ap.add_argument("--n-samples", type=int, default=10000,
                    help="Particles to sample per shot (default: 10000)")
    ap.add_argument("--n-examples", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Shots to process at once (memory-limited by n_samples)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    # Load model
    flow, config, input_tf, particle_tf = load_flow(model_dir, device)
    x_mean = input_tf["x_mean"].numpy()
    x_std = input_tf["x_std"].numpy()
    p_mean = particle_tf["p_mean"].numpy()
    p_std = particle_tf["p_std"].numpy()

    # Load test data
    with h5py.File(args.test_h5, "r") as f:
        test_inputs = f["inputs"][:].astype(np.float32)
        test_particles = f["particles"][:].astype(np.float32)
        n_test = test_inputs.shape[0]

    print(f"[run] Test set: {n_test} shots", flush=True)
    print(f"[run] Sampling {args.n_samples} particles per shot ...", flush=True)

    # Process in batches
    all_true_cov = []
    all_pred_cov = []
    all_true_means = []
    all_pred_means = []
    all_true_particles = []
    all_sampled_particles = []

    for start in range(0, n_test, args.batch_size):
        end = min(start + args.batch_size, n_test)
        batch_inputs = test_inputs[start:end]
        batch_particles = test_particles[start:end]

        # Normalize inputs
        inputs_norm = torch.from_numpy((batch_inputs - x_mean) / x_std)

        # Sample from flow
        sampled = sample_from_flow(flow, inputs_norm, args.n_samples,
                                   device, particle_tf)

        for i in range(end - start):
            true_p = batch_particles[i]  # (N_sub, 6)
            samp_p = sampled[i]          # (n_samples, 6)

            true_cov = compute_covariance_6x6(true_p)
            pred_cov = compute_covariance_6x6(samp_p)
            true_mean = compute_means(true_p)
            pred_mean = compute_means(samp_p)

            all_true_cov.append(true_cov)
            all_pred_cov.append(pred_cov)
            all_true_means.append(true_mean)
            all_pred_means.append(pred_mean)

            # Store for overlay plots
            if len(all_true_particles) < args.n_examples:
                all_true_particles.append(true_p)
                all_sampled_particles.append(samp_p)

        if (start // args.batch_size) % 10 == 0:
            print(f"  processed {end}/{n_test} shots ...", flush=True)

    # ── Covariance metrics ────────────────────────────────────────────────────
    true_cov_arr = np.array(all_true_cov)   # (N, 6, 6)
    pred_cov_arr = np.array(all_pred_cov)   # (N, 6, 6)

    # Filter out any NaN entries
    valid = ~(np.isnan(true_cov_arr).any(axis=(1, 2)) |
              np.isnan(pred_cov_arr).any(axis=(1, 2)))
    true_cov_arr = true_cov_arr[valid]
    pred_cov_arr = pred_cov_arr[valid]

    mae_per_element = np.abs(true_cov_arr - pred_cov_arr).mean(axis=0)
    # MAPE (avoid division by zero)
    denom = np.abs(true_cov_arr)
    denom[denom < 1e-30] = 1e-30
    mape_per_element = (np.abs(true_cov_arr - pred_cov_arr) / denom).mean(axis=0) * 100

    print("\n[results] Covariance MAE (6x6 matrix):")
    for i in range(6):
        row_str = "  ".join(f"{mae_per_element[i,j]:.4e}" for j in range(6))
        print(f"  [{PHASE_SPACE_VARS[i]:2s}] {row_str}")

    print("\n[results] Covariance MAPE % (6x6 matrix):")
    for i in range(6):
        row_str = "  ".join(f"{mape_per_element[i,j]:7.2f}" for j in range(6))
        print(f"  [{PHASE_SPACE_VARS[i]:2s}] {row_str}")

    # ── Mean beam metrics ─────────────────────────────────────────────────────
    true_means_arr = np.array(all_true_means)
    pred_means_arr = np.array(all_pred_means)
    valid_means = ~(np.isnan(true_means_arr).any(axis=1) |
                    np.isnan(pred_means_arr).any(axis=1))
    true_means_arr = true_means_arr[valid_means]
    pred_means_arr = pred_means_arr[valid_means]

    mean_mae = np.abs(true_means_arr - pred_means_arr).mean(axis=0)
    print("\n[results] Mean beam MAE:")
    for i, var in enumerate(PHASE_SPACE_VARS):
        print(f"  mean_{var}: {mean_mae[i]:.6e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[run] Generating plots ...", flush=True)

    # Projection overlays
    plot_projection_overlays(
        all_true_particles, all_sampled_particles,
        output_dir / "projection_overlay.png",
        n_examples=min(args.n_examples, len(all_true_particles)))

    # Covariance diagonal scatter
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, var in enumerate(PHASE_SPACE_VARS):
        ax = axes[i // 3, i % 3]
        true_diag = true_cov_arr[:, i, i]
        pred_diag = pred_cov_arr[:, i, i]
        ax.scatter(true_diag, pred_diag, alpha=0.3, s=5)
        lims = [min(true_diag.min(), pred_diag.min()),
                max(true_diag.max(), pred_diag.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel(f"True var({var})")
        ax.set_ylabel(f"Flow var({var})")
        ax.set_title(f"var({var})  MAE={mae_per_element[i,i]:.3e}")
    plt.tight_layout()
    plt.savefig(output_dir / "covariance_diagonal_scatter.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # Training curve (if available)
    history_path = Path(args.model_dir) / "training_history.csv"
    if history_path.is_file():
        import pandas as pd
        hist = pd.read_csv(history_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist["train_nll"], label="Train NLL")
        ax.plot(hist["val_nll"], label="Val NLL")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL (per particle)")
        ax.legend()
        ax.set_title("Flow Training Curve")
        plt.tight_layout()
        plt.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Save metrics
    import pandas as pd
    metrics_rows = []
    for i in range(6):
        for j in range(6):
            metrics_rows.append({
                "element": f"cov_{PHASE_SPACE_VARS[i]}_{PHASE_SPACE_VARS[j]}",
                "mae": mae_per_element[i, j],
                "mape_pct": mape_per_element[i, j],
            })
    for i, var in enumerate(PHASE_SPACE_VARS):
        metrics_rows.append({
            "element": f"mean_{var}",
            "mae": mean_mae[i],
            "mape_pct": np.nan,
        })
    pd.DataFrame(metrics_rows).to_csv(output_dir / "test_metrics.csv", index=False)

    print(f"\n[done] Metrics saved to {output_dir}/test_metrics.csv", flush=True)
    print(f"[done] All outputs in {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
