"""Plot overlapping beam distributions: actual particles (OpenPMD) vs sampled from predicted covariance.

For each sample, loads the true particle distribution from the .h5 file and
samples particles from the predicted 6x6 covariance matrix (via the trained
model). Plots 2D phase-space projections (x-px, y-py) side by side.

Usage:
    python plot_beam_overlap.py
    python plot_beam_overlap.py --particles-csv particles-571.csv --num-samples 5
    python plot_beam_overlap.py --sample-indices 0 10 42
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from beamphysics import ParticleGroup

from facet2_inj_ml_model_571 import load_model as load_lume_model

from pv_mapping import (
    sim_to_machine_array,
    PV_MAPPING_BY_SIM_PARAM,
)

# Phase-space ordering: x, px, y, py, t, pz, z
# First 6 match the model's covariance output; z is appended as index 6
PHASE_SPACE_LABELS = ["x", "px", "y", "py", "t", "pz", "z"]
PHASE_SPACE_UNITS = ["m", "eV/c", "m", "eV/c", "s", "eV/c", "m"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot overlapping beam distributions: true particles vs predicted covariance."
    )
    parser.add_argument(
        "--particles-csv",
        default="particles-571.csv",
        help="CSV with bmad_final_particles column containing .h5 paths (default: particles-571.csv)",
    )
    parser.add_argument(
        "--particles-column",
        default="bmad_final_particles",
        help="Column containing OpenPMD .h5 file paths (default: bmad_final_particles)",
    )
    parser.add_argument(
        "--input-space",
        default="sim",
        choices=["sim", "machine"],
        help="Model input space: 'sim' or 'machine' (default: sim)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of random samples to plot (default: 5)",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific row indices to plot (overrides --num-samples)",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=10000,
        help="Number of particles to sample from predicted covariance (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        default="overlap-plots",
        help="Directory for output plots (default: overlap-plots)",
    )
    parser.add_argument(
        "--projections",
        nargs="+",
        default=["x-px", "y-py"],
        help="Phase-space projections to plot, e.g. x-px y-py (default: x-px y-py)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full model (covariance + all 6 phase-space means) for sampling with correct beam center",
    )
    return parser


def parse_projection(proj_str):
    """Parse 'x-px' into index pair (0, 1)."""
    parts = proj_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid projection format: {proj_str}. Use e.g. 'x-px'")
    idx_a = PHASE_SPACE_LABELS.index(parts[0])
    idx_b = PHASE_SPACE_LABELS.index(parts[1])
    return idx_a, idx_b


def load_true_particles(h5_path: str):
    """Load particle beam from h5 and return 7D phase-space array (N, 7): x, px, y, py, t, pz, z."""
    beam = ParticleGroup(h5=h5_path)
    # Extract: x, px, y, py, t, pz, z
    particles = np.column_stack([
        beam.x, beam.px, beam.y, beam.py, beam.t, beam.pz, beam.z
    ])
    return particles


def sample_from_covariance(cov_matrix: np.ndarray, n_particles: int, rng: np.random.Generator,
                           mean_vec: np.ndarray = None):
    """Sample particles from a multivariate normal with the given 6x6 covariance.

    Returns (N, 7) array: the 6 covariance dimensions plus z=0 (z is not modeled).
    """
    mean = mean_vec if mean_vec is not None else np.zeros(6)
    # Ensure symmetry and positive semi-definiteness
    cov_sym = (cov_matrix + cov_matrix.T) / 2
    try:
        particles_6d = rng.multivariate_normal(mean, cov_sym, size=n_particles)
    except np.linalg.LinAlgError:
        # Fall back: add small regularization
        cov_reg = cov_sym + np.eye(6) * 1e-30
        particles_6d = rng.multivariate_normal(mean, cov_reg, size=n_particles)
    # Append z=0 column (z is not part of the covariance model)
    z_col = np.zeros((n_particles, 1))
    return np.hstack([particles_6d, z_col])

def plot_overlap(
    true_particles: np.ndarray,
    pred_particles: np.ndarray,
    projections: list,
    sample_label: str,
    output_path: Path,
):
    """Plot overlapping 2D density contours of true vs predicted particles for given projections."""
    from scipy.stats import gaussian_kde

    n_proj = len(projections)
    fig, axes = plt.subplots(1, n_proj, figsize=(7 * n_proj, 5.5))
    if n_proj == 1:
        axes = [axes]

    for ax, (idx_a, idx_b) in zip(axes, projections):
        # Compute shared axis limits from both distributions
        all_a = np.concatenate([true_particles[:, idx_a], pred_particles[:, idx_a]])
        all_b = np.concatenate([true_particles[:, idx_b], pred_particles[:, idx_b]])
        pad_a = (np.percentile(all_a, 99) - np.percentile(all_a, 1)) * 0.15
        pad_b = (np.percentile(all_b, 99) - np.percentile(all_b, 1)) * 0.15
        a_min, a_max = np.percentile(all_a, 1) - pad_a, np.percentile(all_a, 99) + pad_a
        b_min, b_max = np.percentile(all_b, 1) - pad_b, np.percentile(all_b, 99) + pad_b

        # Grid for KDE evaluation
        grid_a, grid_b = np.mgrid[a_min:a_max:200j, b_min:b_max:200j]
        positions = np.vstack([grid_a.ravel(), grid_b.ravel()])

        # Contour levels at ~1σ, 2σ, 3σ equivalent density fractions
        levels_frac = [0.11, 0.39, 0.86]  # approximate enclosed fractions for 3σ, 2σ, 1σ

        def density_levels(data_a, data_b):
            """Compute KDE and return grid values and contour levels."""
            kde = gaussian_kde(np.vstack([data_a, data_b]))
            z = kde(positions).reshape(grid_a.shape)
            # Sort densities to find contour thresholds
            z_sorted = np.sort(z.ravel())[::-1]
            cumsum = np.cumsum(z_sorted) / z_sorted.sum()
            thresholds = [z_sorted[np.searchsorted(cumsum, f)] for f in levels_frac]
            return z, sorted(thresholds)

        # True beam KDE contours (blue)
        z_true, lvl_true = density_levels(true_particles[:, idx_a], true_particles[:, idx_b])
        ax.contour(grid_a, grid_b, z_true, levels=lvl_true,
                   colors="tab:blue", linewidths=[0.8, 1.2, 1.6])
        ax.contourf(grid_a, grid_b, z_true, levels=[lvl_true[0], z_true.max()],
                    colors=["tab:blue"], alpha=0.15)

        # Predicted beam KDE contours (orange)
        z_pred, lvl_pred = density_levels(pred_particles[:, idx_a], pred_particles[:, idx_b])
        ax.contour(grid_a, grid_b, z_pred, levels=lvl_pred,
                   colors="tab:orange", linewidths=[0.8, 1.2, 1.6], linestyles="dashed")
        ax.contourf(grid_a, grid_b, z_pred, levels=[lvl_pred[0], z_pred.max()],
                    colors=["tab:orange"], alpha=0.10)

        ax.set_xlabel(f"{PHASE_SPACE_LABELS[idx_a]} [{PHASE_SPACE_UNITS[idx_a]}]")
        ax.set_ylabel(f"{PHASE_SPACE_LABELS[idx_b]} [{PHASE_SPACE_UNITS[idx_b]}]")
        ax.set_xlim(a_min, a_max)
        ax.set_ylim(b_min, b_max)

        # Legend with proxy artists
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="tab:blue", linewidth=1.5, label="True (OpenPMD)"),
            Line2D([0], [0], color="tab:orange", linewidth=1.5, linestyle="dashed", label="Predicted (Covariance)"),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    fig.suptitle(sample_label, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()

    import pandas as pd

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    particles_col = args.particles_column

    # Parse projections
    projections = [parse_projection(p) for p in args.projections]

    # Load CSV with particle paths and input parameters
    print(f"[run] Loading particles CSV: {args.particles_csv}", flush=True)
    df = pd.read_csv(args.particles_csv, low_memory=False)

    if particles_col not in df.columns:
        raise SystemExit(f"CSV must contain '{particles_col}' column with .h5 file paths")

    # Load the lume-torch model
    print(f"[run] Loading lume-torch model (input_space={args.input_space!r})", flush=True)
    model = load_lume_model(args.input_space, full=args.full)

    # Get input feature names from the model
    pv_cols = model.input_names
    print(f"[run] Model input PVs ({len(pv_cols)}): {pv_cols}", flush=True)

    # Map PV names back to sim feature columns via pv_mapping
    pv_to_sim = {}
    for sim_param, spec in PV_MAPPING_BY_SIM_PARAM.items():
        pv_name = spec["experimental_pv"] or sim_param
        pv_to_sim[pv_name] = sim_param
    feature_cols = [pv_to_sim.get(pv, pv) for pv in pv_cols]
    print(f"[run] Sim feature columns: {feature_cols}", flush=True)

    # Check that input columns exist in CSV
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing model input columns: {missing}")

    # Select samples
    valid_mask = df[particles_col].notna() & (df[particles_col].str.strip() != "")
    valid_indices = df.index[valid_mask].tolist()

    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        n = min(args.num_samples, len(valid_indices))
        sample_indices = sorted(rng.choice(valid_indices, size=n, replace=False))

    print(f"[run] Plotting {len(sample_indices)} samples", flush=True)

    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        h5_path = str(row[particles_col]).strip()

        if not h5_path or h5_path == "nan":
            print(f"[skip] Row {idx}: no particle file", flush=True)
            continue

        print(f"[run] Sample {i+1}/{len(sample_indices)}: row {idx}, file={Path(h5_path).name}", flush=True)

        # Load true particles (no drift for screen 571)
        try:
            true_particles = load_true_particles(h5_path)
        except Exception as e:
            print(f"[skip] Row {idx}: failed to load particles: {e}", flush=True)
            continue

        # Build model input from CSV row
        sim_values = np.array([[float(row[col]) for col in feature_cols]], dtype=np.float32)
        if args.input_space == "machine":
            # Convert sim values to machine PV units for the machine model
            machine_values = sim_to_machine_array(sim_values, feature_cols)[0]
            input_dict = {pv: float(machine_values[k]) for k, pv in enumerate(pv_cols)}
        else:
            # Sim model expects simulator parameter values directly
            input_dict = {pv: float(sim_values[0, k]) for k, pv in enumerate(pv_cols)}
        result = model.evaluate(input_dict)

        # The model returns a dict with 'covariance_matrix' as a tensor
        # Output is already in physical units (M-denormalization is in the lume-torch output transformer)
        pred_cov_phys = result["covariance_matrix"].detach().cpu().numpy().squeeze()

        # If the model predicts phase-space means, use them as the sampling center
        mean_vec = None
        mean_keys = ["mean_x", "mean_px", "mean_y", "mean_py", "mean_t", "mean_pz"]
        if all(k in result for k in mean_keys):
            mean_vec = np.array([float(result[k].squeeze()) for k in mean_keys])

        # Sample particles from predicted covariance
        pred_particles = sample_from_covariance(pred_cov_phys, args.n_particles, rng, mean_vec=mean_vec)

        # Plot
        sample_label = f"Sample {idx} — True particles vs Predicted covariance"
        output_path = output_dir / f"overlap_sample_{idx:05d}.png"
        plot_overlap(true_particles, pred_particles, projections, sample_label, output_path)
        print(f"[run] Saved: {output_path}", flush=True)

    print(f"[run] Done. Plots saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
