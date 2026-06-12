"""End-to-end demo of the FACET-II injector ML model (screen 571).

Demonstrates:
1. Loading the machine-PV model via ``load_model()``
2. Calling ``model.evaluate()`` with a PV-unit input dict
3. Wrapping the model in ``BeamOutputModel`` to produce a particle distribution
4. Overlaying the predicted distribution with the true OpenPMD particles

Usage:
    python end_to_end_demo.py
    python end_to_end_demo.py --particles-csv particles-571.csv --num-samples 3
    python end_to_end_demo.py --sample-indices 0 5 10
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beamphysics import ParticleGroup

from facet2_inj_ml_model_571 import load_model
from BeamOutputModel import BeamOutputModel
from pv_mapping import (
    PV_MAPPING_BY_SIM_PARAM,
    sim_to_machine_array,
    machine_input_names,
)

# Phase-space labels and units (x, px, y, py, t, pz, z)
PHASE_SPACE_LABELS = ["x", "px", "y", "py", "t", "pz", "z"]
PHASE_SPACE_UNITS = ["m", "eV/c", "m", "eV/c", "s", "eV/c", "m"]


def build_parser():
    parser = argparse.ArgumentParser(description="End-to-end FACET-II 571 injector model demo")
    parser.add_argument(
        "--particles-csv", default="particles-571.csv",
        help="CSV with bmad_final_particles column (default: particles-571.csv)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of random samples to demo (default: 3)",
    )
    parser.add_argument(
        "--sample-indices", type=int, nargs="+", default=None,
        help="Specific row indices (overrides --num-samples)",
    )
    parser.add_argument(
        "--n-particles", type=int, default=10000,
        help="Particles to sample from predicted covariance (default: 10000)",
    )
    parser.add_argument(
        "--output-dir", default="demo-output",
        help="Directory for output plots (default: demo-output)",
    )
    parser.add_argument(
        "--min-alive-particles", type=int, default=90000,
        help="Skip samples with fewer alive particles (default: 90000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser


def get_feature_cols_from_model(model):
    """Derive sim-parameter feature columns from the model's input PV names."""
    pv_to_sim = {}
    for sim_param, spec in PV_MAPPING_BY_SIM_PARAM.items():
        pv_name = spec["experimental_pv"] or sim_param
        pv_to_sim[pv_name] = sim_param
    pv_cols = model.input_names
    feature_cols = [pv_to_sim.get(pv, pv) for pv in pv_cols]
    return feature_cols, pv_cols


def sim_row_to_machine_dict(row: pd.Series, feature_cols: list, pv_names: list) -> dict:
    """Convert a CSV row (sim-parameter columns) to a machine-PV input dict."""
    sim_values = np.array([float(row[col]) for col in feature_cols])
    machine_values = sim_to_machine_array(sim_values, feature_cols)
    return {name: float(val) for name, val in zip(pv_names, machine_values)}


def load_true_particles(h5_path: str, min_alive: int = None) -> np.ndarray:
    """Load particle beam from .h5 and return (N, 7) array: x, px, y, py, t, pz, z.

    Filters to alive particles (status == 1) and optionally enforces a minimum count.
    """
    beam = ParticleGroup(h5=h5_path)
    alive_mask = beam["status"] == 1
    n_alive = int(np.count_nonzero(alive_mask))
    if n_alive == 0:
        raise ValueError(f"no alive particles in {h5_path}")
    if min_alive is not None and n_alive < min_alive:
        raise ValueError(f"{n_alive} alive < {min_alive} threshold")
    if n_alive < len(alive_mask):
        beam = beam[alive_mask]
    return np.column_stack([beam.x, beam.px, beam.y, beam.py, beam.t, beam.pz, beam.z])


def predicted_particles_from_beam(output_beam) -> np.ndarray:
    """Extract (N, 7) array from a ParticleGroup produced by BeamOutputModel."""
    return np.column_stack([
        output_beam.x, output_beam.px,
        output_beam.y, output_beam.py,
        output_beam.t, output_beam.pz,
        output_beam.z,
    ])


def plot_overlap(true_particles, pred_particles, sample_label, output_path):
    """Plot x-px and y-py phase-space overlays."""
    projections = [(0, 1), (2, 3)]  # x-px, y-py
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (ia, ib) in zip(axes, projections):
        ax.scatter(
            true_particles[:, ia], true_particles[:, ib],
            s=0.3, alpha=0.4, color="tab:blue", label="True (OpenPMD)", rasterized=True,
        )
        ax.scatter(
            pred_particles[:, ia], pred_particles[:, ib],
            s=0.5, alpha=0.6, color="tab:orange", label="Predicted (Model)", rasterized=True,
        )
        ax.set_xlabel(f"{PHASE_SPACE_LABELS[ia]} [{PHASE_SPACE_UNITS[ia]}]")
        ax.set_ylabel(f"{PHASE_SPACE_LABELS[ib]} [{PHASE_SPACE_UNITS[ib]}]")
        ax.legend(fontsize=8, markerscale=5)

    fig.suptitle(sample_label, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {output_path}")


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Step 1: Load the machine-PV model
    # ------------------------------------------------------------------
    print("STEP 1: Load machine-PV model")
    machine_model = load_model("machine")
    feature_cols, pv_cols = get_feature_cols_from_model(machine_model)
    print(f"  Model loaded (input_space='machine')")
    print(f"  {len(pv_cols)} input PVs, {len(feature_cols)} sim feature columns")

    # ------------------------------------------------------------------
    # Step 2: Demonstrate model.evaluate() with a PV-unit input dict
    # ------------------------------------------------------------------
    print()
    print("STEP 2: Call model.evaluate() with PV-unit inputs")

    # Build an example input from the first valid row of the particles CSV
    df = pd.read_csv(args.particles_csv, low_memory=False)
    particles_col = "bmad_final_particles"

    # Find first valid row for example
    valid_mask = df[particles_col].notna() & (df[particles_col].str.strip() != "")
    for col in feature_cols:
        if col in df.columns:
            valid_mask &= df[col].notna()
    first_valid = df.index[valid_mask][0]
    example_row = df.iloc[first_valid]
    example_input = sim_row_to_machine_dict(example_row, feature_cols, pv_cols)

    print(f"  Input dict (from row {first_valid}):")
    for k, v in example_input.items():
        print(f"    {k}: {v}")

    result = machine_model.evaluate(example_input)
    print(f"\n  evaluate() result keys: {list(result.keys())}")
    if "covariance_matrix" in result:
        cov = result["covariance_matrix"]
        print(f"  covariance_matrix shape: {cov.shape}")
        print(f"  covariance_matrix:\n{cov}")

    # ------------------------------------------------------------------
    # Step 3: Wrap in BeamOutputModel to get particle distribution
    # ------------------------------------------------------------------
    print()
    print("STEP 3: Wrap model in BeamOutputModel for particle generation")
    beam_model = BeamOutputModel(load_model("machine"), n_particles=args.n_particles)
    beam_model.set(example_input)
    output_beam = beam_model.final_particles
    print(f"  Generated {len(output_beam.x)} particles from predicted covariance")
    print(f"  Output beam mean x: {output_beam.x.mean():.6e} m")
    print(f"  Output beam std  x: {output_beam.x.std():.6e} m")

    # ------------------------------------------------------------------
    # Step 4: Compare with true OpenPMD particles (beam overlay plots)
    # ------------------------------------------------------------------
    print()
    print("STEP 4: Beam overlay — predicted vs true particles")

    if particles_col not in df.columns:
        print(f"  [skip] CSV does not contain '{particles_col}' column; skipping overlay plots.")
        return

    valid_indices = df.index[valid_mask].tolist()
    print(f"  Found {len(valid_indices)} rows with valid particles and inputs")

    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        n = min(args.num_samples, len(valid_indices))
        sample_indices = sorted(rng.choice(valid_indices, size=n, replace=False))

    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        h5_path = str(row[particles_col]).strip()
        if not h5_path or h5_path == "nan":
            print(f"  [skip] Row {idx}: no particle file")
            continue

        print(f"\n  Sample {i+1}/{len(sample_indices)}: row {idx}, file={Path(h5_path).name}")

        # Convert sim inputs -> machine PV dict
        machine_input = sim_row_to_machine_dict(row, feature_cols, pv_cols)
        print(f"  Machine-PV inputs:")
        for k, v in list(machine_input.items())[:5]:
            print(f"    {k}: {v}")
        if len(machine_input) > 5:
            print(f"    ... ({len(machine_input) - 5} more)")

        # Evaluate via BeamOutputModel (machine model)
        beam_model.set(machine_input)
        output_beam = beam_model.final_particles
        pred_particles = predicted_particles_from_beam(output_beam)

        # Load true particles
        try:
            true_particles = load_true_particles(h5_path, min_alive=args.min_alive_particles)
        except Exception as e:
            print(f"  [skip] Failed to load particles: {e}")
            continue

        # Plot overlay
        sample_label = f"Row {idx} — True particles vs Model prediction (screen 571)"
        output_path = output_dir / f"demo_overlap_{idx:05d}.png"
        plot_overlap(true_particles, pred_particles, sample_label, output_path)

    print(f"\nDemo complete. Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
