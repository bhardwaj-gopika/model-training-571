"""End-to-end demo of the FACET-II injector ML model (screen 571) — HAAI edition.

This is the HAAI-data-standard counterpart to ``end_to_end_demo.py``. The old demo
read true particles from a per-row ``bmad_final_particles`` .h5 path column. The HAAI
dataset has no such column: ``dataset.csv`` contains only model inputs + targets, and the
true PR10571 particles live *inside* the big ``*_571.h5`` files under
``<uuid>/observables/PR10571/571_particles_<shot>/electron/``.

The link between a ``dataset.csv`` row and its source particles is the provenance sidecar
(``dataset.provenance.csv``) written by ``build_dataset_from_standard.py``. It is row-aligned
1:1 with ``dataset.csv`` and stores ``source_file, uuid, shot`` per row.

Demonstrates:
1. Loading the machine-PV model via ``load_model()``
2. Calling ``model.evaluate()`` with a PV-unit input dict (inputs from ``dataset.csv``)
3. Wrapping the model in ``BeamOutputModel`` to produce a particle distribution
4. Overlaying the predicted distribution with the true HAAI particles

NOTE: Step 4 must run on S3DF (or anywhere the HAAI ``*_571.h5`` files are readable),
since those files are large (>100 GB each) and cannot be copied off-cluster. Steps 1-3
only need ``dataset.csv`` and run anywhere.

Usage:
    # full demo on S3DF (inputs from dataset.csv, true particles from HAAI files)
    python end_to_end_demo_haai.py \
        --dataset-csv dataset.csv \
        --provenance-csv dataset.provenance.csv \
        --haai-dir /sdf/data/ad/ard-online/FACET-II_Training_Data/scraped_data \
        --num-samples 3

    # steps 1-3 only (no true-particle overlay); works off-cluster
    python end_to_end_demo_haai.py --dataset-csv dataset.csv --skip-overlay

    # specific rows
    python end_to_end_demo_haai.py --sample-indices 0 5 10
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy import constants

try:  # HAAI / data_standard environments expose this name
    from pmd_beamphysics import ParticleGroup
except ImportError:  # old 571 pipeline used this name
    from beamphysics import ParticleGroup

from facet2_inj_ml_model_571 import load_model
from BeamOutputModel import BeamOutputModel
from indist_utils import knn_distance, indist_percentile
from pv_mapping import (
    PV_MAPPING_BY_SIM_PARAM,
    sim_to_machine_array,
)

# Phase-space labels and units (x, px, y, py, t, pz, z)
PHASE_SPACE_LABELS = ["x", "px", "y", "py", "t", "pz", "z"]
PHASE_SPACE_UNITS = ["m", "eV/c", "m", "eV/c", "s", "eV/c", "m"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end FACET-II 571 injector model demo (HAAI dataset)"
    )
    parser.add_argument(
        "--dataset-csv", default="dataset.csv",
        help="HAAI dataset CSV with model-input columns (default: dataset.csv)",
    )
    parser.add_argument(
        "--provenance-csv", default=None,
        help="Provenance sidecar with source_file,uuid,shot per row "
             "(default: <dataset-csv>.provenance.csv). Required for Step 4 overlay.",
    )
    parser.add_argument(
        "--haai-dir",
        default="/sdf/data/ad/ard-online/FACET-II_Training_Data/scraped_data",
        help="Directory holding the HAAI *_571.h5 files referenced by the provenance "
             "'source_file' column (default: the S3DF scraped_data path).",
    )
    parser.add_argument(
        "--particles-location", default="PR10571",
        help="Observables location group holding the target PGs (default: PR10571)",
    )
    parser.add_argument(
        "--particles-prefix", default="571_particles",
        help="Per-shot PG group name prefix (default: 571_particles)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
        help="Number of random samples to demo (default: 3)",
    )
    parser.add_argument(
        "--sample-indices", type=int, nargs="+", default=None,
        help="Specific dataset row indices (overrides --num-samples)",
    )
    parser.add_argument(
        "--n-particles", type=int, default=10000,
        help="Particles to sample from predicted covariance (default: 10000)",
    )
    parser.add_argument(
        "--output-dir", default="demo-output-haai",
        help="Directory for output plots (default: demo-output-haai)",
    )
    parser.add_argument(
        "--min-alive-frac", type=float, default=0.9,
        help="Skip samples whose alive fraction (status==1 / total) is below this "
             "(default: 0.9; matches the training filter). Set 0 to disable.",
    )
    parser.add_argument(
        "--skip-overlay", action="store_true",
        help="Only run Steps 1-3 (model evaluation + particle generation); skip the "
             "true-particle overlay. Use when the HAAI .h5 files are not available.",
    )
    parser.add_argument(
        "--train-csv", default=None,
        help="Training-split CSV used as the in-distribution reference for scoring how "
             "well-represented each sample is (default: the --dataset-csv itself).",
    )
    parser.add_argument(
        "--select", choices=("random", "high-density"), default="random",
        help="How to pick overlay samples: 'random' (default) or 'high-density' "
             "(most in-distribution / best-represented settings first).",
    )
    parser.add_argument(
        "--knn-k", type=int, default=10,
        help="Neighbours for the in-distribution kNN distance (default: 10).",
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


def load_particlegroup(parent_grp):
    """Read a ParticleGroup written by pmd_beamphysics `pg.write(group)`.

    On disk the shot lives at `<location>/<name>_<i>/` and contains an openPMD species
    subgroup `electron/`. pmd_beamphysics reads that species group directly, so prefer it;
    fall back to the parent group for other layouts. (Mirrors build_dataset_from_standard.py.)
    """
    for species in ("electron", "positron", "proton"):
        if species in parent_grp:
            return ParticleGroup(h5=parent_grp[species])
    return ParticleGroup(h5=parent_grp)


def load_true_particles_haai(h5_path, uuid, shot, location, prefix,
                             min_alive_frac=None) -> np.ndarray:
    """Load true PR10571 particles for one HAAI shot as an (N, 7) array.

    Navigates <uuid>/observables/<location>/<prefix>_<shot>/electron/ inside the HAAI
    *_571.h5 archive, filters to alive particles (status == 1), and optionally enforces
    a minimum alive fraction. Returns columns: x, px, y, py, t, pz, z.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        if uuid not in f:
            raise ValueError(f"uuid {uuid} not in {Path(h5_path).name}")
        obs = f[uuid].get("observables")
        if obs is None or location not in obs:
            raise ValueError(f"location {location} missing for uuid {uuid}")
        pg_name = f"{prefix}_{shot}"
        loc_grp = obs[location]
        if pg_name not in loc_grp:
            raise ValueError(f"{pg_name} not in {location} for uuid {uuid}")
        beam = load_particlegroup(loc_grp[pg_name])

    status = np.asarray(beam.status)
    n_total = int(status.size)
    alive_mask = status == 1
    n_alive = int(np.count_nonzero(alive_mask))
    if n_alive == 0:
        raise ValueError("no alive particles")
    frac = n_alive / n_total if n_total else 0.0
    if min_alive_frac is not None and frac < min_alive_frac:
        raise ValueError(f"alive frac {frac:.3f} < {min_alive_frac} threshold")
    if n_alive < n_total:
        beam = beam[alive_mask]
    return np.column_stack([beam.x, beam.px, beam.y, beam.py, beam.t, beam.pz, beam.z])


def predicted_particles_from_beam(output_beam) -> np.ndarray:
    """Extract (N, 7) array from a ParticleGroup produced by BeamOutputModel.

    BeamOutputModel converts the surrogate's t-axis to z (z = -c*t) for distgen,
    so the output ParticleGroup has z values but t is constant (tstart). We recover
    t from z via t = -z/c to enable t-pz comparisons with true HAAI particles.
    """
    t_from_z = -output_beam.z / constants.speed_of_light
    return np.column_stack([
        output_beam.x, output_beam.px,
        output_beam.y, output_beam.py,
        t_from_z, output_beam.pz,
        output_beam.z,
    ])


def _cov_2d(points_a, points_b):
    """Return (mean_a, mean_b, 2x2 covariance, correlation) for two 1D arrays."""
    data = np.vstack([points_a, points_b])
    cov = np.cov(data)
    denom = np.sqrt(cov[0, 0] * cov[1, 1])
    corr = cov[0, 1] / denom if denom > 0 else np.nan
    return float(data[0].mean()), float(data[1].mean()), cov, float(corr)


def _draw_cov_ellipse(ax, cx, cy, cov, color, nsig=2.0, label=None):
    """Draw an n-sigma covariance ellipse from a 2x2 covariance matrix."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nsig * np.sqrt(vals)
    ell = Ellipse((cx, cy), width, height, angle=angle, edgecolor=color,
                  facecolor="none", lw=2.0, ls="-", label=label, zorder=5)
    ax.add_patch(ell)


def plot_input_distributions(sample_values, feature_cols, train_df, sample_label, output_path):
    """Plot training-set histograms for each input feature with this sample's value marked."""
    n_features = len(feature_cols)
    ncols = 6
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        train_vals = train_df[col].dropna().values
        ax.hist(train_vals, bins=50, color="0.7", edgecolor="none", density=True)
        val = sample_values[col]
        ax.axvline(val, color="red", lw=2)
        pct = 100.0 * np.mean(train_vals <= val)
        short_name = col.split(":")[-1] if ":" in col else col
        ax.set_title(f"{short_name}\npct={pct:.0f}%", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{sample_label}\nRed line = this sample's value; pct = percentile in training set", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {output_path}")


def plot_overlap(true_particles, pred_particles, sample_label, output_path):
    """Plot x-px, y-py, x-y, and t-pz phase-space overlays with covariance ellipses.

    Adds a 2-sigma covariance ellipse for the true (blue) and predicted (orange)
    distributions on each panel, and annotates the predicted vs true correlation
    coefficient. This separates the covariance-only Gaussian assumption (ellipse
    agreement) from the non-Gaussian shape of the true beam (scatter vs ellipse).
    Returns a dict of per-projection correlations for logging.
    """
    projections = [(0, 1), (2, 3), (0, 2), (4, 5)]  # x-px, y-py, x-y, t-pz
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    corr_out = {}

    for ax, (ia, ib) in zip(axes, projections):
        ax.scatter(
            true_particles[:, ia], true_particles[:, ib],
            s=0.3, alpha=0.4, color="tab:blue", label="True (HAAI)", rasterized=True,
        )
        ax.scatter(
            pred_particles[:, ia], pred_particles[:, ib],
            s=0.5, alpha=0.6, color="tab:orange", label="Predicted (Model)", rasterized=True,
        )

        tcx, tcy, tcov, tcorr = _cov_2d(true_particles[:, ia], true_particles[:, ib])
        pcx, pcy, pcov, pcorr = _cov_2d(pred_particles[:, ia], pred_particles[:, ib])
        _draw_cov_ellipse(ax, tcx, tcy, tcov, "navy", label="True 2σ cov")
        _draw_cov_ellipse(ax, pcx, pcy, pcov, "red", label="Pred 2σ cov")

        proj = f"{PHASE_SPACE_LABELS[ia]}-{PHASE_SPACE_LABELS[ib]}"
        corr_out[proj] = (tcorr, pcorr)
        ax.set_title(f"{proj}:  corr true={tcorr:+.3f}  pred={pcorr:+.3f}", fontsize=10)
        ax.set_xlabel(f"{PHASE_SPACE_LABELS[ia]} [{PHASE_SPACE_UNITS[ia]}]")
        ax.set_ylabel(f"{PHASE_SPACE_LABELS[ib]} [{PHASE_SPACE_UNITS[ib]}]")
        ax.legend(fontsize=8, markerscale=5)

    fig.suptitle(sample_label, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved: {output_path}")
    return corr_out


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    min_alive_frac = args.min_alive_frac if args.min_alive_frac and args.min_alive_frac > 0 else None

    # ------------------------------------------------------------------
    # Step 1: Load the machine-PV model
    # ------------------------------------------------------------------
    print("STEP 1: Load machine-PV model (full: covariance + means)")
    machine_model = load_model("machine", full=True)
    feature_cols, pv_cols = get_feature_cols_from_model(machine_model)
    print(f"  Model loaded (input_space='machine', full=True)")
    print(f"  {len(pv_cols)} input PVs, {len(feature_cols)} sim feature columns")

    # ------------------------------------------------------------------
    # Load the HAAI dataset (model inputs) and verify feature columns
    # ------------------------------------------------------------------
    dataset_csv = Path(args.dataset_csv)
    df = pd.read_csv(dataset_csv, low_memory=False)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Dataset {dataset_csv} is missing model-input columns: {missing}. "
            "Is this the HAAI dataset.csv that matches the deployed model?"
        )

    # ------------------------------------------------------------------
    # Step 2: Demonstrate model.evaluate() with a PV-unit input dict
    # ------------------------------------------------------------------
    print()
    print("STEP 2: Call model.evaluate() with PV-unit inputs")

    first_valid = int(df.index[df[feature_cols].notna().all(axis=1)][0])
    example_row = df.iloc[first_valid]
    example_input = sim_row_to_machine_dict(example_row, feature_cols, pv_cols)

    print(f"  Input dict (from dataset row {first_valid}):")
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
    beam_model = BeamOutputModel(load_model("machine", full=True), n_particles=args.n_particles)
    beam_model.set(example_input)
    output_beam = beam_model.final_particles
    print(f"  Generated {len(output_beam.x)} particles from predicted covariance")
    print(f"  Output beam mean x: {output_beam.x.mean():.6e} m")
    print(f"  Output beam std  x: {output_beam.x.std():.6e} m")

    # ------------------------------------------------------------------
    # Step 4: Compare with true HAAI particles (beam overlay plots)
    # ------------------------------------------------------------------
    print()
    print("STEP 4: Beam overlay — predicted vs true particles")

    if args.skip_overlay:
        print("  [skip] --skip-overlay set; not loading HAAI particles.")
        print(f"\nDemo complete (Steps 1-3). Outputs in {output_dir}/")
        return

    prov_path = Path(args.provenance_csv) if args.provenance_csv \
        else dataset_csv.with_name(dataset_csv.name + ".provenance.csv")
    if not prov_path.is_file():
        print(f"  [skip] Provenance sidecar not found: {prov_path}")
        print("         Step 4 needs source_file/uuid/shot per row. Pass --provenance-csv "
              "or run on S3DF where the sidecar lives.")
        return

    prov = pd.read_csv(prov_path, low_memory=False)
    if len(prov) != len(df):
        print(f"  [warn] provenance rows ({len(prov)}) != dataset rows ({len(df)}); "
              "they may be misaligned. Ensure both were concatenated in the same order.")
    n_aligned = min(len(prov), len(df))

    haai_dir = Path(args.haai_dir)
    valid_indices = list(range(n_aligned))
    print(f"  {n_aligned} dataset rows aligned with provenance")
    print(f"  HAAI files expected under: {haai_dir}")

    # In-distribution scoring: how well-represented each setting is in the training
    # data (0 = denser than nearly all training points; 100 = OOD). Answers the
    # supervisor's "how in-distribution are these?" and enables high-density picking.
    feat_ok = df[feature_cols].notna().all(axis=1).values[:n_aligned]
    query_X = df[feature_cols].values[:n_aligned].astype(float)
    indist_pct = np.full(n_aligned, np.nan)
    train_ref = Path(args.train_csv) if args.train_csv else dataset_csv
    train_df = None
    try:
        train_df = pd.read_csv(train_ref, usecols=feature_cols, low_memory=False)
        train_X = train_df[feature_cols].dropna().values.astype(float)
        dq, dref = knn_distance(query_X, train_X, k=args.knn_k)
        indist_pct = indist_percentile(dq, dref)
        print(f"  In-distribution reference: {train_ref.name} "
              f"({len(train_X)} rows, k={args.knn_k})")
    except Exception as e:
        print(f"  [warn] in-distribution scoring unavailable: {e}")

    if "distgen:total_charge:value" in df.columns:
        charge_pc = df["distgen:total_charge:value"].values[:n_aligned].astype(float)
    else:
        charge_pc = np.full(n_aligned, np.nan)

    if args.sample_indices is not None:
        sample_indices = [i for i in args.sample_indices if i < n_aligned]
    elif args.select == "high-density":
        # smallest in-distribution percentile first = best-represented settings
        ranked = [int(i) for i in np.argsort(indist_pct) if feat_ok[i]]
        sample_indices = sorted(ranked[: args.num_samples])
        print(f"  Selecting {len(sample_indices)} highest-density (most in-distribution) samples")
    else:
        cand = [i for i in valid_indices if feat_ok[i]]
        n = min(args.num_samples, len(cand))
        sample_indices = sorted(rng.choice(cand, size=n, replace=False).tolist())

    for i, idx in enumerate(sample_indices):
        prov_row = prov.iloc[idx]
        source_file = str(prov_row["source_file"]).strip()
        uuid = str(prov_row["uuid"]).strip()
        shot = int(prov_row["shot"])
        h5_path = haai_dir / source_file
        if not h5_path.is_file():
            matches = list(haai_dir.glob(f"*/{source_file}"))
            if matches:
                h5_path = matches[0]

        print(f"\n  Sample {i+1}/{len(sample_indices)}: dataset row {idx}")
        print(f"    source_file={source_file}  uuid={uuid}  shot={shot}")
        c, p = charge_pc[idx], indist_pct[idx]
        cstr = f"{c:.0f} pC ({c/1000:.2f} nC)" if np.isfinite(c) else "charge n/a"
        pstr = f"in-dist pct={p:.0f}" if np.isfinite(p) else "in-dist n/a"
        print(f"    charge={cstr}  {pstr} (0=well-represented, 100=OOD)")

        if not h5_path.is_file():
            print(f"  [skip] HAAI file not found: {h5_path}")
            continue

        # Convert dataset-row sim inputs -> machine PV dict, evaluate the model
        machine_input = sim_row_to_machine_dict(df.iloc[idx], feature_cols, pv_cols)
        beam_model.set(machine_input)
        output_beam = beam_model.final_particles
        pred_particles = predicted_particles_from_beam(output_beam)

        # Load true particles from the HAAI archive
        try:
            true_particles = load_true_particles_haai(
                str(h5_path), uuid, shot,
                args.particles_location, args.particles_prefix,
                min_alive_frac=min_alive_frac,
            )
        except Exception as e:
            print(f"  [skip] Failed to load HAAI particles: {e}")
            continue

        sample_label = (
            f"Row {idx} — {cstr}, {pstr} (0=well-represented, 100=OOD)\n"
            f"True vs Model prediction (screen 571)"
        )
        output_path = output_dir / f"demo_overlap_{idx:05d}.png"
        corrs = plot_overlap(true_particles, pred_particles, sample_label, output_path)
        for proj, (tcorr, pcorr) in corrs.items():
            print(f"    {proj} correlation: true={tcorr:+.3f}  pred={pcorr:+.3f}  "
                  f"(Δ={pcorr - tcorr:+.3f})")

        # Input distribution context: where does this sample sit in the training data?
        if train_df is not None:
            inputs_path = output_dir / f"demo_inputs_{idx:05d}.png"
            plot_input_distributions(
                df.iloc[idx], feature_cols, train_df, sample_label, inputs_path
            )

    print(f"\nDemo complete. Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
