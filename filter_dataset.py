"""Filter dataset.csv by beam-quality thresholds before train/val/test splitting.

Reconstructs 6x6 covariance matrices from Cholesky columns, M-denormalizes to
physical units, computes beam properties, and removes rows that exceed any
threshold.

Usage:
    python filter_dataset.py --input dataset.csv --output dataset-filtered.csv
    python filter_dataset.py --input dataset.csv --sigma-x-mm 2.0 --sigma-y-mm 2.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

M_DIAG = np.array([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6], dtype=np.float64)
M_INV = np.diag(1.0 / M_DIAG)
M_E_C = 511e3  # electron rest mass × c in eV

DEFAULT_THRESHOLDS = {
    "sigma_x_mm": 5.0,
    "sigma_y_mm": 5.0,
    "rel_energy_spread": 5e-3,
    "emit_x_norm_um": 20.0,
    "emit_y_norm_um": 20.0,
}


def chol_vectors_to_covariance(chol_vectors: np.ndarray) -> np.ndarray:
    """Convert (N, 21) Cholesky vectors to (N, 6, 6) covariance matrices."""
    n = chol_vectors.shape[0]
    tril_idx = np.tril_indices(6)
    L = np.zeros((n, 6, 6), dtype=chol_vectors.dtype)
    L[:, tril_idx[0], tril_idx[1]] = chol_vectors
    return L @ np.swapaxes(L, -1, -2)


def compute_beam_properties(cov_mnorm: np.ndarray, mean_pz: np.ndarray) -> dict:
    """Compute beam properties from M-normalized covariance matrices."""
    cov_phys = np.einsum("ij,njk,lk->nil", M_INV, cov_mnorm, M_INV)

    sigma_x_mm = np.sqrt(np.abs(cov_phys[:, 0, 0])) * 1e3
    sigma_y_mm = np.sqrt(np.abs(cov_phys[:, 2, 2])) * 1e3
    std_pz = np.sqrt(np.abs(cov_phys[:, 5, 5]))
    rel_espread = std_pz / np.abs(mean_pz)

    det_x = np.abs(cov_phys[:, 0, 0] * cov_phys[:, 1, 1] - cov_phys[:, 0, 1] ** 2)
    det_y = np.abs(cov_phys[:, 2, 2] * cov_phys[:, 3, 3] - cov_phys[:, 2, 3] ** 2)
    emit_x_norm_um = np.sqrt(det_x) / M_E_C * 1e6
    emit_y_norm_um = np.sqrt(det_y) / M_E_C * 1e6

    return {
        "sigma_x_mm": sigma_x_mm,
        "sigma_y_mm": sigma_y_mm,
        "rel_energy_spread": rel_espread,
        "emit_x_norm_um": emit_x_norm_um,
        "emit_y_norm_um": emit_y_norm_um,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Filter dataset by beam-quality thresholds."
    )
    parser.add_argument("--input", default="dataset.csv", help="Input dataset CSV")
    parser.add_argument("--output", default="dataset-filtered.csv", help="Output filtered CSV")
    parser.add_argument("--sigma-x-mm", type=float, default=DEFAULT_THRESHOLDS["sigma_x_mm"])
    parser.add_argument("--sigma-y-mm", type=float, default=DEFAULT_THRESHOLDS["sigma_y_mm"])
    parser.add_argument("--rel-energy-spread", type=float, default=DEFAULT_THRESHOLDS["rel_energy_spread"])
    parser.add_argument("--emit-x-um", type=float, default=DEFAULT_THRESHOLDS["emit_x_norm_um"])
    parser.add_argument("--emit-y-um", type=float, default=DEFAULT_THRESHOLDS["emit_y_norm_um"])
    return parser


def main():
    args = build_parser().parse_args()

    thresholds = {
        "sigma_x_mm": args.sigma_x_mm,
        "sigma_y_mm": args.sigma_y_mm,
        "rel_energy_spread": args.rel_energy_spread,
        "emit_x_norm_um": args.emit_x_um,
        "emit_y_norm_um": args.emit_y_um,
    }

    print(f"[run] Reading {args.input}", flush=True)
    df = pd.read_csv(args.input, low_memory=False)
    n_total = len(df)
    print(f"[run] Loaded {n_total} rows", flush=True)

    # Extract Cholesky columns and mean_pz
    chol_cols = sorted([c for c in df.columns if c.startswith("cov_chol_")],
                       key=lambda c: int(c.split("_")[-1]))
    if len(chol_cols) != 21:
        raise SystemExit(f"Expected 21 cov_chol_* columns, found {len(chol_cols)}")

    if "mean_pz" not in df.columns:
        raise SystemExit("Column 'mean_pz' required for relative energy spread calculation")

    chol_values = df[chol_cols].values.astype(np.float64)
    mean_pz = df["mean_pz"].values.astype(np.float64)

    # Reconstruct covariance matrices
    cov_mnorm = chol_vectors_to_covariance(chol_values)

    # Compute beam properties
    props = compute_beam_properties(cov_mnorm, mean_pz)

    # Apply thresholds
    mask = np.ones(n_total, dtype=bool)
    print(f"\n[filter] Applying beam-quality thresholds:", flush=True)
    for key, limit in thresholds.items():
        cut = props[key] < limit
        n_pass = cut.sum()
        mask &= cut
        print(f"  {key} < {limit}: {n_pass}/{n_total} ({n_pass/n_total*100:.1f}%) pass", flush=True)

    n_filtered = mask.sum()
    n_removed = n_total - n_filtered
    print(f"\n[filter] Combined: {n_filtered}/{n_total} ({n_filtered/n_total*100:.1f}%) pass", flush=True)
    print(f"[filter] Removed: {n_removed} ({n_removed/n_total*100:.1f}%)", flush=True)

    # Write filtered dataset
    df_filtered = df[mask].reset_index(drop=True)
    df_filtered.to_csv(args.output, index=False)
    print(f"\n[run] Saved {len(df_filtered)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
