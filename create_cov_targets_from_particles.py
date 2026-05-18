"""Create Cholesky-flattened covariance targets from particles OpenPMD files.

Optionally drifts particles to an exact z position and applies an M-normalization
(C_norm = M @ C @ M^T) before the Cholesky decomposition.
"""
#python create_cov_targets_from_particles.py dump-particles_241-not-null.csv cov-targets.csv --progress-every 100 --drop-failed
#python create_cov_targets_from_particles.py dump-particles_241-not-null.csv cov-targets.csv --drift-to-z 0.942084 --normalize --progress-every 100 --drop-failed
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from beamphysics import ParticleGroup


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read particles_241 .h5 paths from CSV, compute covariance via "
            "ParticleGroup, apply Cholesky, flatten lower-triangular non-zero "
            "entries, and save as target columns."
        )
    )
    parser.add_argument("input_csv", help="Input CSV containing particles_241 column")
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="dump-particles_241-cov-targets.csv",
        help="Output CSV path (default: dump-particles_241-cov-targets.csv)",
    )
    parser.add_argument(
        "--particles-column",
        default="particles_241",
        help="Column containing OpenPMD .h5 file paths (default: particles_241)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N rows (default: 200)",
    )
    parser.add_argument(
        "--nonzero-tol",
        type=float,
        default=0.0,
        help=(
            "Treat abs(value) <= tol as zero when flattening lower-triangular "
            "Cholesky entries (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--drop-failed",
        action="store_true",
        help="Drop rows where covariance/cholesky extraction failed",
    )
    parser.add_argument(
        "--errors-csv",
        default=None,
        help=(
            "Optional CSV path to write failed-row details, including "
            "exception type and message"
        ),
    )
    parser.add_argument(
        "--drift-to-z",
        type=float,
        default=None,
        help=(
            "Drift all particles to this z position (meters) before computing "
            "covariance. Required for element 241 (z=0.942084) to get meaningful "
            "t spread. (default: no drift)"
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Apply M-normalization C_norm = M @ C @ M^T with "
            "M = diag(1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6) before Cholesky. "
            "Brings all phase-space dimensions to comparable scales."
        ),
    )
    return parser


# M-normalization matrix: brings (x, px, y, py, t, pz) to comparable scales
M_DIAG = np.array([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6])
M = np.diag(M_DIAG)


def normalize_covariance(cov: np.ndarray) -> np.ndarray:
    """Apply M @ cov @ M^T normalization."""
    return M @ cov @ M.T


def cholesky_nonzero_vector(cov, tol=0.0):
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError(f"Expected square covariance matrix, got shape={cov_arr.shape}")

    chol = np.linalg.cholesky(cov_arr)
    lower_triangle = chol[np.tril_indices(chol.shape[0])]

    # Only filter if an explicit nonzero tolerance is set by the user.
    # By default (tol=0.0) keep all 21 elements for a 6x6 matrix so
    # every row has the same fixed-length output vector for ML training.
    if tol > 0:
        return lower_triangle[np.abs(lower_triangle) > tol]
    return lower_triangle


def main():
    args = build_parser().parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    print(f"[run] Reading CSV: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)

    if args.particles_column not in df.columns:
        raise SystemExit(f"Column not found: {args.particles_column}")

    total_rows = len(df)
    vectors = []
    statuses = []
    error_types = []
    error_messages = []
    expected_len = None

    for i, file_path in enumerate(df[args.particles_column], start=1):
        status = "ok"
        vec = None
        error_type = ""
        error_message = ""

        try:
            if pd.isna(file_path) or not str(file_path).strip():
                status = "missing_path"
            else:
                group = ParticleGroup(str(file_path))
                if args.drift_to_z is not None:
                    group.drift_to_z(z=args.drift_to_z)
                cov = np.asarray(group.cov('x', 'px', 'y', 'py', 't', 'pz'), dtype=float)
                if args.normalize:
                    cov = normalize_covariance(cov)
                vec = cholesky_nonzero_vector(cov, tol=args.nonzero_tol)

                if expected_len is None:
                    expected_len = len(vec)
                elif len(vec) != expected_len:
                    status = f"shape_mismatch:{len(vec)}"
                    error_type = "shape_mismatch"
                    error_message = f"Expected {expected_len}, got {len(vec)}"
                    vec = None

        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            error_type = type(exc).__name__
            error_message = str(exc)
            vec = None

        vectors.append(vec)
        statuses.append(status)
        error_types.append(error_type)
        error_messages.append(error_message)

        if args.progress_every and i % args.progress_every == 0:
            ok_count = sum(s == "ok" for s in statuses)
            fail_count = len(statuses) - ok_count
            print(
                f"[run] {i}/{total_rows} processed (ok={ok_count}, failed={fail_count})",
                flush=True,
            )

    out_df = df.copy()
    out_df["cov_target_status"] = statuses
    out_df["cov_target_error_type"] = error_types
    out_df["cov_target_error_message"] = error_messages

    status_counts = Counter(statuses)
    failure_counts = [(status, count) for status, count in status_counts.items() if status != "ok"]

    if failure_counts:
        print("[run] Failure summary:", flush=True)
        for status, count in sorted(failure_counts, key=lambda item: (-item[1], item[0]))[:10]:
            print(f"[run]   {status}: {count}", flush=True)

    if args.errors_csv:
        errors_df = out_df[out_df["cov_target_status"] != "ok"].copy()
        errors_df.insert(0, "source_row_index", errors_df.index)
        errors_path = Path(args.errors_csv)
        print(
            f"[run] Writing {errors_path} (failed_rows={len(errors_df)})",
            flush=True,
        )
        errors_df.to_csv(errors_path, index=False)

    if expected_len is None:
        raise SystemExit("No valid covariance vectors were generated. Check file access.")

    target_col_names = [f"cov_chol_{k}" for k in range(expected_len)]
    target_matrix = np.full((len(df), expected_len), np.nan, dtype=float)

    for row_idx, vec in enumerate(vectors):
        if vec is not None and len(vec) == expected_len:
            target_matrix[row_idx, :] = vec

    targets_df = pd.DataFrame(target_matrix, columns=target_col_names)
    out_df = pd.concat([out_df, targets_df], axis=1)

    if args.drop_failed:
        out_df = out_df[out_df["cov_target_status"] == "ok"].copy()

    ok_total = int((out_df["cov_target_status"] == "ok").sum())
    fail_total = len(out_df) - ok_total

    print(
        (
            f"[run] Writing {output_csv} "
            f"(rows={len(out_df)}, target_dim={expected_len}, ok={ok_total}, failed={fail_total})"
        ),
        flush=True,
    )
    out_df.to_csv(output_csv, index=False)
    print("[run] Done", flush=True)


if __name__ == "__main__":
    main()

