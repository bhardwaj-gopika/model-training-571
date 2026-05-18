"""Extract mean beam energy and time from particle distributions.

Reads OpenPMD .h5 files from a CSV column, computes the mean energy (eV)
and mean time (s) for each particle distribution, and appends them as new
columns to the output CSV.
"""
# Example usage:
# python extract_beam_means.py particles-571.csv beam-means-571.csv --particles-column bmad_final_particles --progress-every 100 --drop-failed

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from beamphysics import ParticleGroup


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read particle .h5 paths from CSV, compute mean beam energy and "
            "mean time from each distribution, and save as new columns."
        )
    )
    parser.add_argument("input_csv", help="Input CSV containing particle file paths")
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="beam-means.csv",
        help="Output CSV path (default: beam-means.csv)",
    )
    parser.add_argument(
        "--particles-column",
        default="bmad_final_particles",
        help="Column containing OpenPMD .h5 file paths (default: bmad_final_particles)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N rows (default: 200)",
    )
    parser.add_argument(
        "--drop-failed",
        action="store_true",
        help="Drop rows where extraction failed",
    )
    parser.add_argument(
        "--errors-csv",
        default=None,
        help="Optional CSV path to write failed-row details",
    )
    return parser


def main():
    args = build_parser().parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    print(f"[run] Reading CSV: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)

    if args.particles_column not in df.columns:
        raise SystemExit(f"Column not found: {args.particles_column}")

    total_rows = len(df)
    mean_energies = []
    mean_times = []
    statuses = []
    error_types = []
    error_messages = []

    for i, file_path in enumerate(df[args.particles_column], start=1):
        status = "ok"
        mean_energy = np.nan
        mean_time = np.nan
        error_type = ""
        error_message = ""

        try:
            if pd.isna(file_path) or not str(file_path).strip():
                status = "missing_path"
            else:
                group = ParticleGroup(str(file_path))
                mean_energy = group.avg("energy")
                mean_time = group.avg("t")
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            error_type = type(exc).__name__
            error_message = str(exc)

        mean_energies.append(mean_energy)
        mean_times.append(mean_time)
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
    out_df["mean_energy"] = mean_energies
    out_df["mean_time"] = mean_times
    out_df["beam_means_status"] = statuses
    out_df["beam_means_error_type"] = error_types
    out_df["beam_means_error_message"] = error_messages

    status_counts = Counter(statuses)
    failure_counts = [(s, c) for s, c in status_counts.items() if s != "ok"]

    if failure_counts:
        print("[run] Failure summary:", flush=True)
        for status, count in sorted(failure_counts, key=lambda item: (-item[1], item[0]))[:10]:
            print(f"[run]   {status}: {count}", flush=True)

    if args.errors_csv:
        errors_df = out_df[out_df["beam_means_status"] != "ok"].copy()
        errors_df.insert(0, "source_row_index", errors_df.index)
        errors_path = Path(args.errors_csv)
        print(f"[run] Writing {errors_path} (failed_rows={len(errors_df)})", flush=True)
        errors_df.to_csv(errors_path, index=False)

    if args.drop_failed:
        before = len(out_df)
        out_df = out_df[out_df["beam_means_status"] == "ok"].copy()
        print(f"[run] Dropped {before - len(out_df)} failed rows", flush=True)

    ok_total = int((out_df["beam_means_status"] == "ok").sum())
    fail_total = len(out_df) - ok_total

    print(
        f"[run] Writing {output_csv} (rows={len(out_df)}, ok={ok_total}, failed={fail_total})",
        flush=True,
    )
    out_df.to_csv(output_csv, index=False)

    # Print summary stats for successful extractions
    valid = out_df[out_df["beam_means_status"] == "ok"]
    if len(valid) > 0:
        print(f"\n[summary] mean_energy: mean={valid['mean_energy'].mean():.6e}, "
              f"std={valid['mean_energy'].std():.6e}, "
              f"min={valid['mean_energy'].min():.6e}, "
              f"max={valid['mean_energy'].max():.6e}", flush=True)
        print(f"[summary] mean_time:   mean={valid['mean_time'].mean():.6e}, "
              f"std={valid['mean_time'].std():.6e}, "
              f"min={valid['mean_time'].min():.6e}, "
              f"max={valid['mean_time'].max():.6e}", flush=True)

    print("[run] Done", flush=True)


if __name__ == "__main__":
    main()
