"""Build the 2D projection dataset for the CNN sanity check.

Reads *_571.h5 files (HAAI Data Standard) and produces an HDF5 dataset where each
sample contains the 18 scalar inputs and three 64x64 normalized 2D histograms of the
beam distribution (x-px, y-py, t-pz projections).

Two modes:
  --compute-edges   Scan shots to determine global bin edges (1st-99th percentile
                    + 10% padding). Writes edges to an .npz file.
  (default)         Bin each shot using pre-computed edges from --edges-file.

Example workflow:
    # Pass 1: compute bin edges from a subset of files
    python build_projection_dataset.py \\
        /sdf/.../scraped_data/*_571.h5 \\
        --compute-edges --edges-output bin_edges.npz \\
        --min-alive-frac 0.9 --wrap-phase --input-filter \\
        --max-shots-per-batch 5

    # Pass 2: build the full dataset
    python build_projection_dataset.py \\
        /sdf/.../scraped_data/*_571.h5 \\
        --output projections_dataset.h5 --edges-file bin_edges.npz \\
        --min-alive-frac 0.9 --wrap-phase --input-filter \\
        --progress-every 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import h5py

from build_dataset_from_standard import (
    batch_ids,
    n_shots,
    read_feature_matrix,
    load_particlegroup,
    build_filter_specs,
    wrap_deg,
    file_campaign,
    FEATURE_COLS,
    FEATURE_MAP,
    INPUT_RANGES,
    PHASE_WRAP_COLS,
    CHARGE_COL,
    CAMPAIGNS,
)

PHASE_SPACE_VARS = ["x", "px", "y", "py", "t", "pz"]
PROJECTION_PAIRS = [("x", "px"), ("y", "py"), ("t", "pz")]
N_BINS = 64


def compute_bin_edges(all_values: dict[str, list[np.ndarray]], percentile_lo=1.0,
                      percentile_hi=99.0, padding_frac=0.10) -> dict[str, np.ndarray]:
    """Compute fixed bin edges from collected particle values.

    Uses the percentile range + padding to define edges that cover most particles
    without being dominated by outliers.
    """
    edges = {}
    for var in PHASE_SPACE_VARS:
        combined = np.concatenate(all_values[var])
        lo = np.percentile(combined, percentile_lo)
        hi = np.percentile(combined, percentile_hi)
        span = hi - lo
        lo -= padding_frac * span
        hi += padding_frac * span
        edges[var] = np.linspace(lo, hi, N_BINS + 1)
    return edges


def bin_particles(pg_alive, edges: dict[str, np.ndarray]) -> np.ndarray:
    """Bin alive particles into three 64x64 normalized 2D histograms.

    Returns (3, 64, 64) float32 array. Each histogram sums to 1.
    """
    projections = np.zeros((3, N_BINS, N_BINS), dtype=np.float32)

    for i, (var_a, var_b) in enumerate(PROJECTION_PAIRS):
        a_vals = getattr(pg_alive, var_a)
        b_vals = getattr(pg_alive, var_b)
        hist, _, _ = np.histogram2d(
            a_vals, b_vals,
            bins=[edges[var_a], edges[var_b]],
        )
        total = hist.sum()
        if total > 0:
            hist /= total
        projections[i] = hist.astype(np.float32)

    return projections


def process_file_edges(path, args):
    """Scan a file and collect particle values for edge computation."""
    pg_location = args.particles_location
    pg_prefix = args.particles_prefix

    seed_key = f"{Path(path).name}:{args.sample_seed}".encode()
    rng = np.random.default_rng(zlib.crc32(seed_key))

    all_values = {var: [] for var in PHASE_SPACE_VARS}
    n_collected = 0

    filter_specs = build_filter_specs(args)

    with h5py.File(path, "r") as f:
        ids = batch_ids(f)
        if args.limit_batches:
            ids = ids[: args.limit_batches]

        for bi, uuid in enumerate(ids, start=1):
            g = f[uuid]
            if "observables" not in g:
                continue
            obs = g["observables"]
            n = n_shots(g)
            feats = read_feature_matrix(obs, n)

            if args.wrap_phase:
                for _c in PHASE_WRAP_COLS:
                    _j = FEATURE_COLS.index(_c)
                    feats[:, _j] = wrap_deg(feats[:, _j])

            if pg_location not in obs:
                continue
            loc_grp = obs[pg_location]

            shot_indices = range(n)
            if args.max_shots_per_batch is not None and n > args.max_shots_per_batch:
                shot_indices = sorted(
                    rng.choice(n, size=args.max_shots_per_batch, replace=False))
            elif args.sample_frac is not None and args.sample_frac < 1.0:
                mask = rng.random(n) < args.sample_frac
                shot_indices = [i for i in range(n) if mask[i]]

            for i in shot_indices:
                pg_name = f"{pg_prefix}_{i}"
                if pg_name not in loc_grp:
                    continue

                try:
                    pg = load_particlegroup(loc_grp[pg_name])
                except Exception:
                    continue

                status = np.asarray(pg.status)
                n_total = int(status.size)
                alive = status == 1
                n_alive = int(np.count_nonzero(alive))
                frac = (n_alive / n_total) if n_total else 0.0

                if args.min_alive_frac is not None and frac < args.min_alive_frac:
                    continue
                if n_alive == 0:
                    continue

                feat_row = feats[i]
                if not np.all(np.isfinite(feat_row)):
                    continue

                if filter_specs:
                    if any(feat_row[j] < lo or feat_row[j] > hi
                           for j, lo, hi in filter_specs):
                        continue

                pg_alive = pg[alive] if n_alive < n_total else pg

                for var in PHASE_SPACE_VARS:
                    vals = getattr(pg_alive, var)
                    # Subsample particles for edge computation (memory efficiency)
                    if len(vals) > 5000:
                        idx = rng.choice(len(vals), size=5000, replace=False)
                        vals = vals[idx]
                    all_values[var].append(vals)
                n_collected += 1

            if args.progress_every and bi % args.progress_every == 0:
                print(f"[{Path(path).name}] batch {bi}/{len(ids)}  "
                      f"collected={n_collected}", flush=True)

    return all_values, n_collected


def process_file_bin(path, out_h5, args, edges, done_set):
    """Process a file and write binned projections to the output HDF5."""
    pg_location = args.particles_location
    pg_prefix = args.particles_prefix

    seed_key = f"{Path(path).name}:{args.sample_seed}".encode()
    rng = np.random.default_rng(zlib.crc32(seed_key))

    kept = 0
    skipped = {"missing_pg": 0, "too_few_alive": 0, "nan_feature": 0,
               "resumed": 0, "subsampled": 0, "input_filter": 0}

    filter_specs = build_filter_specs(args)

    with h5py.File(path, "r") as f:
        ids = batch_ids(f)
        if args.limit_batches:
            ids = ids[: args.limit_batches]
        n_batches = len(ids)

        for bi, uuid in enumerate(ids, start=1):
            g = f[uuid]
            if "observables" not in g:
                continue
            obs = g["observables"]
            n = n_shots(g)
            feats = read_feature_matrix(obs, n)

            if args.wrap_phase:
                for _c in PHASE_WRAP_COLS:
                    _j = FEATURE_COLS.index(_c)
                    feats[:, _j] = wrap_deg(feats[:, _j])

            if pg_location not in obs:
                skipped["missing_pg"] += n
                continue
            loc_grp = obs[pg_location]

            shot_indices = range(n)
            if args.max_shots_per_batch is not None and n > args.max_shots_per_batch:
                shot_indices = sorted(
                    rng.choice(n, size=args.max_shots_per_batch, replace=False))
            elif args.sample_frac is not None and args.sample_frac < 1.0:
                mask = rng.random(n) < args.sample_frac
                shot_indices = [i for i in range(n) if mask[i]]
                skipped["subsampled"] += n - len(shot_indices)

            for i in shot_indices:
                if args.resume and (uuid, i) in done_set:
                    skipped["resumed"] += 1
                    continue

                pg_name = f"{pg_prefix}_{i}"
                if pg_name not in loc_grp:
                    skipped["missing_pg"] += 1
                    continue

                try:
                    pg = load_particlegroup(loc_grp[pg_name])
                except Exception:
                    skipped["missing_pg"] += 1
                    continue

                status = np.asarray(pg.status)
                n_total = int(status.size)
                alive = status == 1
                n_alive = int(np.count_nonzero(alive))
                frac = (n_alive / n_total) if n_total else 0.0

                if args.min_alive_frac is not None and frac < args.min_alive_frac:
                    skipped["too_few_alive"] += 1
                    continue
                if n_alive == 0:
                    skipped["too_few_alive"] += 1
                    continue

                pg_alive_pg = pg[alive] if n_alive < n_total else pg

                feat_row = feats[i]
                if not np.all(np.isfinite(feat_row)):
                    skipped["nan_feature"] += 1
                    continue

                if filter_specs:
                    if any(feat_row[j] < lo or feat_row[j] > hi
                           for j, lo, hi in filter_specs):
                        skipped["input_filter"] += 1
                        continue

                projections = bin_particles(pg_alive_pg, edges)

                # Append to HDF5 datasets
                idx = out_h5["inputs"].shape[0]
                out_h5["inputs"].resize(idx + 1, axis=0)
                out_h5["projections"].resize(idx + 1, axis=0)
                out_h5["provenance_uuid"].resize(idx + 1, axis=0)
                out_h5["provenance_shot"].resize(idx + 1, axis=0)

                out_h5["inputs"][idx] = feat_row.astype(np.float32)
                out_h5["projections"][idx] = projections
                out_h5["provenance_uuid"][idx] = uuid.encode("utf-8")
                out_h5["provenance_shot"][idx] = i

                kept += 1

                if args.probe:
                    print(f"[probe] uuid={uuid} shot={i} "
                          f"n_alive={n_alive} frac={frac:.3f}")
                    print(f"[probe] projection sums: "
                          f"{projections[0].sum():.4f}, "
                          f"{projections[1].sum():.4f}, "
                          f"{projections[2].sum():.4f}")
                    print(f"[probe] max values: "
                          f"{projections[0].max():.6f}, "
                          f"{projections[1].max():.6f}, "
                          f"{projections[2].max():.6f}")
                    print("[probe] OK - one sample written, stopping.")
                    return kept, skipped

            if args.progress_every and bi % args.progress_every == 0:
                out_h5.flush()
                print(f"[{Path(path).name}] batch {bi}/{n_batches}  "
                      f"kept={kept}  skipped={skipped}", flush=True)

    return kept, skipped


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="One or more standard *_571.h5 files")
    ap.add_argument("--output", default=None,
                    help="Output HDF5 path (for binning mode)")
    ap.add_argument("--compute-edges", action="store_true",
                    help="Mode: scan shots and compute global bin edges")
    ap.add_argument("--edges-output", default="bin_edges.npz",
                    help="Where to save computed edges (with --compute-edges)")
    ap.add_argument("--edges-file", default="bin_edges.npz",
                    help="Pre-computed bin edges file (for binning mode)")
    ap.add_argument("--screen", default="571")
    ap.add_argument("--particles-location", default="PR10571")
    ap.add_argument("--particles-prefix", default="571_particles")
    ap.add_argument("--min-alive-frac", type=float, default=0.9)
    ap.add_argument("--input-filter", action="store_true")
    ap.add_argument("--charge-cut", action="store_true")
    ap.add_argument("--wrap-phase", action="store_true")
    ap.add_argument("--campaign", choices=(*CAMPAIGNS, "all"), default="all")
    ap.add_argument("--max-shots-per-batch", type=int, default=None)
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--probe", action="store_true",
                    help="Write one sample and stop (verify setup)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (uuid, shot) pairs already in the output HDF5")
    return ap


def main():
    args = build_parser().parse_args()
    if args.min_alive_frac is not None and args.min_alive_frac <= 0:
        args.min_alive_frac = None
    if args.charge_cut and not args.input_filter:
        print("[WARN] --charge-cut has no effect without --input-filter; ignoring.",
              file=sys.stderr)

    t0 = time.perf_counter()

    # ── Mode 1: Compute bin edges ─────────────────────────────────────────────
    if args.compute_edges:
        print("[run] Mode: computing bin edges", flush=True)
        all_values = {var: [] for var in PHASE_SPACE_VARS}
        total_collected = 0

        for path in args.inputs:
            if not os.path.isfile(path):
                print(f"[WARN] not found, skipping: {path}", file=sys.stderr)
                continue
            if args.campaign != "all" and file_campaign(path) != args.campaign:
                print(f"[skip] {Path(path).name}: wrong campaign", flush=True)
                continue
            print(f"[run] scanning {path}", flush=True)
            file_values, n_collected = process_file_edges(path, args)
            for var in PHASE_SPACE_VARS:
                all_values[var].extend(file_values[var])
            total_collected += n_collected

        print(f"[run] Collected values from {total_collected} shots", flush=True)

        if total_collected == 0:
            raise SystemExit("No shots collected — check filters and file paths.")

        edges = compute_bin_edges(all_values)
        np.savez(args.edges_output, **{var: edges[var] for var in PHASE_SPACE_VARS})
        print(f"[done] Bin edges saved to {args.edges_output}", flush=True)
        for var in PHASE_SPACE_VARS:
            print(f"  {var}: [{edges[var][0]:.6e}, {edges[var][-1]:.6e}]", flush=True)

        dt = time.perf_counter() - t0
        print(f"[all] edge computation done ({dt:.1f}s)", flush=True)
        return

    # ── Mode 2: Bin particles using pre-computed edges ────────────────────────
    if args.output is None:
        raise SystemExit("--output is required in binning mode (omit --compute-edges)")

    if not os.path.isfile(args.edges_file):
        raise SystemExit(f"Edges file not found: {args.edges_file}\n"
                         f"Run with --compute-edges first.")

    edge_data = np.load(args.edges_file)
    edges = {var: edge_data[var] for var in PHASE_SPACE_VARS}
    print(f"[run] Loaded bin edges from {args.edges_file}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing provenance for resume
    done_set = set()
    if args.resume and out_path.is_file():
        with h5py.File(out_path, "r") as existing:
            if "provenance_uuid" in existing:
                uuids = existing["provenance_uuid"][:]
                shots = existing["provenance_shot"][:]
                for u, s in zip(uuids, shots):
                    uid = u.decode("utf-8") if isinstance(u, bytes) else str(u)
                    done_set.add((uid, int(s)))
        print(f"[resume] {len(done_set)} shots already done", flush=True)

    mode = "a" if (args.resume and out_path.is_file()) else "w"
    with h5py.File(out_path, mode) as out_h5:
        if "inputs" not in out_h5:
            out_h5.create_dataset(
                "inputs", shape=(0, len(FEATURE_COLS)), maxshape=(None, len(FEATURE_COLS)),
                dtype=np.float32, chunks=(100, len(FEATURE_COLS)))
            out_h5.create_dataset(
                "projections", shape=(0, 3, N_BINS, N_BINS),
                maxshape=(None, 3, N_BINS, N_BINS),
                dtype=np.float32, chunks=(10, 3, N_BINS, N_BINS))
            dt = h5py.special_dtype(vlen=bytes)
            out_h5.create_dataset(
                "provenance_uuid", shape=(0,), maxshape=(None,), dtype=dt)
            out_h5.create_dataset(
                "provenance_shot", shape=(0,), maxshape=(None,), dtype=np.int32)
            # Store bin edges and metadata
            for var in PHASE_SPACE_VARS:
                out_h5.attrs[f"bin_edges_{var}"] = edges[var]
            out_h5.attrs["n_bins"] = N_BINS
            out_h5.attrs["projection_pairs"] = json.dumps(
                [list(p) for p in PROJECTION_PAIRS])
            out_h5.attrs["feature_cols"] = json.dumps(FEATURE_COLS)

        total_kept = 0
        for path in args.inputs:
            if not os.path.isfile(path):
                print(f"[WARN] not found, skipping: {path}", file=sys.stderr)
                continue
            if args.campaign != "all" and file_campaign(path) != args.campaign:
                print(f"[skip] {Path(path).name}: wrong campaign", flush=True)
                continue
            print(f"[run] processing {path}"
                  + (f"  (resume: {len(done_set)} done)" if done_set else ""),
                  flush=True)
            kept, skipped = process_file_bin(path, out_h5, args, edges, done_set)
            total_kept += kept
            print(f"[done] {Path(path).name}: kept={kept}  skipped={skipped}",
                  flush=True)
            if args.probe:
                break

    dt = time.perf_counter() - t0
    print(f"[all] total kept={total_kept}  -> {out_path}  ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
