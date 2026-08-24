"""Build particle-level dataset for conditional normalizing flow training.

For each shot, subsamples N particles (default 2048) and stores the raw 6D
phase-space coordinates (x, px, y, py, t, pz) along with the 18 conditioning
inputs. This avoids any binning or covariance reduction — the flow learns the
full distribution directly.

Uses the same filtering logic as build_dataset_from_standard.py.

Example:
    python build_particle_dataset.py \
        /sdf/.../2026-07-23/*_571.h5 \
        --output particles_flow.h5 \
        --n-particles 2048 \
        --min-alive-frac 0.9 --wrap-phase --input-filter \
        --max-shots-per-batch 10 --progress-every 200
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


def extract_particles(pg_alive, n_particles: int, rng) -> np.ndarray | None:
    """Extract n_particles from a ParticleGroup, subsampling if needed.

    Returns (n_particles, 6) float64 array, or None if not enough particles.
    """
    n_available = len(pg_alive.x)
    if n_available < 10:
        return None

    coords = np.column_stack([getattr(pg_alive, var) for var in PHASE_SPACE_VARS])

    if n_available >= n_particles:
        idx = rng.choice(n_available, size=n_particles, replace=False)
        return coords[idx].astype(np.float64)
    else:
        # Fewer particles than requested — use all (pad with NaN)
        padded = np.full((n_particles, 6), np.nan, dtype=np.float64)
        padded[:n_available] = coords
        return padded


def process_file(path, out_h5, args, done_set):
    """Process one HDF5 file, appending particle samples to output."""
    pg_location = args.particles_location
    pg_prefix = args.particles_prefix
    n_particles = args.n_particles

    seed_key = f"{Path(path).name}:{args.sample_seed}".encode()
    rng = np.random.default_rng(zlib.crc32(seed_key))

    kept = 0
    skipped = {"missing_pg": 0, "too_few_alive": 0, "nan_feature": 0,
               "subsampled": 0, "input_filter": 0}

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

                feat_row = feats[i]
                if not np.all(np.isfinite(feat_row)):
                    skipped["nan_feature"] += 1
                    continue

                if filter_specs:
                    if any(feat_row[j] < lo or feat_row[j] > hi
                           for j, lo, hi in filter_specs):
                        skipped["input_filter"] += 1
                        continue

                pg_alive = pg[alive] if n_alive < n_total else pg
                particles = extract_particles(pg_alive, n_particles, rng)
                if particles is None:
                    skipped["too_few_alive"] += 1
                    continue

                # Append to HDF5
                idx = out_h5["inputs"].shape[0]
                out_h5["inputs"].resize(idx + 1, axis=0)
                out_h5["particles"].resize(idx + 1, axis=0)
                out_h5["provenance_uuid"].resize(idx + 1, axis=0)
                out_h5["provenance_shot"].resize(idx + 1, axis=0)
                out_h5["n_alive"].resize(idx + 1, axis=0)

                out_h5["inputs"][idx] = feat_row.astype(np.float32)
                out_h5["particles"][idx] = particles.astype(np.float32)
                out_h5["provenance_uuid"][idx] = uuid.encode("utf-8")
                out_h5["provenance_shot"][idx] = i
                out_h5["n_alive"][idx] = n_alive

                kept += 1

                if args.probe:
                    print(f"[probe] uuid={uuid} shot={i} "
                          f"n_alive={n_alive} frac={frac:.3f}")
                    print(f"[probe] particles shape: {particles.shape}")
                    print(f"[probe] x range: [{particles[:,0].min():.6e}, "
                          f"{particles[:,0].max():.6e}]")
                    print(f"[probe] pz range: [{particles[:,5].min():.6e}, "
                          f"{particles[:,5].max():.6e}]")
                    nan_count = np.isnan(particles).any(axis=1).sum()
                    print(f"[probe] NaN-padded rows: {nan_count}")
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
    ap.add_argument("inputs", nargs="+", help="One or more *_571.h5 files")
    ap.add_argument("--output", required=True, help="Output HDF5 path")
    ap.add_argument("--n-particles", type=int, default=2048,
                    help="Particles to subsample per shot (default: 2048)")
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
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--resume", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    if args.min_alive_frac is not None and args.min_alive_frac <= 0:
        args.min_alive_frac = None
    if args.charge_cut and not args.input_filter:
        print("[WARN] --charge-cut has no effect without --input-filter; ignoring.",
              file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
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
    n_particles = args.n_particles
    n_features = len(FEATURE_COLS)

    t0 = time.perf_counter()
    total_kept = 0

    with h5py.File(out_path, mode) as out_h5:
        if "inputs" not in out_h5:
            out_h5.create_dataset(
                "inputs", shape=(0, n_features), maxshape=(None, n_features),
                dtype=np.float32, chunks=(100, n_features))
            out_h5.create_dataset(
                "particles", shape=(0, n_particles, 6),
                maxshape=(None, n_particles, 6),
                dtype=np.float32, chunks=(10, n_particles, 6))
            dt = h5py.special_dtype(vlen=bytes)
            out_h5.create_dataset(
                "provenance_uuid", shape=(0,), maxshape=(None,), dtype=dt)
            out_h5.create_dataset(
                "provenance_shot", shape=(0,), maxshape=(None,), dtype=np.int32)
            out_h5.create_dataset(
                "n_alive", shape=(0,), maxshape=(None,), dtype=np.int32)
            # Metadata
            out_h5.attrs["n_particles"] = n_particles
            out_h5.attrs["phase_space_vars"] = json.dumps(PHASE_SPACE_VARS)
            out_h5.attrs["feature_cols"] = json.dumps(FEATURE_COLS)

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
            kept, skipped = process_file(path, out_h5, args, done_set)
            total_kept += kept
            print(f"[done] {Path(path).name}: kept={kept}  skipped={skipped}",
                  flush=True)
            if args.probe:
                break

    dt = time.perf_counter() - t0
    print(f"[all] total kept={total_kept}  -> {out_path}  ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
