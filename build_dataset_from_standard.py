"""Build the 571 covariance-model training dataset from HAAI-data-standard HDF5 files.

Reads one or more `*_571.h5` files stored in the HAAI Data Standard (v0.1.0) format
produced by the FACET-II scraping pipeline and streams a trainable `dataset.csv` shard.

For every shot it:
  1. Reads the 15 scalar control-parameter inputs from `observables/<loc>/<name>`.
  2. Loads the PR10571 ParticleGroup, keeps alive particles (status==1), and applies
     an alive-particle filter (fractional by default, absolute optionally).
  3. Computes the 6x6 phase-space covariance of (x, px, y, py, t, pz), applies the
     M-normalization  C_norm = M @ C @ M^T, and flattens the lower-triangular Cholesky
     factor into 21 `cov_chol_*` targets.
  4. Computes the 6 phase-space means `mean_{x,px,y,py,t,pz}`.

Output columns (exactly what train.py expects: features = all non-target columns):
    18 inputs + mean_x..mean_pz (6) + cov_chol_0..cov_chol_20 (21) = 45 columns.

    Inputs (18):
      - 15 machine-control params (gun, solenoid, quads, linacs)
      - distgen:t_dist:sigma_t:value  (from VCCF)
      - distgen:total_charge:value    (from VCCF)
      - impact_VCC_Cal                (bin_size attribute of VCCF/VCC image)

A sidecar `<output>.provenance.csv` records per-row (source_file, uuid, shot, n_total,
n_alive, alive_frac) for traceability / debugging. It is NOT used for training.

Streaming + resumable: rows are appended shot-by-shot, so memory is constant and a
crashed/timed-out run can be resumed with --resume (skips already-written (uuid, shot)
pairs for the same source file).

Example (single file, quick probe first):
    python build_dataset_from_standard.py \
        /sdf/.../scraped_data/Batching_2_Screen_April_1_571.h5 \
        --output shards/Batching_2_Screen_April_1_571.csv \
        --normalize --min-alive-frac 0.9 --probe

Full single-file run:
    python build_dataset_from_standard.py \
        /sdf/.../scraped_data/Batching_2_Screen_April_1_571.h5 \
        --output shards/Batching_2_Screen_April_1_571.csv \
        --normalize --min-alive-frac 0.9 --progress-every 200 --resume
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import h5py

try:  # data_standard / examples use this name
    from pmd_beamphysics import ParticleGroup
except ImportError:  # old 571 pipeline used this name
    from beamphysics import ParticleGroup


# ── M-normalization (identical to create_targets_from_particles.py) ───────────
M_DIAG = np.array([1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6])
M = np.diag(M_DIAG)

PHASE_SPACE_VARS = ["x", "px", "y", "py", "t", "pz"]
MEAN_COLS = [f"mean_{v}" for v in PHASE_SPACE_VARS]
CHOL_COLS = [f"cov_chol_{i}" for i in range(21)]

# Output feature column name  ->  (observables location group, dataset name).
# Names match the OLD dataset column names so downstream tooling stays consistent.
FEATURE_MAP = [
    ("CQ10121:b1_gradient",           "CQ10121",    "b1_gradient"),
    ("GUNF:rf_field_scale",           "GUNF",       "rf_field_scale"),
    ("GUNF:theta0_deg",               "GUNF",       "theta0_deg"),
    ("L0AF_phase:theta0_deg",         "L0AF_phase", "theta0_deg"),
    ("L0AF_scale:rf_field_scale",     "L0AF_scale", "rf_field_scale"),
    ("L0BF_phase:theta0_deg",         "L0BF_phase", "theta0_deg"),
    ("L0BF_scale:rf_field_scale",     "L0BF_scale", "rf_field_scale"),
    ("QA10361",                       "QA10361",    "QA10361"),
    ("QA10371",                       "QA10371",    "QA10371"),
    ("QE10425",                       "QE10425",    "QE10425"),
    ("QE10441",                       "QE10441",    "QE10441"),
    ("QE10511",                       "QE10511",    "QE10511"),
    ("QE10525",                       "QE10525",    "QE10525"),
    ("SOL10111:solenoid_field_scale", "SOL10111",   "solenoid_field_scale"),
    ("SQ10122:b1_gradient",           "SQ10122",    "b1_gradient"),
    # VCCF scalar inputs (supervisor confirmed locations Jul 2026)
    ("distgen:t_dist:sigma_t:value",  "VCCF",       "distgen:t_dist:sigma_t:value"),
    ("distgen:total_charge:value",    "VCCF",       "distgen:total_charge:value"),
    # NOTE: distgen:VCC is a 2D image stack under VCCF/VCC -- not added here as a scalar.
    # impact_VCC_Cal is the bin_size attribute of VCCF/VCC (read separately below).
]
FEATURE_COLS = [name for name, _, _ in FEATURE_MAP] + ["impact_VCC_Cal"]
DATASET_COLS = FEATURE_COLS + MEAN_COLS + CHOL_COLS
PROVENANCE_COLS = ["source_file", "uuid", "shot", "n_total", "n_alive", "alive_frac"]

# ── Supervisor's per-input range filters (Aug 2026, widened for retraining) ───
# Keyed by FEATURE_COLS names (his impact_/bmad_ prefixes dropped to match). The
# charge cut is kept separate: it is applied ONLY when --charge-cut is passed, so
# the default model generalizes across charge (see README-HAAI-retrain.md).
CHARGE_COL = "distgen:total_charge:value"
INPUT_RANGES: dict[str, tuple[float, float]] = {
    # 241 section inputs
    "GUNF:theta0_deg": (-90.0, -45.0),
    "impact_VCC_Cal": (0.000004, 0.000008),
    "GUNF:rf_field_scale": (40.0e6, 60.0e6),
    "SOL10111:solenoid_field_scale": (0.16, 0.35),
    "CQ10121:b1_gradient": (-0.12, 0.12),
    "SQ10122:b1_gradient": (-0.12, 0.12),
    "distgen:t_dist:sigma_t:value": (0.23, 2.0),
    # L0AFEND section inputs
    "L0AF_phase:theta0_deg": (-20.0, 20.0),
    "L0AF_scale:rf_field_scale": (25.0e6, 35.0e6),
    # 571 section inputs
    "L0BF_phase:theta0_deg": (-20.0, 20.0),
    "L0BF_scale:rf_field_scale": (50.0e6, 70.0e6),
    "QA10361": (1.0, 4.0),
    "QA10371": (-4.0, -1.0),
    "QE10425": (2.0, 8.0),
    "QE10441": (-10.0, -2.0),
    "QE10511": (-1.0, 8.0),
    "QE10525": (-10.0, 4.0),
    # charge added dynamically only when --charge-cut is set:
    CHARGE_COL: (750.0, 2000.0),
}

# Phase inputs wrapped into (-180, 180] when --wrap-phase is set. Fixes the
# L0AF/L0BF phase-wrap artifact (bimodal ~9.5 deg / ~358 deg -> single band).
PHASE_WRAP_COLS = ("GUNF:theta0_deg", "L0AF_phase:theta0_deg", "L0BF_phase:theta0_deg")


def wrap_deg(x: np.ndarray) -> np.ndarray:
    """Wrap degrees into (-180, 180]."""
    return ((x + 180.0) % 360.0) - 180.0


def build_filter_specs(args):
    """Return list of (col_index, lo, hi) for the active input-range filters.

    Empty when --input-filter is off. Excludes the charge cut unless --charge-cut
    is also set.
    """
    if not args.input_filter:
        return []
    specs = []
    for col, (lo, hi) in INPUT_RANGES.items():
        if col == CHARGE_COL and not args.charge_cut:
            continue
        specs.append((FEATURE_COLS.index(col), lo, hi))
    return specs


def normalize_covariance(cov: np.ndarray) -> np.ndarray:
    """Apply M @ cov @ M^T normalization."""
    return M @ cov @ M.T


def cholesky_lower_vector(cov: np.ndarray) -> np.ndarray:
    """Cholesky-decompose and return the 21 lower-triangular entries (row-major)."""
    chol = np.linalg.cholesky(np.asarray(cov, dtype=float))
    return chol[np.tril_indices(chol.shape[0])]


def load_particlegroup(parent_grp):
    """Read a ParticleGroup written by pmd_beamphysics `pg.write(group)`.

    On disk the shot lives at `<location>/<name>_<i>/` and (per the inspected files)
    contains an openPMD species subgroup `electron/{position,momentum,particleStatus,
    time,weight}`. pmd_beamphysics reads that species group directly, so prefer it;
    fall back to the parent group for other layouts.
    """
    for species in ("electron", "positron", "proton"):
        if species in parent_grp:
            return ParticleGroup(h5=parent_grp[species])
    return ParticleGroup(h5=parent_grp)


def batch_ids(f):
    """Ordered list of batch (UUID) group names."""
    if "IDs" in f.attrs:
        return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                for x in f.attrs["IDs"]]
    return [k for k in f.keys() if k != "lattice"]


def n_shots(batch_grp):
    bd = np.asarray(batch_grp.attrs["batch_dims"]).ravel()
    return 1 if bd.size == 0 else int(np.prod(bd))


def read_feature_matrix(obs, n):
    """Return (n, 17) float array of the control-param inputs for this batch.

    Columns: 15 machine-control params + distgen:t_dist:sigma_t:value +
    distgen:total_charge:value (both from VCCF) + impact_VCC_Cal (from VCCF/VCC
    bin_size attribute, broadcast to all n shots).

    Missing feature -> column of NaN (row later dropped). Handles both batched
    (shape (n,)) and single-shot (scalar) datasets.
    """
    cols = []
    for _name, loc, dset in FEATURE_MAP:
        vals = np.full(n, np.nan, dtype=np.float64)
        if loc in obs and dset in obs[loc]:
            ds = obs[loc][dset]
            arr = np.asarray(ds[()], dtype=np.float64).ravel()
            if arr.size == n:
                vals = arr
            elif arr.size == 1 and n == 1:
                vals = arr
            elif arr.size == 1:
                # scalar shared across all shots in the batch
                vals = np.full(n, float(arr[0]), dtype=np.float64)
        cols.append(vals)

    # impact_VCC_Cal: taken from the bin_size attribute of VCCF/VCC, multiplied by
    # unit_multiplier. As of the 2026-09-18 regeneration, bin_size may be either a
    # scalar (legacy, one value per UUID broadcast to all shots) OR a string naming
    # an observable dataset with per-shot values (shape (n,)). The observable
    # reference is resolved first against VCCF/<name>, then against the observables
    # root using a "/"-separated path.
    vcc_cal = np.full(n, np.nan, dtype=np.float64)
    if "VCCF" in obs and "VCC" in obs["VCCF"]:
        vcc_ds = obs["VCCF"]["VCC"]
        raw = vcc_ds.attrs.get("bin_size", None)
        mult = float(vcc_ds.attrs.get("unit_multiplier", 1.0))
        if raw is not None:
            resolved = None
            # Case 1: bin_size names an observable dataset (per-shot values).
            if isinstance(raw, (bytes, bytearray)) or isinstance(raw, str):
                ref = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
                ref = ref.strip()
                target = None
                if ref in obs["VCCF"]:
                    target = obs["VCCF"][ref]
                elif ref in obs:
                    target = obs[ref]
                elif "/" in ref and ref in obs:
                    target = obs[ref]
                if target is not None:
                    arr = np.asarray(target[()], dtype=np.float64).ravel()
                    if arr.size == n:
                        resolved = arr
                    elif arr.size == 1:
                        resolved = np.full(n, float(arr[0]), dtype=np.float64)
                if resolved is None:
                    # String that doesn't resolve — try parsing it as a number
                    try:
                        resolved = np.full(n, float(ref), dtype=np.float64)
                    except (TypeError, ValueError):
                        resolved = None
            # Case 2: bin_size is a numeric scalar / array (legacy format).
            else:
                arr = np.asarray(raw, dtype=np.float64).ravel()
                if arr.size == n:
                    resolved = arr
                elif arr.size == 1:
                    resolved = np.full(n, float(arr[0]), dtype=np.float64)
            if resolved is not None:
                vcc_cal = resolved * mult
    cols.append(vcc_cal)

    return np.column_stack(cols)


# ── Campaign classification (per supervisor, filename-based) ──────────────────
# The real VCC-acquisition dates are NOT in the file attrs (those are scrape
# dates). Supervisor's rule for mapping a *_571.h5 filename to a campaign:
#   - filename contains "2025"  -> January 2025  ("doughnut" laser spot)
#   - filename contains "April" -> April 2024    ("other")
#   - neither                   -> January 2024  ("line"; the most data)
# NOTE: always eyeball the VCC images after filtering to confirm (labels were
# not originally intended as campaign tags).
CAMPAIGNS = ("jan2024", "jan2025", "april2024")


def file_campaign(path):
    """Classify a *_571.h5 file into a campaign from its filename."""
    name = Path(path).name.lower()
    if "2025" in name:
        return "jan2025"
    if "april" in name:
        return "april2024"
    return "jan2024"


def load_done_set(prov_path, source_file):
    """Return set of (uuid, shot) already written for this source file (for --resume)."""
    done = set()
    if not os.path.isfile(prov_path):
        return done
    import csv
    with open(prov_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("source_file") == source_file:
                done.add((row["uuid"], int(row["shot"])))
    return done


def process_file(path, out_fh, prov_fh, args, done):
    """Stream one standard _571.h5 file, appending kept rows to open file handles."""
    screen = args.screen
    pg_location = args.particles_location
    pg_prefix = args.particles_prefix

    # Deterministic per-file RNG so subsampling is reproducible and resume-safe
    # (the same shots are chosen on a resumed run or a re-submitted array task).
    # zlib.crc32 is stable across processes, unlike Python's hash() (PYTHONHASHSEED).
    seed_key = f"{Path(path).name}:{args.sample_seed}".encode()
    rng = np.random.default_rng(zlib.crc32(seed_key))

    kept = 0
    skipped = {"missing_pg": 0, "too_few_alive": 0, "bad_cov": 0,
               "nan_feature": 0, "resumed": 0, "subsampled": 0,
               "vcc_filter": 0, "input_filter": 0}

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

            # Wrap phase inputs before filtering/writing so the range cuts and the
            # stored dataset both use the single-band representation.
            if args.wrap_phase:
                for _c in PHASE_WRAP_COLS:
                    _j = FEATURE_COLS.index(_c)
                    feats[:, _j] = wrap_deg(feats[:, _j])

            if pg_location not in obs:
                skipped["missing_pg"] += n
                continue
            loc_grp = obs[pg_location]

            # Choose which shots in this batch to keep (uniform subsampling).
            shot_indices = range(n)
            if args.max_shots_per_batch is not None and n > args.max_shots_per_batch:
                shot_indices = sorted(
                    rng.choice(n, size=args.max_shots_per_batch, replace=False))
            elif args.sample_frac is not None and args.sample_frac < 1.0:
                mask = rng.random(n) < args.sample_frac
                shot_indices = [i for i in range(n) if mask[i]]
                skipped["subsampled"] += n - len(shot_indices)

            for i in shot_indices:
                if args.resume and (uuid, i) in done:
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

                # Alive-particle filter (fractional preferred; absolute optional).
                if args.min_alive_frac is not None and frac < args.min_alive_frac:
                    skipped["too_few_alive"] += 1
                    continue
                if (args.min_alive_particles is not None
                        and n_alive < args.min_alive_particles):
                    skipped["too_few_alive"] += 1
                    continue
                if n_alive == 0:
                    skipped["too_few_alive"] += 1
                    continue

                pg_alive = pg[alive] if n_alive < n_total else pg

                try:
                    cov = np.asarray(
                        pg_alive.cov("x", "px", "y", "py", "t", "pz"), dtype=float)
                    if args.normalize:
                        cov = normalize_covariance(cov)
                    chol = cholesky_lower_vector(cov)
                    means = [float(pg_alive.avg(v)) for v in PHASE_SPACE_VARS]
                except Exception:
                    skipped["bad_cov"] += 1
                    continue

                feat_row = feats[i]
                if not np.all(np.isfinite(feat_row)):
                    skipped["nan_feature"] += 1
                    continue

                # Supervisor's per-input range filter (opt-in via --input-filter;
                # charge cut only when --charge-cut is also set).
                if filter_specs:
                    if any(feat_row[j] < lo or feat_row[j] > hi
                           for j, lo, hi in filter_specs):
                        skipped["input_filter"] += 1
                        continue

                # Optional domain filter: keep only shots with VCC calibration in range.
                vcc_cal = float(feat_row[FEATURE_COLS.index("impact_VCC_Cal")])
                if args.vcc_cal_min is not None and vcc_cal < args.vcc_cal_min:
                    skipped["vcc_filter"] += 1
                    continue
                if args.vcc_cal_max is not None and vcc_cal > args.vcc_cal_max:
                    skipped["vcc_filter"] += 1
                    continue

                row = np.concatenate([feat_row, np.asarray(means), chol])
                out_fh.write(",".join(f"{v:.10g}" for v in row) + "\n")
                prov_fh.write(
                    f"{Path(path).name},{uuid},{i},{n_total},{n_alive},{frac:.6f}\n")
                kept += 1

                if args.probe:
                    out_fh.flush()
                    prov_fh.flush()
                    print(f"[probe] shot uuid={uuid} shot={i} "
                          f"n_total={n_total} n_alive={n_alive} frac={frac:.3f}")
                    print(f"[probe] means={means}")
                    print(f"[probe] chol[0:3]={chol[:3]}")
                    print("[probe] OK - one row written, stopping.")
                    return kept, skipped

            if args.progress_every and bi % args.progress_every == 0:
                out_fh.flush()
                prov_fh.flush()
                print(f"[{Path(path).name}] batch {bi}/{n_batches}  "
                      f"kept={kept}  skipped={skipped}", flush=True)

    return kept, skipped


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="One or more standard *_571.h5 files")
    ap.add_argument("--output", required=True, help="Output dataset shard CSV path")
    ap.add_argument("--screen", default="571", help="Screen id (default: 571)")
    ap.add_argument("--particles-location", default="PR10571",
                    help="Observables location holding the target PGs (default: PR10571)")
    ap.add_argument("--particles-prefix", default="571_particles",
                    help="Per-shot PG group name prefix (default: 571_particles)")
    ap.add_argument("--normalize", action="store_true",
                    help="Apply M-normalization C_norm = M C M^T before Cholesky "
                         "(M=diag(1e3,1e-6,1e3,1e-6,1e12,1e-6)). Match training convention.")
    ap.add_argument("--min-alive-frac", type=float, default=0.9,
                    help="Keep shots whose alive fraction (status==1 / total in the 571 PG) "
                         ">= this. Robust across files with different particle counts. "
                         "Default: 0.9. Set to 0 to disable.")
    ap.add_argument("--min-alive-particles", type=int, default=None,
                    help="Optional ABSOLUTE alive-count threshold (old behavior, e.g. 90000). "
                         "Applied in addition to --min-alive-frac if both given.")
    ap.add_argument("--progress-every", type=int, default=200,
                    help="Print progress every N batches (default: 200)")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="Only process the first N batches per file (debug)")
    ap.add_argument("--sample-frac", type=float, default=None,
                    help="Uniformly keep this fraction of shots per batch (0<f<=1). "
                         "Reduces I/O and dataset size while preserving the input "
                         "distribution. Applied per file with a deterministic RNG.")
    ap.add_argument("--max-shots-per-batch", type=int, default=None,
                    help="Keep at most this many shots per 50-shot batch (overrides "
                         "--sample-frac). E.g. 10 -> ~1/5 of the data, evenly across files.")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="Seed for reproducible shot subsampling (default: 0)")
    ap.add_argument("--campaign", choices=(*CAMPAIGNS, "all"), default="all",
                    help="Keep only input files from one laser-spot campaign, classified "
                         "by filename: 'jan2025' (contains '2025', doughnut), 'april2024' "
                         "(contains 'April'), 'jan2024' (neither, line; most data). "
                         "Default: 'all' (no campaign filtering). Always verify with the "
                         "VCC images after filtering.")
    ap.add_argument("--vcc-cal-min", type=float, default=None,
                    help="Optional lower bound for impact_VCC_Cal (meters). "
                         "Use with --vcc-cal-max to restrict to a consistent VCC domain.")
    ap.add_argument("--vcc-cal-max", type=float, default=None,
                    help="Optional upper bound for impact_VCC_Cal (meters). "
                         "Use with --vcc-cal-min to restrict to a consistent VCC domain.")
    ap.add_argument("--input-filter", action="store_true",
                    help="Apply the supervisor's per-input range cuts (Aug 2026, "
                         "INPUT_RANGES). Rejects shots with any control input out of "
                         "range. The charge cut is EXCLUDED unless --charge-cut is also "
                         "given, so the default model generalizes across charge.")
    ap.add_argument("--charge-cut", action="store_true",
                    help="Also enforce the distgen:total_charge:value range (750-2000 pC). "
                         "Off by default (full charge span kept). Requires --input-filter.")
    ap.add_argument("--wrap-phase", action="store_true",
                    help="Wrap phase inputs (GUNF/L0AF/L0BF theta0_deg) into (-180,180] "
                         "before filtering and writing. Fixes the L0AF/L0BF phase-wrap "
                         "artifact. Recommended for all HAAI retrains.")
    ap.add_argument("--probe", action="store_true",
                    help="Process only until the first row is written, print diagnostics, "
                         "and exit. Use to validate a file/env cheaply before a full run.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (uuid, shot) pairs already present in the provenance sidecar "
                         "for the same source file, and append.")
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
    prov_path = str(out_path) + ".provenance.csv"

    resume = args.resume and out_path.is_file() and os.path.isfile(prov_path)
    mode = "a" if resume else "w"

    t0 = time.perf_counter()
    total_kept = 0
    with open(out_path, mode) as out_fh, open(prov_path, mode) as prov_fh:
        if not resume:
            out_fh.write(",".join(DATASET_COLS) + "\n")
            prov_fh.write(",".join(PROVENANCE_COLS) + "\n")

        for path in args.inputs:
            if not os.path.isfile(path):
                print(f"[WARN] not found, skipping: {path}", file=sys.stderr)
                continue
            if args.campaign != "all" and file_campaign(path) != args.campaign:
                print(f"[skip] {Path(path).name}: campaign={file_campaign(path)} "
                      f"!= --campaign {args.campaign}", flush=True)
                continue
            done = load_done_set(prov_path, Path(path).name) if resume else set()
            print(f"[run] processing {path}"
                  + (f"  (resume: {len(done)} shots already done)" if done else ""),
                  flush=True)
            kept, skipped = process_file(path, out_fh, prov_fh, args, done)
            total_kept += kept
            print(f"[done] {Path(path).name}: kept={kept}  skipped={skipped}", flush=True)
            if args.probe:
                break

    dt = time.perf_counter() - t0
    print(f"[all] total kept rows={total_kept}  -> {out_path}  ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
