"""Split the projection HDF5 dataset into train/val/test aligned with the covariance splits.

Uses the provenance sidecar CSVs from the covariance pipeline to determine which
(uuid, shot) pairs belong to each split. If no provenance files exist, falls back
to a random 70/15/15 split with the same seed as split_dataset.py.

Example:
    python split_projection_dataset.py \
        --input projections_dataset.h5 \
        --train-prov dataset-train.csv.provenance.csv \
        --val-prov dataset-val.csv.provenance.csv \
        --test-prov dataset-test.csv.provenance.csv \
        --output-dir .

    # Produces: projections_train.h5, projections_val.h5, projections_test.h5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import h5py


def load_provenance_keys(prov_path: str) -> set[tuple[str, int]]:
    """Load (uuid, shot) pairs from a provenance CSV."""
    import csv
    keys = set()
    with open(prov_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row["uuid"], int(row["shot"])))
    return keys


def copy_subset(src_h5, indices: np.ndarray, dst_path: str):
    """Write a subset of the source HDF5 to a new file."""
    with h5py.File(dst_path, "w") as dst:
        # Copy datasets at selected indices
        dst.create_dataset("inputs", data=src_h5["inputs"][indices])
        dst.create_dataset("projections", data=src_h5["projections"][indices])

        # Provenance
        uuids = src_h5["provenance_uuid"][indices]
        shots = src_h5["provenance_shot"][indices]
        dt = h5py.special_dtype(vlen=bytes)
        dst.create_dataset("provenance_uuid", data=uuids, dtype=dt)
        dst.create_dataset("provenance_shot", data=shots)

        # Copy attributes (bin edges, metadata)
        for key, val in src_h5.attrs.items():
            dst.attrs[key] = val

    print(f"  -> {dst_path}: {len(indices)} samples", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Full projection HDF5 dataset")
    ap.add_argument("--train-prov", default=None,
                    help="Provenance CSV for training split")
    ap.add_argument("--val-prov", default=None,
                    help="Provenance CSV for validation split")
    ap.add_argument("--test-prov", default=None,
                    help="Provenance CSV for test split")
    ap.add_argument("--output-dir", default=".",
                    help="Directory for output split files")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for fallback split (default: 42)")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as src:
        n_total = src["inputs"].shape[0]
        print(f"[run] Input dataset: {n_total} samples", flush=True)

        # Try provenance-based splitting
        has_prov = (args.train_prov and os.path.isfile(args.train_prov) and
                    args.val_prov and os.path.isfile(args.val_prov) and
                    args.test_prov and os.path.isfile(args.test_prov))

        if has_prov:
            print("[run] Splitting by provenance alignment", flush=True)
            train_keys = load_provenance_keys(args.train_prov)
            val_keys = load_provenance_keys(args.val_prov)
            test_keys = load_provenance_keys(args.test_prov)

            # Read provenance from HDF5
            uuids_raw = src["provenance_uuid"][:]
            shots_raw = src["provenance_shot"][:]

            train_idx, val_idx, test_idx = [], [], []
            unmatched = 0
            for i in range(n_total):
                uid = uuids_raw[i]
                if isinstance(uid, bytes):
                    uid = uid.decode("utf-8")
                shot = int(shots_raw[i])
                key = (uid, shot)

                if key in train_keys:
                    train_idx.append(i)
                elif key in val_keys:
                    val_idx.append(i)
                elif key in test_keys:
                    test_idx.append(i)
                else:
                    unmatched += 1

            train_idx = np.array(train_idx)
            val_idx = np.array(val_idx)
            test_idx = np.array(test_idx)

            if unmatched > 0:
                print(f"[WARN] {unmatched} samples not found in any provenance file "
                      f"(different filtering?). These are dropped.", flush=True)

        else:
            print("[run] No provenance files found — using random 70/15/15 split",
                  flush=True)
            rng = np.random.default_rng(args.seed)
            perm = rng.permutation(n_total)
            n_train = int(n_total * args.train_frac)
            n_val = int(n_total * args.val_frac)

            train_idx = np.sort(perm[:n_train])
            val_idx = np.sort(perm[n_train:n_train + n_val])
            test_idx = np.sort(perm[n_train + n_val:])

        print(f"[run] Split: train={len(train_idx)}, val={len(val_idx)}, "
              f"test={len(test_idx)}", flush=True)

        copy_subset(src, train_idx, str(output_dir / "projections_train.h5"))
        copy_subset(src, val_idx, str(output_dir / "projections_val.h5"))
        copy_subset(src, test_idx, str(output_dir / "projections_test.h5"))

    print("[done] Split complete.", flush=True)


if __name__ == "__main__":
    main()
