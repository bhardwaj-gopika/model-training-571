"""Inspect a HAAI-data-standard HDF5 file structure without loading particle data.

Usage:
    python inspect_standard_h5.py /path/to/Some_571.h5
    python inspect_standard_h5.py /path/to/Some_571.h5 --max-batches 2 --max-depth 6

Prints:
  - root attributes (Data_Standard_Version, IDs)
  - lattice group summary
  - for the first few batch groups:
      * batch attributes (batch_dims, simulation_*, run_information_*)
      * every observables location and its datasets (shape, dtype, units, control)
      * multi_location_data datasets + DATA_LOCATIONS
      * where ParticleGroups live (group names only, no data read)
"""
from __future__ import annotations

import argparse
import numpy as np
import h5py


def dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x


def show_attrs(obj, indent):
    for k, v in obj.attrs.items():
        if isinstance(v, (bytes, bytearray)):
            v = dec(v)
        elif isinstance(v, np.ndarray):
            flat = [dec(x) for x in v.ravel()[:12]]
            v = f"array{v.shape} {flat}{'...' if v.size > 12 else ''}"
        print(f"{indent}@{k} = {v}")


def is_particlegroup(g):
    """Heuristic: openPMD ParticleGroup groups usually contain a species subgroup
    with position/momentum, or an @openPMD attr, or 'electron' subgroup."""
    if not isinstance(g, h5py.Group):
        return False
    keys = set(g.keys())
    if keys & {"electron", "positron", "proton"}:
        return True
    if "position" in keys and "momentum" in keys:
        return True
    return False


def walk(obj, indent="  ", depth=0, max_depth=6, pg_seen=None):
    if pg_seen is None:
        pg_seen = set()
    if depth > max_depth:
        print(f"{indent}... (max depth)")
        return
    for name, item in obj.items():
        if isinstance(item, h5py.Dataset):
            units = dec(item.attrs.get("units", ""))
            ctrl = item.attrs.get("control", "")
            nfd = item.attrs.get("num_feature_dims", "")
            print(f"{indent}{name}  [DATASET shape={item.shape} dtype={item.dtype}] "
                  f"units={units!r} control={ctrl} nfd={nfd}")
        else:
            if is_particlegroup(item):
                print(f"{indent}{name}/  <-- ParticleGroup (subkeys: {list(item.keys())})")
                continue
            # Collapse repetitive per-shot PG containers: show first, count rest
            print(f"{indent}{name}/")
            show_attrs(item, indent + "  ")
            children = list(item.items())
            # If children look like many numbered PG shots, summarize
            numbered = [c for c in children if c[0].split("_")[-1].isdigit()
                        or c[0].isdigit()]
            non_numbered = [c for c in children if c not in numbered]
            if len(numbered) > 3 and all(isinstance(c[1], h5py.Group) for c in numbered):
                print(f"{indent}  ({len(numbered)} numbered subgroups, showing first)")
                walk_one = dict([numbered[0]])
                for nm, it in walk_one.items():
                    print(f"{indent}  {nm}/")
                    walk(it, indent + "    ", depth + 2, max_depth, pg_seen)
                # Also show any non-numbered datasets/groups (scalars, images, etc.)
                if non_numbered:
                    print(f"{indent}  -- non-numbered items also in this group:")
                    for nm, it in non_numbered:
                        if isinstance(it, h5py.Dataset):
                            units = dec(it.attrs.get("units", ""))
                            ctrl = it.attrs.get("control", "")
                            nfd = it.attrs.get("num_feature_dims", "")
                            bin_size = it.attrs.get("bin_size", "")
                            print(f"{indent}  {nm}  [DATASET shape={it.shape} dtype={it.dtype}] "
                                  f"units={units!r} control={ctrl} nfd={nfd}"
                                  + (f" bin_size={bin_size}" if bin_size != "" else ""))
                            print(f"{indent}    attrs: { {k: dec(v) if isinstance(v,(bytes,bytearray)) else v for k,v in it.attrs.items()} }")
                        else:
                            print(f"{indent}  {nm}/  [group]")
            else:
                walk(item, indent + "  ", depth + 1, max_depth, pg_seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-batches", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        print("=" * 70)
        print(f"FILE: {args.path}")
        print("=" * 70)
        print("ROOT ATTRIBUTES:")
        show_attrs(f, "  ")

        print("\nTOP-LEVEL KEYS:", list(f.keys()))

        if "lattice" in f:
            print("\nLATTICE:")
            show_attrs(f["lattice"], "  ")
            if "lattice_files" in f["lattice"]:
                print("  lattice_files:", list(f["lattice"]["lattice_files"].keys()))

        ids = [dec(x) for x in f.attrs["IDs"]] if "IDs" in f.attrs else \
            [k for k in f.keys() if k != "lattice"]
        print(f"\nNUM BATCHES: {len(ids)}")

        for bi, bid in enumerate(ids[: args.max_batches]):
            print("\n" + "-" * 70)
            print(f"BATCH [{bi}] id={bid}")
            print("-" * 70)
            g = f[bid]
            show_attrs(g, "  ")
            if "observables" in g:
                print("  observables/")
                walk(g["observables"], "    ", 0, args.max_depth)


if __name__ == "__main__":
    main()
