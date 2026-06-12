"""Split a prepared dataset CSV into train, validation, and test sets."""
# /Users/gopikab/Documents/modeling/.venv/bin/python split_dataset.py dataset.csv
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(
        description="Split a dataset CSV into train/validation/test CSV files."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the prepared dataset CSV",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Fraction of rows to use for training (default: 0.70)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of rows to use for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of rows to use for testing (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before splitting (default: 42)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling and preserve the original row order",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output CSVs (default: same directory as input)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output file prefix (default: input filename stem)",
    )
    return parser


def validate_fractions(train_fraction, val_fraction, test_fraction):
    fractions = [train_fraction, val_fraction, test_fraction]
    if any(f <= 0 for f in fractions):
        raise ValueError("All split fractions must be positive.")

    total = sum(fractions)
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.6f}."
        )


def main():
    args = build_parser().parse_args()
    validate_fractions(args.train_fraction, args.val_fraction, args.test_fraction)

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix if args.prefix else input_csv.stem

    print(f"[run] Reading: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)
    n_rows = len(df)
    print(f"[run] Loaded {n_rows} rows, {len(df.columns)} columns", flush=True)

    if args.no_shuffle:
        shuffled_df = df
        print("[run] Shuffling disabled; preserving original row order", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        shuffled_df = df.iloc[rng.permutation(n_rows)].reset_index(drop=True)
        print(f"[run] Shuffled rows with seed={args.seed}", flush=True)

    train_end = int(n_rows * args.train_fraction)
    val_end = train_end + int(n_rows * args.val_fraction)

    train_df = shuffled_df.iloc[:train_end].reset_index(drop=True)
    val_df = shuffled_df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = shuffled_df.iloc[val_end:].reset_index(drop=True)

    outputs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for split_name, split_df in outputs.items():
        output_path = output_dir / f"{prefix}-{split_name}.csv"
        split_df.to_csv(output_path, index=False)
        print(
            f"[run] Saved {split_name}: {len(split_df)} rows -> {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()

