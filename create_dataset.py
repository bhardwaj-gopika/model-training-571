"""Create the ML training dataset from the approved input and target columns."""
import argparse
from pathlib import Path

import pandas as pd

# Input columns for 571 model
INPUT_COLUMNS = [
    "CQ10121:b1_gradient", #QUAD:IN10:121:BACT
    "GUNF:rf_field_scale", #KLYS:LI10:21:AMPL
    "GUNF:theta0_deg", #KLYS:LI10:21:PHAS
    "L0AF_phase:theta0_deg", #KLYS:LI10:31:PHAS
    "L0AF_scale:rf_field_scale",#null
    "L0BF_phase:theta0_deg",#KLYS:LI10:41:PHAS
    "L0BF_scale:rf_field_scale",#null
    "QA10361",#QUAD:IN10:361:BACT
    "QA10371",#QUAD:IN10:371:BACT
    "QE10425",#QUAD:IN10:425:BACT
    "QE10441",#QUAD:IN10:441:BACT
    "QE10511",#QUAD:IN10:511:BACT
    "QE10525",#QUAD:IN10:525:BACT
    "SOL10111:solenoid_field_scale",#SOLN:IN10:121:BACT
    "SQ10122:b1_gradient",#QUAD:IN10:122:BACT
    "distgen:VCC",
    "distgen:t_dist:sigma_t:value",#null
    "distgen:total_charge:value",#TORO:IN10:591:TMIT_PC
    "impact_VCC_Cal"#null
]

CHOL_TARGET_COLUMNS = [f"cov_chol_{index}" for index in range(21)]
MEAN_TARGET_COLUMNS = ["mean_energy", "mean_time"]
TARGET_COLUMNS = MEAN_TARGET_COLUMNS + CHOL_TARGET_COLUMNS

REQUIRED_COLUMNS = INPUT_COLUMNS + TARGET_COLUMNS


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create ML training dataset from the approved input and target columns."
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV (e.g. dump-particles_241-cov-targets.csv from the cov targets script)",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="dataset.csv",
        help="Output dataset CSV path (default: dataset.csv)",
    )
    parser.add_argument(
        "--drop-null-rows",
        action="store_true",
        help="Drop any remaining rows with null values after column filtering",
    )
    return parser


def main():
    args = build_parser().parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    print(f"[run] Reading: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)
    rows_in = len(df)
    cols_in = len(df.columns)
    print(f"[run] Loaded {rows_in} rows, {cols_in} columns", flush=True)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    print(f"[run] Selected {len(REQUIRED_COLUMNS)} approved columns", flush=True)

    if args.drop_null_rows:
        rows_before_dropna = len(df)
        df = df.dropna(axis=0, how="any")
        dropped = rows_before_dropna - len(df)
        print(f"[run] Dropped {dropped} null rows ({len(df)} remaining)", flush=True)

    print(f"[run] Final dataset: {len(df)} rows, {len(df.columns)} columns", flush=True)
    print(f"[run] Remaining columns:\n  " + "\n  ".join(df.columns.tolist()), flush=True)

    df.to_csv(output_csv, index=False)
    print(f"[run] Saved to {output_csv}", flush=True)


if __name__ == "__main__":
    main()

