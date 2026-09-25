#!/bin/bash
# SLURM array job: build one dataset shard per HAAI-standard *_571.h5 file.
#
# Each array task processes ONE 571 file into shards/<name>.csv (streaming, resumable).
# Reading the PR10571 ParticleGroups across 17 TB is I/O-bound, not GPU work, so this
# runs on a CPU partition. Parallelism = one task per file.
#
# Usage:
#   1. Edit DATA_DIR / OUTPUT_DIR / conda env below if needed.
#   2. Generate the file list (571 files only) and count them:
#         ls /sdf/data/ad/ard-online/FACET-II_Training_Data/scraped_data/*_571.h5 > file_list_571.txt
#         wc -l file_list_571.txt         # -> set --array=0-(N-1) below
#   3. Submit:
#         sbatch --array=0-$(( $(wc -l < file_list_571.txt) - 1 ))%8 make_dataset.sh
#      (%8 caps concurrent tasks at 8; tune to your I/O budget.)
#   4. After all shards finish, concatenate:
#         bash concat_shards.sh
#
#SBATCH --job-name=build571
#SBATCH --account=ad:ard-online
#SBATCH --partition=milano          # CPU partition (no GPU needed)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/build571_%A_%a.out

set -euo pipefail

DATA_DIR=/sdf/data/ad/ard-online/FACET-II_Training_Data/scraped_data
OUTPUT_DIR=${OUTPUT_DIR:-shards}
FILE_LIST=file_list_571.txt

# Uniform shot subsampling. Each batch has 50 shots; keeping 10 -> ~1/5 of the data
# (~300k shots across all 21 files, ~6x the old training set) at a fraction of the I/O.
# Set to 50 (or comment out the flag below) to use every shot.
SHOTS_PER_BATCH=${SHOTS_PER_BATCH:-10}

# HAAI retrain filters (toggle via env). Defaults = option 2 (charge-generalizing):
#   phase wrap ON, supervisor input-range filter ON, charge cut OFF.
# For the 1-nC A/B baseline, submit with CHARGE_CUT=1 and a separate OUTPUT_DIR, e.g.
#   OUTPUT_DIR=shards-1nc CHARGE_CUT=1 sbatch --array=0-20%8 make_dataset.sh
WRAP_PHASE=${WRAP_PHASE:-1}
INPUT_FILTER=${INPUT_FILTER:-1}
CHARGE_CUT=${CHARGE_CUT:-0}

FILTER_FLAGS=()
[[ "$WRAP_PHASE"   == "1" ]] && FILTER_FLAGS+=(--wrap-phase)
[[ "$INPUT_FILTER" == "1" ]] && FILTER_FLAGS+=(--input-filter)
[[ "$CHARGE_CUT"   == "1" ]] && FILTER_FLAGS+=(--charge-cut)

mkdir -p "$OUTPUT_DIR" logs

# --- environment (same env used by gpu.sh) -----------------------------------
export CONDA_PREFIX=/sdf/group/ad/beamphysics/rroussel/miniforge3/
export PATH=${CONDA_PREFIX}/bin/:$PATH
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda activate gpsr

# --- pick this task's file ----------------------------------------------------
FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FILE_LIST")
if [[ -z "$FILE" ]]; then
    echo "No file for array index $SLURM_ARRAY_TASK_ID"; exit 0
fi
BASE=$(basename "$FILE" .h5)
OUT="$OUTPUT_DIR/${BASE}.csv"

echo "[task $SLURM_ARRAY_TASK_ID] $FILE -> $OUT"
echo "[task $SLURM_ARRAY_TASK_ID] filter flags: ${FILTER_FLAGS[*]:-none}"

python build_dataset_from_standard.py "$FILE" \
    --output "$OUT" \
    --normalize \
    --min-alive-frac 0.9 \
    --max-shots-per-batch "$SHOTS_PER_BATCH" \
    ${FILTER_FLAGS[@]+"${FILTER_FLAGS[@]}"} \
    --progress-every 100 \
    --resume

echo "[task $SLURM_ARRAY_TASK_ID] done."
