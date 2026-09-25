#!/bin/bash
# Concatenate per-file dataset shards (shards/*.csv) into a single dataset.csv.
# Keeps the header from the first shard only. Skips the .provenance.csv sidecars.
set -euo pipefail

SHARD_DIR=${1:-shards}
OUT=${2:-dataset.csv}

first=1
: > "$OUT"
for f in "$SHARD_DIR"/*.csv; do
    case "$f" in
        *.provenance.csv) continue ;;   # skip provenance sidecars
    esac
    if [[ $first -eq 1 ]]; then
        cat "$f" >> "$OUT"
        first=0
    else
        tail -n +2 "$f" >> "$OUT"        # drop header line
    fi
done

rows=$(( $(wc -l < "$OUT") - 1 ))
echo "[concat] $OUT  rows=$rows  (from $SHARD_DIR)"

# Optional: also merge provenance sidecars
PROV_OUT="${OUT%.csv}.provenance.csv"
first=1
: > "$PROV_OUT"
for f in "$SHARD_DIR"/*.provenance.csv; do
    [[ -e "$f" ]] || continue
    if [[ $first -eq 1 ]]; then cat "$f" >> "$PROV_OUT"; first=0
    else tail -n +2 "$f" >> "$PROV_OUT"; fi
done
[[ -s "$PROV_OUT" ]] && echo "[concat] $PROV_OUT  rows=$(( $(wc -l < "$PROV_OUT") - 1 ))"
