# CNN 2D Projection Sanity Check — Plan

**Date**: 2026-08-21

## Motivation

The current covariance-prediction MLP (18 scalar inputs → 21 Cholesky factors + 6 means) is giving poor results despite:
- M-normalization + z-score standardization
- Dropout tuning (0.05, 0.1, 0.0)
- Expanded HAAI data (~300k+ samples)
- Input-range filtering, phase wrapping, alive-particle filtering

The concern is whether the problem is:
- **(A) The data** — noisy, badly filtered, or weak input-output relationship
- **(B) The covariance representation** — something off with how we compute/use the cov matrix elements

## Diagnostic Approach

Train a plain MLP to directly predict binned 2D phase-space projections (x-px, y-py, t-pz as 64x64 histograms) from the same 18 inputs.

- **If CNN succeeds** → data is fine, problem is the covariance representation → proceed to normalizing flow
- **If CNN also fails** → problem is upstream in the data, need to investigate further

## Steps

### Step 1: Build Projection Dataset (`build_projection_dataset.py`)

Read the same HAAI `*_571.h5` files with the same filtering. Instead of computing covariance, bin alive particles into three 64x64 2D histograms (x-px, y-py, t-pz), each normalized to sum=1.

Two passes:
1. Compute global bin edges (1st–99th percentile + 10% padding) from training shots
2. Bin all shots using those fixed edges

Output: HDF5 with `/inputs` (N,18), `/projections` (N,3,64,64), `/bin_edges`, provenance keys.

### Step 2: Split Data

Align train/val/test splits with the existing covariance model's split using provenance keys (uuid, shot) so the comparison is apples-to-apples.

### Step 3: Train MLP (`train_cnn.py`)

Architecture (plain FC, no convolutions):
```
18 inputs (z-scored)
→ Linear(18, 256) + ELU
→ Linear(256, 512) + ELU
→ Linear(512, 1024) + ELU
→ Linear(1024, 2048) + ELU
→ 3 heads: Linear(2048, 4096) + Softplus → reshape 64x64 → normalize to sum=1
```

Loss: MSE on normalized histograms (+ optional SSIM component).
Training: Adam lr=1e-3, batch_size=128, patience=30, up to 200 epochs.

### Step 4: Evaluate (`analyze_cnn.py`)

Key deliverables:
1. **Overlay plots**: True beam density vs predicted density (same style as current `plot_beam_overlap.py`)
2. **SSIM per projection**: Quantitative measure of prediction quality
3. **Covariance recovery**: Compute 2x2 covariance from predicted histograms, compare to true — directly comparable to the current MLP's covariance predictions

### Step 5: Interpret Results

| CNN Result | Interpretation | Next Step |
|------------|---------------|-----------|
| Good overlays, high SSIM | Data is fine; covariance representation is the problem | Try normalizing flow |
| Poor overlays, low SSIM | Input-output relationship too weak or data issue | Investigate data quality |

## Files to Create

- `build_projection_dataset.py` — dataset builder
- `train_cnn.py` — training script
- `analyze_cnn.py` — evaluation and overlay plots
- `gpu_cnn.sh` — SLURM job script

## Dependencies

- `pytorch-msssim` (for SSIM metric, optional)
- All other deps already available (torch, numpy, h5py, matplotlib, scipy)

## Notes

- No modifications to existing covariance pipeline
- Reuses filtering/iteration logic from `build_dataset_from_standard.py`
- Same test set for fair comparison
- If this works, next step is a conditional normalizing flow (full 6D, using `zuko` library)
