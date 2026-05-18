# 571 Covariance Surrogate Model

ML surrogate model predicting the 6×6 beam covariance matrix (+ mean energy/time) from 19 accelerator input parameters at the FACET-II 571 location.

## Pipeline

### 1. Data Preparation

```bash
# Extract covariance targets (Cholesky factors) from particle h5 files
# --normalize applies M-normalization before Cholesky decomposition
python create_cov_targets_from_particles.py particles-571.csv cov-targets.csv \
    --particles-column bmad_final_particles --normalize --drop-failed --progress-every 100

# Extract mean beam values (energy, time)
python extract_beam_means.py

# Combine into single dataset with sim inputs + targets
python create_dataset.py

# Split into train/val/test (70/15/15)
python split_dataset.py
```

**M-Normalization:** The raw 6×6 covariance matrix has elements spanning wildly different scales (e.g., position in meters vs momentum in eV/c). Before taking the Cholesky decomposition, we apply $C_{\text{norm}} = M \cdot C \cdot M^T$ with:

$$M = \text{diag}(10^3,\ 10^{-6},\ 10^3,\ 10^{-6},\ 10^{12},\ 10^{-6})$$

This brings all phase-space dimensions (x, px, y, py, t, pz) to comparable numerical scales, which makes the Cholesky elements better-conditioned for ML training. The inverse transform $C_{\text{phys}} = M^{-1} C_{\text{norm}} M^{-1T}$ recovers physical units at inference time.

**Output:** `dataset-train.csv`, `dataset-val.csv`, `dataset-test.csv`

### 2. Training

```bash
python train.py --cov-loss l1 --epochs 200 --patience 40 --batch-size 256 --lr 1e-3 \
    --finetune-batch-sizes 32 8 2 --finetune-epochs-per-stage 300 \
    --finetune-lr 1e-4 --finetune-lr-decay 0.5 \
    --finetune-plateau-patience 5 --finetune-min-lr 1e-6 \
    --output-dir model-output-571-more-data
```

- **Architecture:** MLP backbone (100→200→200→300→300→200→100→100→100, ELU, Dropout 0.05), dual heads for Cholesky (21 outputs) and mean beam (2 outputs)
- **Loss:** L1 in normalized covariance space (CovarianceAwareLoss)
- **Training:** 200 epochs base + 3 finetuning stages (batch 32→8→2, 300 epochs each) with LR annealing
- **Parameters:** 316k

### 3. Analysis

```bash
# Standard metrics (R², MAE, MAPE per element, scatter plots)
python analyze_covariance.py --model-dir model-output-571-more-data --output-dir analysis-more-data

# Filtered metrics (beam quality thresholds)
python analyze_filtered.py --model-dir model-output-571-more-data --output-dir analysis-more-data-filtered

# 2D phase-space ellipse comparisons against particle ground truth
python compare_2d_distributions_571.py --output-dir compare-2d-571
```

### 4. Inference

```bash
python infer_covariance.py --model-dir model-output-571-more-data --input-csv new_inputs.csv
```

## Diagnostic Experiments

### Overfit Test
Tested if model capacity is sufficient by training with dropout=0 at 1x and 2x width.

```bash
python overfit_test.py --cov-loss l1 --output-dir overfit-test-571
python overfit_test.py --cov-loss l1 --hidden-mult 2.0 --output-dir overfit-test-571-2x
```

**Result:** Both hit the same train loss floor (~0.042). Larger model only increases overfitting gap. → Capacity is sufficient.

### Emittance-Weighted Loss
Weighted loss inversely proportional to beam emittance (good beams get higher weight).

```bash
python train_emittance_weighted.py --cov-loss l1 --output-dir model-output-571-emitw \
    --weight-method emit_avg --weight-temperature 1.0 \
    --finetune-batch-sizes 32 8 2 --finetune-epochs-per-stage 50 --finetune-lr 1e-4
```

**Result:** Modest MAPE improvement (42% → 36%), but R² slightly worse. Spatial cross-terms unchanged. → Loss function not the bottleneck.

### Learning Curve
Trained at 10/25/50/75/100% of data to determine if model is data-limited or capacity-limited.

```bash
python learning_curve.py --cov-loss l1 --output-dir learning-curve-571
```

**Result:** R² and test loss improve linearly with data size (no plateau). → Model is data-limited.

## Key Results (15.4k training samples)

| Element | R² |
|---|---|
| cov_00 (σ²_x) | 0.76 |
| cov_11 (σ²_px) | 0.78 |
| cov_22 (σ²_y) | 0.79 |
| cov_33 (σ²_py) | 0.72 |
| cov_55 (σ²_pz) | 0.91 |
| mean_energy | 0.998 |
| mean_time | 0.88 |
| **Overall mean R²** | **0.51** |

## Conclusions

1. Model is **data-limited** — more training samples (target 50k+) is the primary path to improvement
2. Architecture is adequate (316k params) — larger models don't reduce train loss
3. Emittance-weighted loss gives marginal gains — not the bottleneck
4. Spatial elements (x, px, y, py) improved dramatically with more data (R² ~0 → ~0.75)

## File Descriptions

| File | Purpose |
|---|---|
| `train.py` | Main training script (dual-head covariance + mean beam) |
| `train_emittance_weighted.py` | Training with emittance-weighted loss |
| `analyze_covariance.py` | Post-training analysis (R², MAE, MAPE, scatter plots) |
| `analyze_filtered.py` | Analysis filtered by beam quality thresholds |
| `compare_2d_distributions_571.py` | 2D ellipse comparison vs particle ground truth |
| `learning_curve.py` | Train at data fractions, plot samples vs accuracy |
| `overfit_test.py` | Capacity test (train with no regularization) |
| `create_cov_targets_from_particles.py` | Extract Cholesky targets from h5 particles |
| `create_dataset.py` | Combine inputs + targets into dataset CSV |
| `split_dataset.py` | Train/val/test split |
| `pv_mapping.py` | Sim parameter ↔ machine PV name mapping |
| `infer_covariance.py` | Run inference on new inputs |
