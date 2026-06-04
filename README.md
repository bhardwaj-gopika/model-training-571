# 571 Covariance Surrogate Model

ML surrogate model predicting the full 6×6 beam covariance matrix and all 6 phase-space mean values (x, px, y, py, t, pz) from 19 accelerator input parameters at the FACET-II 571 location.

## Pipeline

### 1. Target Extraction

Extract Cholesky-flattened covariance targets and 6 phase-space means from particle `.h5` files. The `--min-alive-particles` flag filters out samples where too many particles were lost in simulation (outliers that distort training).

```bash
python create_targets_from_particles.py particles-571.csv targets-571.csv \
    --particles-column bmad_final_particles \
    --normalize --drop-failed --progress-every 1000 \
    --min-alive-particles 90000
```

- `--normalize`: applies M-normalization before Cholesky decomposition
- `--min-alive-particles 90000`: skips samples with <90k alive particles (out of 100k total), removing ~14% outlier samples where massive particle loss produces extreme covariance values

### 2. Dataset Assembly

Combines 19 simulator input columns with targets into a single CSV. Drops rows with null values.

```bash
python create_dataset.py --inputs particles-571.csv --targets targets-571.csv --output dataset.csv
```

Output columns: 19 inputs + 6 means + 21 Cholesky elements = 46 columns.

### 3. Train/Val/Test Split

```bash
python split_dataset.py --input dataset.csv --seed 42
```

Output: `dataset-train.csv`, `dataset-val.csv`, `dataset-test.csv` (70/15/15 split).

### 4. Training

```bash
python train.py --cov-loss l1 --epochs 200 --patience 40 --batch-size 256 --lr 1e-3 \
    --finetune-batch-sizes 32 8 2 --finetune-epochs-per-stage 300 \
    --finetune-lr 1e-4 --finetune-lr-decay 0.5 \
    --finetune-plateau-patience 5 --finetune-min-lr 1e-6 \
    --output-dir model-output-571-alive
```

On SLURM (GPU):
```bash
sbatch gpu.sh
```

- **Architecture:** MLP backbone (100→200→200→300→300→200→100→100→100, ELU, Dropout 0.05), dual heads for Cholesky (21 outputs) and mean beam (6 outputs: mean_x, mean_px, mean_y, mean_py, mean_t, mean_pz)
- **Loss:** L1 in normalized covariance space (CovarianceAwareLoss)
- **Training:** 200 epochs base + 3 finetuning stages (batch 32→8→2, 300 epochs each) with LR annealing
- **Parameters:** 316k

### 5. Analysis

```bash
# Standard metrics (R², MAE, MAPE per element, scatter plots)
python analyze_covariance.py --model-dir model-output-571-alive --output-dir analysis-alive

# Filtered metrics (beam quality thresholds)
python analyze_filtered.py --model-dir model-output-571-alive --output-dir analysis-filtered-alive
```

### 6. Inference & LUME Export

Runs inference on test set, validates LUME-Torch models (sim + machine input spaces), and exports deployable YAML model files.

```bash
python infer_covariance.py --model-dir model-output-571-alive --input-csv dataset-test.csv
```

This generates:
- `inference-output/predicted_covariances.csv` — flat predictions
- `lumetorchyaml-sim/`, `lumetorchyaml-machine/` — cov-only LUME models
- `lumetorchyaml-sim-full/`, `lumetorchyaml-machine-full/` — cov + mean LUME models

### 7. Package Model & Overlap Plots

Copy LUME model files into the deployable package and regenerate overlap plots:

```bash
# Copy model files into package
cp -r lumetorchyaml-sim/ ../facet2-model-571/facet2_inj_ml_model_571/resources/lumetorchyaml-sim/
cp -r lumetorchyaml-sim-full/ ../facet2-model-571/facet2_inj_ml_model_571/resources/lumetorchyaml-sim-full/
cp -r lumetorchyaml-machine/ ../facet2-model-571/facet2_inj_ml_model_571/resources/lumetorchyaml-machine/
cp -r lumetorchyaml-machine-full/ ../facet2-model-571/facet2_inj_ml_model_571/resources/lumetorchyaml-machine-full/

# Reinstall package
cd ../facet2-model-571 && pip install -e . && cd -

# Generate overlap plots (KDE contours with true/predicted 2x2 covariance annotations)
python plot_beam_overlap.py \
    --particles-csv particles-571.csv \
    --input-space sim \
    --num-samples 5 \
    --full \
    --min-alive-particles 90000 \
    --output-dir overlap-plots-alive
```

**M-Normalization:** The raw 6×6 covariance matrix has elements spanning wildly different scales (e.g., position in meters vs momentum in eV/c). Before taking the Cholesky decomposition, we apply $C_{\text{norm}} = M \cdot C \cdot M^T$ with:

$$M = \text{diag}(10^3,\ 10^{-6},\ 10^3,\ 10^{-6},\ 10^{12},\ 10^{-6})$$

This brings all phase-space dimensions (x, px, y, py, t, pz) to comparable numerical scales, which makes the Cholesky elements better-conditioned for ML training. The inverse transform $C_{\text{phys}} = M^{-1} C_{\text{norm}} M^{-1T}$ recovers physical units at inference time.

**Alive Particle Filtering:** Simulations start with 100,000 particles. Some configurations cause massive particle loss (>10% dead), producing extreme covariance outliers that distort Z-score standardization and degrade training. The `--min-alive-particles 90000` threshold removes these (~9,400 samples, ~14%), improving covariance MAPE by 20–38% on x-px elements.

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
| mean_x | — |
| mean_px | — |
| mean_y | — |
| mean_py | — |
| mean_t | — |
| mean_pz | — |
| **Overall mean R²** | **0.51** |

## Conclusions

1. Model is **data-limited** — more training samples (target 50k+) is the primary path to improvement
2. Architecture is adequate (316k params) — larger models don't reduce train loss
3. Emittance-weighted loss gives marginal gains — not the bottleneck
4. Spatial elements (x, px, y, py) improved dramatically with more data (R² ~0 → ~0.75)

## File Descriptions

| File | Purpose |
|---|---|
| `create_targets_from_particles.py` | Step 1: Extract Cholesky targets + 6 phase-space means from `.h5` particles, with `--min-alive-particles` filtering |
| `create_dataset.py` | Step 2: Combine inputs + targets into dataset CSV |
| `split_dataset.py` | Step 3: Train/val/test split |
| `train.py` | Step 4: Model training (dual-head covariance + mean beam) |
| `analyze_covariance.py` | Step 5: Post-training analysis (R², MAE, MAPE, scatter plots) |
| `analyze_filtered.py` | Step 5b: Analysis filtered by beam quality thresholds |
| `infer_covariance.py` | Step 6: Inference + LUME-Torch model export & validation |
| `plot_beam_overlap.py` | Step 7: KDE contour overlap plots (true particles vs predicted covariance), with `--min-alive-particles` filtering and 2×2 covariance annotations |
| `pv_mapping.py` | Sim parameter ↔ machine PV name mapping |
| `lume_model_utils.py` | Custom lume-torch transforms (M-denorm, CovMeanTorchModel) |
| `compare_data_distributions.py` | Compare old vs new dataset distributions |
| `make_summary.py` | Summary comparison figure |
| `gpu.sh` | SLURM job script for training (80GB RAM, 1 GPU, 10h) |
| `gpu2.sh` | SLURM job script for target extraction |
