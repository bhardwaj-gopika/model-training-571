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

### 3. Beam-Quality Filtering

Filters the dataset to keep only samples within physically reasonable beam thresholds. This removes extreme outliers that the model doesn't need to learn.

```bash
python filter_dataset.py --input dataset.csv --output dataset-filtered.csv
```

Default thresholds:
| Parameter | Threshold |
|---|---|
| σ_x | < 5 mm |
| σ_y | < 5 mm |
| Relative energy spread | < 5×10⁻³ |
| Normalized emittance ε_x | < 20 μm |
| Normalized emittance ε_y | < 20 μm |

### 4. Train/Val/Test Split

```bash
python split_dataset.py --input dataset-filtered.csv --seed 42
```

Output: `dataset-filtered-train.csv`, `dataset-filtered-val.csv`, `dataset-filtered-test.csv` (70/15/15 split).

### 5. Training

```bash
python train2.py \
    --cov-loss l1 --epochs 200 --patience 40 --batch-size 256 --lr 1e-3 \
    --dropout 0.0 \
    --finetune-batch-sizes 32 8 2 --finetune-epochs-per-stage 300 \
    --finetune-lr 1e-4 --finetune-lr-decay 0.5 \
    --finetune-plateau-patience 5 --finetune-min-lr 1e-6 \
    --train-csv dataset-filtered-train.csv --val-csv dataset-filtered-val.csv \
    --output-dir model-output-571-filtered2
```

On SLURM (GPU):
```bash
sbatch gpu.sh
```

- **Architecture:** MLP backbone (100→200→200→300→300→200→100→100→100, ELU, **no dropout**), dual heads for Cholesky (21 outputs) and mean beam (6 outputs: mean_x, mean_px, mean_y, mean_py, mean_t, mean_pz)
- **Loss:** L1 in normalized covariance space (CovarianceAwareLoss)
- **Training:** 200 epochs base + 3 finetuning stages (batch 32→8→2, 300 epochs each) with LR annealing
- **Parameters:** ~316k
- **Key difference from `train.py`:** configurable dropout (set to 0), configurable activation function, gradient clipping support, saves `model_config.json`

### 6. Analysis

```bash
# Standard metrics (R², MAE, MAPE per element, scatter plots)
python analyze_covariance2.py --model-dir model-output-571-filtered2 --output-dir analysis-filtered2

# Filtered metrics (beam quality thresholds)
python analyze_filtered2.py --model-dir model-output-571-filtered2 --output-dir analysis-filtered2-filtered
```

### 7. Inference & LUME Export

Runs inference on test set, validates LUME-Torch models (sim + machine input spaces), and exports deployable YAML model files.

```bash
python infer_covariance2.py --model-dir model-output-571-filtered2 --input-csv dataset-filtered-test.csv
```

This generates:
- `inference-output/predicted_covariances.csv` — flat predictions
- `lumetorchyaml-sim/`, `lumetorchyaml-machine/` — cov-only LUME models
- `lumetorchyaml-sim-full/`, `lumetorchyaml-machine-full/` — cov + mean LUME models

### 8. Package Model & Overlap Plots

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
    --output-dir overlap-plots
```

### 9. End-to-End Demo

Demonstrates the full inference pipeline: loading the packaged model, calling `evaluate()`, wrapping in `BeamOutputModel` to produce a particle distribution, and overlaying with true OpenPMD particles.

```bash
python end_to_end_demo.py --particles-csv particles-571.csv --num-samples 3
```

## Technical Details

**M-Normalization:** The raw 6×6 covariance matrix has elements spanning wildly different scales (e.g., position in meters vs momentum in eV/c). Before taking the Cholesky decomposition, we apply $C_{\text{norm}} = M \cdot C \cdot M^T$ with:

$$M = \text{diag}(10^3,\ 10^{-6},\ 10^3,\ 10^{-6},\ 10^{12},\ 10^{-6})$$

This brings all phase-space dimensions (x, px, y, py, t, pz) to comparable numerical scales, which makes the Cholesky elements better-conditioned for ML training. The inverse transform $C_{\text{phys}} = M^{-1} C_{\text{norm}} M^{-1T}$ recovers physical units at inference time.

**Alive Particle Filtering:** Simulations start with 100,000 particles. Some configurations cause massive particle loss (>10% dead), producing extreme covariance outliers that distort Z-score standardization and degrade training. The `--min-alive-particles 90000` threshold removes these (~9,400 samples, ~14%), improving covariance MAPE by 20–38% on x-px elements.

**Beam-Quality Filtering:** After alive-particle filtering, an additional beam-quality filter removes samples with extreme beam sizes, energy spreads, or emittances that fall outside the operating regime of interest. This focuses the model on the physically relevant region of parameter space.

**No Dropout:** The final model is trained with dropout disabled (`--dropout 0.0`). With sufficient data (~53k samples after filtering) and beam-quality filtering removing outliers, regularization via dropout is unnecessary and removing it improves prediction accuracy.

## File Descriptions

| File | Purpose |
|---|---|
| `create_targets_from_particles.py` | Step 1: Extract Cholesky targets + 6 phase-space means from `.h5` particles, with `--min-alive-particles` filtering |
| `create_dataset.py` | Step 2: Combine inputs + targets into dataset CSV |
| `filter_dataset.py` | Step 3: Filter dataset by beam-quality thresholds (σ, emittance, energy spread) |
| `split_dataset.py` | Step 4: Train/val/test split |
| `train2.py` | Step 5: Model training (dual-head covariance + mean beam, configurable dropout/activation) |
| `analyze_covariance2.py` | Step 6: Post-training analysis (R², MAE, MAPE, scatter plots) |
| `analyze_filtered2.py` | Step 6b: Analysis filtered by beam quality thresholds |
| `infer_covariance2.py` | Step 7: Inference + LUME-Torch model export & validation |
| `plot_beam_overlap.py` | Step 8: KDE contour overlap plots (true particles vs predicted covariance) |
| `end_to_end_demo.py` | Step 9: Full pipeline demo with packaged model + BeamOutputModel |
| `BeamOutputModel.py` | Wraps covariance + mean predictions into a sampled particle distribution |
| `pv_mapping.py` | Sim parameter ↔ machine PV name mapping |
| `lume_model_utils.py` | Custom lume-torch transforms (M-denorm, CovMeanTorchModel) |
| `gpu.sh` | SLURM job script for training (80GB RAM, 1 GPU, 10h) |
| `gpu2.sh` | SLURM job script for target extraction |
