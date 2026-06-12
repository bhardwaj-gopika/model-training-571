# AGENTS.md — FACET-II Injector ML Surrogate (571 Screen)

## What This Codebase Does

This repository builds and evaluates an **ML surrogate model for the FACET-II particle accelerator injector** at SLAC. The model predicts the **6×6 beam covariance matrix** and **6 phase-space means** (x, px, y, py, t, pz) at **screen 571** from 19 accelerator control parameters.

The covariance matrix encodes beam size, divergence, emittance, and correlations in all six phase-space dimensions. Direct physics simulation is expensive; this surrogate replaces it with a fast neural network for use in online optimization and diagnostics.

### Scientific Context

- **Accelerator**: FACET-II electron beam injector at SLAC National Accelerator Laboratory
- **Screen 571**: A diagnostic screen downstream of the injector where beam properties are measured
- **Phase space**: The 6D space (x, px, y, py, t, pz) describing particle positions and momenta
- **Covariance prediction**: Instead of predicting the full 6×6 symmetric matrix (21 unique elements) directly, the model predicts the **lower-triangular Cholesky factor L** such that Σ = L Lᵀ. This guarantees positive semi-definiteness.
- **M-normalization**: A diagonal scaling matrix M = diag(1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6) is applied before Cholesky decomposition to bring all covariance elements to comparable magnitude. Denormalization: Σ_phys = M⁻¹ Σ_norm M⁻¹ᵀ.
- **Alive particle filtering**: Simulations start with 100,000 particles. Samples with <90,000 alive particles are filtered out during target extraction (`--min-alive-particles 90000`). These low-survival samples produce extreme covariance outliers that distort Z-score standardization and degrade training. This removes ~14% of samples (~9,400) and improves covariance MAPE by 20–38%.
- **Beam-quality filtering**: After alive-particle filtering, samples outside physically reasonable beam thresholds are removed (σ_x,σ_y < 5 mm, relative energy spread < 5×10⁻³, normalized emittance ε_x,ε_y < 20 μm). This focuses the model on the operating regime of interest.
- **No dropout**: The final model is trained with dropout disabled (`--dropout 0.0`). With ~53k training samples and beam-quality filtering removing outliers, regularization via dropout is unnecessary.

## Pipeline Steps

### 1. Target Extraction (`create_targets_from_particles.py`)

Reads `.h5` particle files listed in `particles-571.csv` and extracts:
- 21 Cholesky-flattened covariance targets (M-normalized)
- 6 phase-space means (mean_x, mean_px, mean_y, mean_py, mean_t, mean_pz)

Filters out samples with too few alive particles to remove outliers.

```bash
python create_targets_from_particles.py particles-571.csv targets-571.csv \
  --normalize --drop-failed --progress-every 1000 \
  --min-alive-particles 90000
```

### 2. Dataset Assembly (`create_dataset.py`)

Combines 19 simulator input columns with targets into a single CSV. Drops rows with null values.

```bash
python create_dataset.py \
  --inputs particles-571.csv \
  --targets targets-571.csv \
  --output dataset.csv
```

Output columns: 19 inputs + 6 means + 21 Cholesky elements = 46 columns.

### 3. Beam-Quality Filtering (`filter_dataset.py`)

Filters the dataset to keep only samples within physically reasonable beam thresholds.

```bash
python filter_dataset.py --input dataset.csv --output dataset-filtered.csv
```

Default thresholds: σ_x < 5 mm, σ_y < 5 mm, relative energy spread < 5×10⁻³, ε_x < 20 μm, ε_y < 20 μm.

### 4. Train/Val/Test Split (`split_dataset.py`)

Splits `dataset-filtered.csv` into 70/15/15 train/val/test splits.

```bash
python split_dataset.py --input dataset-filtered.csv --seed 42
```

Outputs: `dataset-filtered-train.csv`, `dataset-filtered-val.csv`, `dataset-filtered-test.csv`.

### 5. Training (`train2.py`)

Trains a `CovarianceSurrogateModel` MLP with dual output heads:
- **Cholesky head**: 21 outputs → reconstructed 6×6 covariance via L @ Lᵀ
- **Mean head**: 6 outputs → phase-space means

Architecture: 100 → 200 → 200 → 300 → 300 → 200 → 100 → 100 → 100 (ELU activation, **no dropout**, ~316k parameters).

Training strategy: 200 base epochs + 3 finetuning stages (batch 32→8→2, 300 epochs each, LR cosine annealing).

Key improvements over `train.py`: configurable dropout (set to 0), configurable activation function, gradient clipping support, saves `model_config.json`.

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

Saves to `model-output-571-filtered2/`: `model.pt`, `input_transformers.pt`, `output_transformers.pt`, `covariance_transformers.pt`, `mean_transformers.pt`, `training_history.csv`, `model_config.json`.

### 6. Analysis (`analyze_covariance2.py`)

Evaluates trained model on test set. Computes R², MAE, MAPE, RMSE per covariance element and mean target.

```bash
python analyze_covariance2.py \
  --model-dir model-output-571-filtered2 \
  --test dataset-filtered-test.csv \
  --output-dir analysis-filtered2
```

Outputs: `test_metrics.csv`, `mean_beam_metrics.csv`, scatter/bar plots (PNG).

### 7. Filtered Analysis (`analyze_filtered2.py`)

Re-evaluates on test samples passing beam-quality cuts.

```bash
python analyze_filtered2.py \
  --model-dir model-output-571-filtered2 \
  --test dataset-filtered-test.csv \
  --output-dir analysis-filtered2-filtered
```

### 8. Inference & LUME Export (`infer_covariance2.py`)

Runs inference and exports LUME-Torch YAML model files for deployment. Validates both simulator-parameter and machine-PV input spaces. Creates cov-only and full (cov + mean) model variants.

```bash
python infer_covariance2.py \
  --model-dir model-output-571-filtered2 \
  --input-csv dataset-filtered-test.csv
```

Outputs:
- `inference-output/predicted_covariances.csv` — flat predictions
- `lumetorchyaml-sim/`, `lumetorchyaml-machine/` — cov-only LUME models
- `lumetorchyaml-sim-full/`, `lumetorchyaml-machine-full/` — cov + mean LUME models

Machine-PV mapping is defined in `pv_mapping.py` (affine transform: 19 parameters with per-channel scale and offset).

### 9. Package Model Update

Copy LUME model files into the deployable `facet2-model-571` package:

```bash
cp -r lumetorchyaml-sim/ ../facet2-model-571/facet2_inj_ml_model_571/resources/lumetorchyaml-sim/
cp -r lumetorchyaml-sim-full/ ../facet2-model-571/facet2_inj_ml_model_571/resources/lumetorchyaml-sim-full/
cp -r lumetorchyaml-machine/ ../facet2-model-571/facet2_inj_ml_model_571/resources/lumetorchyaml-machine/
cp -r lumetorchyaml-machine-full/ ../facet2-model-571/facet2_inj_ml_model_571/resources/lumetorchyaml-machine-full/
cd ../facet2-model-571 && pip install -e . && cd -
```

### 10. Overlap Plots (`plot_beam_overlap.py`)

Generates KDE contour overlap plots comparing true particle distributions against samples from predicted covariance. Displays true and predicted 2×2 covariance sub-matrices on each panel. Uses `--min-alive-particles` to match the training filter.

```bash
python plot_beam_overlap.py \
  --particles-csv particles-571.csv \
  --input-space sim \
  --num-samples 5 \
  --full \
  --min-alive-particles 90000 \
  --output-dir overlap-plots
```

### 11. End-to-End Demo (`end_to_end_demo.py`)

Demonstrates the full inference pipeline: loading the packaged model via `facet2_inj_ml_model_571`, calling `evaluate()`, wrapping in `BeamOutputModel` to produce a sampled particle distribution, and overlaying with true OpenPMD particles.

```bash
python end_to_end_demo.py --particles-csv particles-571.csv --num-samples 3
```

## Key Outputs & Transformations

| Artifact | Description |
|----------|-------------|
| `model.pt` | Trained PyTorch model weights (CovarianceSurrogateModel) |
| `model_config.json` | Model architecture config (dropout, activation) |
| `input_transformers.pt` | Input standardization (x_mean, x_std, feature_cols) |
| `output_transformers.pt` | Combined output standardization (y_mean, y_std) |
| `covariance_transformers.pt` | Cholesky target standardization (cov_mean, cov_std, cov_labels) |
| `mean_transformers.pt` | Mean target standardization (mean_y_mean, mean_y_std, mean_cols) |
| `training_history.csv` | Per-epoch train_loss and val_loss |
| `test_metrics.csv` | R², MAE, MAPE, RMSE per Cholesky element on test set |
| `mean_beam_metrics.csv` | R², MAE, MAPE for phase-space mean predictions |
| LUME YAML files | lume-torch model descriptors for deployment (sim & machine input spaces) |

**Transform chain (inference)**:
1. Raw inputs → standardize (subtract mean, divide by std)
2. Model forward pass → standardized Cholesky vector + standardized means
3. Destandardize outputs
4. Cholesky vector → L matrix → Σ_norm = L Lᵀ (M-normalized covariance)
5. Σ_phys = M⁻¹ Σ_norm M⁻¹ᵀ (physical-unit covariance)

## Directory Structure

```
modeling-571-t/
├── particles-571.csv           # Particle file paths + sim input columns (~65k rows)
├── dataset.csv                 # Combined inputs + targets (46 columns)
├── dataset-filtered.csv        # Beam-quality-filtered dataset
├── dataset-filtered-train.csv  # Filtered training split (70%)
├── dataset-filtered-val.csv    # Filtered validation split (15%)
├── dataset-filtered-test.csv   # Filtered test split (15%)
│
├── create_targets_from_particles.py   # Step 1: extract targets from .h5 files (--min-alive-particles)
├── create_dataset.py                  # Step 2: merge inputs + targets
├── filter_dataset.py                  # Step 3: beam-quality filtering
├── split_dataset.py                   # Step 4: train/val/test split
├── train2.py                          # Step 5: model training (no dropout, configurable)
├── analyze_covariance2.py             # Step 6: test-set evaluation
├── analyze_filtered2.py               # Step 7: filtered evaluation
├── infer_covariance2.py               # Step 8: inference & LUME export
├── plot_beam_overlap.py               # Step 10: KDE overlap plots
├── end_to_end_demo.py                 # Step 11: full pipeline demo
├── BeamOutputModel.py                 # Wraps cov+mean predictions into particle distribution
│
├── pv_mapping.py               # Machine PV ↔ sim parameter affine mapping
├── lume_model_utils.py          # Custom lume-torch transforms (M-denorm, CovMeanTorchModel)
│
├── gpu.sh                      # SLURM job script for training (80GB RAM, 1 GPU, 10h)
├── gpu2.sh                     # SLURM job script for target extraction
│
├── model-output-571-filtered2/ # Trained model checkpoint & transformers (filtered, no dropout)
│   ├── model.pt
│   ├── model_config.json
│   ├── input_transformers.pt
│   ├── output_transformers.pt
│   ├── covariance_transformers.pt
│   ├── mean_transformers.pt
│   └── training_history.csv
│
├── lumetorchyaml-sim/          # LUME-Torch model (sim-parameter inputs, cov-only)
├── lumetorchyaml-sim-full/     # LUME-Torch model (sim inputs, cov + means)
├── lumetorchyaml-machine/      # LUME-Torch model (machine PV inputs, cov-only)
├── lumetorchyaml-machine-full/ # LUME-Torch model (machine PVs, cov + means)
│
└── inference-output/           # Inference result CSVs
```

## Dependencies

Inferred from imports (no `requirements.txt` in this directory):

- `torch`, `torchvision`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `lume-torch` (SLAC beamphysics org)
- `openPMD-beamphysics` (`ParticleGroup` class)
- `botorch` (`AffineInputTransform`)
- `facet2-inj-ml-model-571` (deployable model package)

The deployable model package is in `facet2-model-571/` with its own `pyproject.toml` (requires `lume-torch`, `numpy`, `pandas`, `torch`, Python ≥ 3.11).

## Running Tests

There are no unit tests in this directory. Validation is done via the analysis scripts:

```bash
# Full test-set evaluation
python analyze_covariance2.py --model-dir model-output-571-filtered2 --test dataset-filtered-test.csv --output-dir analysis-filtered2

# Filtered evaluation (beam-quality cuts)
python analyze_filtered2.py --model-dir model-output-571-filtered2 --test dataset-filtered-test.csv --output-dir analysis-filtered2-filtered
```

The packaged model (`facet2-model-571/`) has tests:
```bash
cd ../facet2-model-571
pip install -e ".[dev]"
pytest
```

## Conventions

- **Cholesky ordering**: 21 elements from flattening the lower triangle of a 6×6 matrix, row-major: cov_chol_0 through cov_chol_20.
- **M-normalization**: Always applied before Cholesky decomposition. M_DIAG = [1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6].
- **Input/output standardization**: Z-score normalization (subtract mean, divide by std) computed from training set and saved in transformer `.pt` files.
- **Loss function**: `CovarianceAwareLoss` — computed in M-normalized covariance space (reconstructs Σ from Cholesky, compares element-wise).
- **Column naming**: Input columns use simulator parameter names (e.g., `CQ10121:b1_gradient`). Machine PV names (e.g., `QUAD:IN10:121:BCTRL`) are mapped via `pv_mapping.py`.
