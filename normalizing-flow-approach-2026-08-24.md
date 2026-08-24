# Conditional Normalizing Flow for Beam Distribution Modeling

**Date**: 2026-08-24

## What We're Doing

We're training a conditional normalizing flow to model the full 6D phase-space distribution of the beam at screen PR10571, conditioned on 18 machine settings. Instead of predicting summary statistics (covariance matrix), the flow learns the actual probability density p(x, px, y, py, t, pz | machine settings) and can generate particle samples from it.

## Why: 1-to-1 Comparison with the Covariance Model

The current approach predicts a 6x6 covariance matrix + 6 means — this implicitly assumes the beam distribution is Gaussian. If the real beam has non-Gaussian features (asymmetric tails, correlations beyond second-order, multi-modal structure), the covariance representation cannot capture them by construction.

The comparison tells us:
- **If the flow's sampled covariance matches ground truth better than the MLP's predicted covariance** → the beam is non-Gaussian, and the covariance model fails because it's the wrong representation, not because the data is bad
- **If both perform similarly** → the beam is approximately Gaussian, and the covariance model's problems are elsewhere (data, training, normalization)
- **If the flow also fails** → the 18 inputs may not contain enough information to determine the beam distribution at this screen

The flow is the natural successor because it subsumes the covariance model — you can always compute the covariance from flow samples, but you can't recover the full distribution from a covariance matrix.

## What is a Normalizing Flow?

A normalizing flow is a generative model that transforms a simple base distribution (standard normal) into a complex target distribution through a sequence of invertible, differentiable transformations. Because the transformations are invertible, we can compute exact log-likelihoods — no adversarial training or reconstruction losses needed.

**Training objective**: maximize the log-likelihood of observed particles under the model. This is principled and stable compared to GANs or VAEs.

**Conditional flow**: the transformations depend on a context vector (our 18 machine settings). Given new settings, the flow transforms a standard normal into the predicted beam distribution for those settings.

## Why Neural Spline Flow (NSF)?

We chose NSF (Neural Spline Flow with rational-quadratic splines) over alternatives:

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **Affine coupling (RealNVP)** | Fast, simple | Limited expressiveness; struggles with multi-modal or heavy-tailed distributions |
| **MAF (Masked Autoregressive)** | Very expressive | Slow sampling (sequential); fast density eval |
| **NSF (Neural Spline)** | Expressive + fast sampling + fast density | Slightly more parameters per layer |

NSF uses monotone rational-quadratic splines as the element-wise transformations. This gives:
- **Flexible tail behavior**: splines can model heavy tails and sharp peaks that affine transforms cannot
- **Fast in both directions**: unlike MAF, sampling is parallel (important for generating 10,000+ particle beams at inference)
- **Stable training**: splines are bounded and smooth, reducing NaN/instability issues common with other flow architectures

For particle accelerator beams, which can have asymmetric halos, energy-time correlations, and non-elliptical phase-space shapes, NSF's flexibility is important.

## Architecture

```
Conditional NSF:
  - Input (context): 18 machine settings (z-scored)
  - Output (features): 6D particle coordinates (x, px, y, py, t, pz)
  - Transforms: 8 spline coupling layers
  - Each layer: MLP conditioner [128, 128] -> spline parameters (8 bins)
  - Base distribution: 6D standard normal
```

The flow is trained on 2048 subsampled particles per shot. Each training step:
1. Take a batch of 64 shots (64 × 2048 = 131k particles)
2. Compute log p(particle | machine settings) for each particle
3. Minimize negative log-likelihood

## How to Use at Inference

```python
# Given new machine settings:
context = normalize(machine_settings)  # (1, 18)

# Sample a full beam:
samples = flow(context).sample((10000,))  # (10000, 6) particles

# Compute any statistic you want:
covariance = np.cov(samples.T)           # 6x6 matrix
emittance = compute_emittance(samples)   # derived quantity
projections = np.histogram2d(...)        # 2D images
```

This is strictly more powerful than the covariance model: you get the full distribution, from which you can derive covariance, means, emittance, projections, or any other statistic.

## Evaluation Plan

For an apples-to-apples comparison with the covariance model:
1. Sample 10,000 particles from the flow for each test shot
2. Compute the 6x6 covariance from the samples
3. Compare to ground truth using the same metrics (MAE, MAPE, R² per element)
4. Also compare 2D projection overlays (flow samples vs true particles)
5. Report mean beam values (mean_x, mean_px, etc.) from flow samples vs true

## Potential Advantages for FACET-II

- **No Gaussian assumption**: captures real beam structure (halos, asymmetries, correlations)
- **Direct particle generation**: output can plug directly into downstream simulations (no need to sample from a Gaussian approximation via `BeamOutputModel.py`)
- **Richer loss signal**: NLL uses every particle as a training signal, not just 27 summary numbers (21 Cholesky + 6 means)
- **Natural uncertainty**: sample multiple beams to see how much the predicted distribution varies

## Pipeline: Step-by-Step Code Walkthrough

### Step 1: Build Particle Dataset (`build_particle_dataset.py`)

```bash
python build_particle_dataset.py \
    /sdf/data/ad/ard-online/FACET-II_Training_Data/2026-07-23/*_571.h5 \
    --output particles_flow.h5 \
    --n-particles 2048 \
    --min-alive-frac 0.9 --wrap-phase --input-filter \
    --max-shots-per-batch 10 --progress-every 200
```

**What it does**: Reads the same HAAI-standard HDF5 files used by the covariance pipeline. For each shot that passes the filters:

1. Opens the ParticleGroup at `observables/PR10571/571_particles_<i>/electron/`
2. Filters to alive particles (status == 1)
3. Randomly subsamples 2048 particles from the alive set
4. Stores their raw 6D coordinates: (x, px, y, py, t, pz)
5. Also stores the 18 machine-setting inputs for that shot

**Filtering logic** (identical to covariance pipeline):
- `--min-alive-frac 0.9`: drop shots where <90% of particles survived to the screen
- `--input-filter`: reject shots with any of the 18 inputs outside the supervisor's defined ranges (gun phase, solenoid strength, quad strengths, etc.)
- `--wrap-phase`: wrap GUNF/L0AF/L0BF phase angles into (-180°, 180°] to fix the bimodal artifact

**Output**: HDF5 file with:
- `/inputs` — (N_shots, 18) machine settings per shot
- `/particles` — (N_shots, 2048, 6) particle coordinates per shot
- `/provenance_uuid`, `/provenance_shot` — traceability back to source

**Why 2048 particles**: Balance between capturing the distribution shape (need enough points for tails and structure) and memory/speed (the flow evaluates log-prob for every particle in every batch). Shots have 20k+ alive particles, so 2048 is a ~10% subsample that still gives good covariance estimates.

---

### Step 2: Split into Train/Val/Test (`split_particle_dataset.py`)

```bash
python split_particle_dataset.py --input particles_flow.h5
```

**What it does**: Splits the monolithic HDF5 into three files using a random 70/15/15 permutation (seed=42, same as covariance pipeline). If provenance CSVs from the covariance pipeline are available, it aligns the splits by (uuid, shot) keys so the exact same shots are in each split — enabling direct comparison.

**Output**: `particles_train.h5`, `particles_val.h5`, `particles_test.h5`

---

### Step 3: Train the Flow (`train_flow.py`)

```bash
python train_flow.py \
    --train-h5 particles_train.h5 \
    --val-h5 particles_val.h5 \
    --test-h5 particles_test.h5 \
    --output-dir model-output-flow \
    --epochs 100 --batch-size 64 --lr 5e-4 \
    --n-transforms 8 --hidden-features 128 128 --bins 8 \
    --patience 20
```

**Data preparation before training**:
1. Computes z-score normalization for inputs (mean/std from training set inputs)
2. Computes z-score normalization for particles (mean/std across all training particles)
3. Both normalizations are saved as `input_transformers.pt` and `particle_transformers.pt` for use at inference

**Training logic — what happens each epoch**:

For each mini-batch of 64 shots:
1. Load 64 shots: each has 18 inputs + 2048 particles of shape (6,)
2. Normalize inputs: `inputs_norm = (inputs - x_mean) / x_std`
3. Normalize particles: `particles_norm = (particles - p_mean) / p_std`
4. Expand the 18-dim context vector so each of the 2048 particles in a shot shares the same context → shape becomes (64 × 2048, 18) = (131,072 context vectors)
5. Flatten particles to (131,072, 6)
6. Pass context through the flow to get a conditional distribution object
7. Evaluate `log_prob(particle | context)` for each particle — this is the flow computing: "given these machine settings, how likely is this particle at this position in phase space?"
8. Loss = negative mean log-probability (we want to maximize likelihood)
9. Backpropagate through the flow's spline parameters and conditioner MLPs
10. Gradient clipping (max norm 1.0) for stability

**What the flow learns**: The spline parameters in each coupling layer are produced by a small MLP that takes the 18 machine settings as input. So the flow is learning: "when the gun phase is -60° and the solenoid is 0.25 T and quad Q1 is 3.2 T/m, the beam distribution looks like THIS shape in 6D." The shape is encoded as a sequence of spline transformations applied to a standard normal.

**Validation**: Same NLL computation on held-out shots. If val NLL stops improving for 20 epochs, training stops.

**Key hyperparameters**:
- `--n-transforms 8`: depth of the flow (8 coupling layers). More = more expressive but slower/harder to train
- `--hidden-features 128 128`: size of the conditioner MLP in each layer. Larger = more capacity to make the transformation depend on the machine settings
- `--bins 8`: number of knot points in each rational-quadratic spline. More bins = finer control over the transformation shape
- `--batch-size 64`: shots per batch. Each shot contributes 2048 log-prob evaluations
- `--grad-clip 1.0`: prevents exploding gradients from spline edge cases

---

### Step 4: Evaluate (`analyze_flow.py`)

```bash
python analyze_flow.py \
    --test-h5 particles_test.h5 \
    --model-dir model-output-flow \
    --output-dir analysis-flow \
    --n-samples 10000 --n-examples 16
```

**What it does for each test shot**:

1. **Sample from the flow**: Pass the shot's 18 machine settings as context, draw 10,000 particles from the learned distribution. This is the flow doing: base normal → 8 spline transforms (conditioned on settings) → predicted beam.

2. **Compute covariance from samples**: Take the 10,000 sampled particles, compute `np.cov(samples.T)` → 6x6 covariance matrix. This is the flow's "prediction" of the covariance, derived from its learned distribution rather than directly predicted.

3. **Compare to ground truth**: The test set has the true 2048 particles. Compute their covariance. Compare element-by-element: MAE, MAPE — same metrics as `analyze_covariance2.py` uses for the MLP model.

4. **Overlay plots**: For selected test shots, scatter plot the true particles (blue) and flow-sampled particles (red) in the three canonical 2D projections (x-px, y-py, t-pz). If the flow learned well, the red and blue clouds should overlap.

5. **Mean beam comparison**: Compute mean of each coordinate from flow samples vs true. Reports MAE per variable (mean_x, mean_px, ...).

**Key diagnostic output**:
- `covariance_diagonal_scatter.png`: for each of the 6 diagonal elements (var_x, var_px, ...), scatter plot of true vs flow-derived values. Points on the diagonal = perfect.
- `projection_overlay.png`: visual check — do the sampled beams look like the real beams?
- `test_metrics.csv`: per-element MAE and MAPE for all 36 covariance elements + 6 means. Directly comparable to the covariance model's `test_metrics.csv`.

**Why 10,000 samples at inference** (vs 2048 during training): At inference we want accurate statistics from the flow's distribution. 10,000 samples gives a covariance estimate with ~1% statistical noise, which is well below the model error we're trying to measure. During training we use fewer (2048) because we're bounded by what's available in the ground truth per shot.

---

## Current Status

- Dataset built: 2048 particles subsampled per shot from MOGA training data
- Training in progress: NSF with 8 transforms, hidden [128, 128], 8 spline bins
- Evaluation pending: will compare flow-derived covariance vs MLP-predicted covariance on same test set
