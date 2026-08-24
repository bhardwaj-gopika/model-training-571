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

## Current Status

- Dataset built: 2048 particles subsampled per shot from MOGA training data
- Training in progress: NSF with 8 transforms, hidden [128, 128], 8 spline bins
- Evaluation pending: will compare flow-derived covariance vs MLP-predicted covariance on same test set
