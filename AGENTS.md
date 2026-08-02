# Design Principles

## Core thesis

A hypernetwork should generate MLP weights that implement a convolution,
conditioned on a few (x, y) examples from a dataset. No weight-space input,
no VAE bottleneck — just **examples → weights**.

## What works

### DatasetEncoder: few examples → embedding

A small transformer (1 layer, 4 heads) over K example pairs extracts the
convolution signature. It sees K enc points; loss is computed on different
N_loss points — no cheating possible. The encoder must learn the underlying
pattern, not memorize.

### WeightDecoder: embedding → θ

A plain MLP maps the dataset embedding to flat weight vector θ. No low-rank
factorization, no outer products — direct prediction. Zero-init the last layer
so initial output is near-zero, letting the model discover the right scale.

### Training: end-to-end functional loss

`L = MSE(f(θ)(x), y)` — compare the generated MLP's outputs to dataset
targets. No weight-space metrics, no intermediate representations. The
gradient flows directly from function mismatch to weight prediction.

## What doesn't work

- **Functional distillation of input MLPs** (`f_gen ≈ f_in`): gradient signal
  is too weak — f_in is already near-oracle, there's nothing to improve.
- **VAE bottleneck**: frozen decoder kills variation in generated weights;
  all latent codes produce nearly identical θ.
- **Low-rank weight factorization** (`W = Σ s_k · a_k ⊗ b_k`): cannot
  represent convolution-like weight matrices even at full rank.
- **Weight-space reconstruction**: MLP weights encode permutation noise;
  learning to reconstruct them teaches permutation variance, not function.
- **Dataset ID embedding without examples**: a learned lookup table has no
  information about the actual convolution — the model must *see* the data.

## Architecture invariants

- Encoder and decoder share no parameters — clean separation of
  «understand the dataset» and «produce the weights».
- Encoder sees different points than the loss — prevents memorization.
- No VAE, no hypernetwork-on-weights, no effective-map intermediate.
- MLP hidden dimension small enough that SGD leaves room for improvement,
  large enough to learn the function (H=16 for L=32 conv1d).

## Metrics that matter

- **f_gen vs f_in**: generated MLP should match or beat the trained input MLP.
- **Ratio to oracle conv**: how many times worse than the ideal convolution.
- **Denoising gain** (>1 means generated MLP is closer to clean conv).
- **Toeplitzness of M = W2 @ W1**: structural indicator (but not the goal —
  ReLU non-linearity allows non-Toeplitz M for convolution-equivalent functions).

## Corpus

- Fixed corpus of MLPs trained to convergence on convolution datasets.
- Used only to train MLPs; the DatasetHypernet never sees MLP weights.
- Default: 10 datasets × 500 MLPs, H=16, L=32.
