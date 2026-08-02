# DiffusionExtremums

A hypernetwork that generates MLP weights implementing a convolution, conditioned on a few (x, y) examples from a dataset. No weight-space input, no VAE bottleneck — just examples to weights.

Architecture: DatasetEncoder (transformer over example pairs) extracts the convolution signature, WeightDecoder (MLP) produces flat weight vector theta. Loss is end-to-end functional: MSE between generated MLP outputs and dataset targets.

## Quick start

```bash
# 1. Collect corpus (10 datasets x 500 MLPs, H=16)
bash scripts/collect_targets.sh

# 2. Train the hypernetwork (2000 steps)
bash scripts/train_dataset_hypernet.sh
```

## Corpus

Fixed corpus of ReLU MLPs trained to convergence on convolution datasets. The DatasetHypernet never sees MLP weights — only dataset examples.

- Default: 10 datasets x 500 MLPs, H=16, L=32
- Families: MA, DIFF, GAUSS, MATCH, RAND
- Configurable via collect_targets.sh env vars

## Metrics

- f_gen vs f_in: generated MLP should match or beat trained input MLP
- Ratio to oracle conv: how many times worse than ideal convolution
- Denoising gain: greater than 1 means generated MLP is closer to clean conv
- Toeplitzness of M = W2 @ W1: structural indicator

## Layout

| Directory | Description |
|-----------|-------------|
| src/models/dataset_hypernet.py | Core model: DatasetEncoder + WeightDecoder |
| src/models/mlp.py | Target MLP architecture (L=32, H=16) |
| src/models/weight_codec.py | Flat weight vector encoding |
| src/data/corpus_loader.py | Corpus loading |
| src/data/dataset.py | 1D convolution dataset generation |
| src/training/ | Training loop, MLP training, corpus collection |
| src/evaluation/ | Metrics: toeplitzness, kernel recovery |
| src/scripts/ | CLI entrypoints |
| scripts/ | Shell wrappers |
| data/processed/ | Collected corpus |
| results/ | Checkpoints, metrics, visualizations |
