# DiffusionExtremums

Effective-map regressor project: from a dataset **configuration** to an **effective linear map**, and from the map to **MLP weights** that implement a convolution-like (Toeplitz) mapping.

## Pipeline

```
config[14]  →  effective map[1056]  →  (SVD)  →  MLP weights[8352]  →  functional evaluation
```

- **Config features** (14-dim): family one-hot, kernel taps (zero-padded), radius, noise_std — see [`src/models/config_encoder.py`](src/models/config_encoder.py:1).
- **Effective map** (1056-dim): `M = W2 @ W1` [1024] + `b_eff` [32], which removes the gauge freedom of the `(W1, W2)` factorization of a linear MLP — see [`src/models/effective_map.py`](src/models/effective_map.py:1).
- **Oracle map**: the exact effective map built analytically from the known convolution kernel (MLP convention, `M @ x == conv1d_same(x, k)`) — see [`src/models/oracle_map.py`](src/models/oracle_map.py:1).
- **Regressor**: `14 → 256 → 512 → 512 → 1056` MLP trained on oracle (or learned) effective-map targets — see [`src/models/effective_map_regressor.py`](src/models/effective_map_regressor.py:1).
- **SVD factorization** converts a predicted effective map back into MLP weights — see [`effective_map_to_weights()`](src/models/effective_map.py:112).

The main validated result: the oracle-target regressor recovers the convolution structure on held-out configs with functional MSE ≈ **1.14×** the oracle convolution (Toeplitz-ness 0.001354, kernel cosine 0.99994). Details in [`plans/plan_ru.md`](plans/plan_ru.md:1).

## Data collection

1. **Generate dataset configs** (5 families, L=32, K∈{3,5,7}) — [`src/data/families.py`](src/data/families.py:1), [`src/data/registry.py`](src/data/registry.py:1).
2. **Collect targets**: train 50 MLPs per config to convergence (`n_train=1024`) and save the converged weights — [`src/training/collect_targets.py`](src/training/collect_targets.py:1).
3. **Convert to effective maps**: `weights.pt` → `eff_maps.pt` (per-config MLP-averaged effective maps) — [`src/scripts/convert_targets.py`](src/scripts/convert_targets.py:1).

## Training

Train the effective-map regressor on the collected targets (oracle or learned mode):

```bash
# Oracle-target baseline (default)
bash scripts/train_effective_map_regressor.sh

# Learned-target sanity check
TARGET_MODE=learned CHECKPOINT_DIR=results/regressor_sanity \
    bash scripts/train_effective_map_regressor.sh
```

Or directly:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
python -m src.scripts.train_effective_map_regressor \
    --target-mode oracle --checkpoint-dir results/regressor_oracle
```

## Evaluation

```bash
# Full oracle-regressor evaluation (metrics + figures)
bash scripts/evaluate_regressor.sh

# 3-method comparison (from-scratch MLP, learned conv, oracle conv)
bash scripts/evaluate.sh
```

Or directly:

```bash
python -m src.scripts.evaluate_oracle_regressor \
    --checkpoint results/regressor_oracle/regressor_best.pt \
    --n-eval-train 20 --n-eval-val 20

# Regenerate the primary figures/mse_comparison.png
python -m src.scripts.update_primary_mse_figure
```

Metrics: normalized/raw effective-map MSE, Toeplitz-ness (mean diagonal std), kernel recovery (cosine/L2), functional test MSE vs oracle conv / target MLP / oracle regressor.

## Smoke tests

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
python -m src.smoke.test_effective_map         # effective-map codec + SVD round-trip
python -m src.smoke.test_oracle_map            # oracle map matches conv1d_same
python -m src.smoke.test_effective_map_regressor  # regressor training smoke
python -m src.smoke.test_mlp_targets           # target collection pipeline
```

## Repo layout

| Directory | Description |
|-----------|-------------|
| `src/data/` | 1D dataset generation, families, registry |
| `src/models/` | MLP, weight codec, effective map, oracle map, config encoder, regressor |
| `src/training/` | Target collection, MLP training, regressor training loop |
| `src/evaluation/` | 3-method comparison, regressor metrics, visualization |
| `src/scripts/` | CLI entrypoints (`python -m src.scripts.*`) |
| `src/smoke/` | Smoke tests (`python -m src.smoke.*`) |
| `scripts/` | Shell wrappers (`bash scripts/*.sh`) |
| `data/processed/` | Collected/converted targets |
| `results/` | Checkpoints, metrics, logs |
| `figures/` | Generated plots |
| `plans/` | Technical specification (Russian) |
