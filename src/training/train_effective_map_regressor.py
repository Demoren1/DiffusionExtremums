"""Training loop for the deterministic effective-map regressor (sanity check).

This is a supervised baseline. It trains a small MLP (``EffectiveMapRegressor``)
to map the 14-dim config feature vector directly to the 1056-dim effective
linear map, using standard regression (AdamW + MSE in normalized effective-map
space + cosine LR + early stopping on held-out config validation MSE).

Data protocol (per task spec):
- One target per config: average the 50 per-MLP effective maps along the MLP
  axis, making the target shape ``[n_configs, 1056]``.
- Train/validation split **by config**: a deterministic split is reproduced via
  ``np.random.default_rng(seed)`` with ``val_configs`` held out.
- ``WeightNormalizer`` is fit **on only train config targets**; training is in
  normalized space and predictions are inverse-transformed for functional
  evaluation.
- Split indices are written into the checkpoint/result artifacts.

Target modes
-----------
``target_mode`` selects how the per-config regression target is constructed:

- ``"learned"`` (default, backwards-compatible): average the 50 per-MLP
  effective maps from ``eff_maps.pt``. These targets contain finite-sample MLP
  estimation noise.
- ``"oracle"``: build the **exact** effective map from the config's known
  data-generation kernel via
  :func:`src.models.oracle_map.kernel_to_oracle_effective_map` (MLP convention,
  ``b_eff = 0``). This is a deterministic, noise-free target that demonstrates
  the ceiling achievable when the kernel is supplied as conditioning. The
  ``eff_maps.pt`` file is still loaded (for the config split / n_mlp metadata
  and for optional comparison), but the regression targets are the oracle maps.

The checkpoint format stores: model state, normalizer state, the training
config, and the train/val config indices.
"""
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.config_encoder import CONFIG_FEATURE_DIM, configs_to_features
from src.models.effective_map import DEFAULT_EFF_D, DEFAULT_H, DEFAULT_L
from src.models.effective_map_regressor import (
    EffectiveMapRegressor,
    RegressorConfig,
)
from src.models.oracle_map import kernel_to_oracle_effective_map
from src.models.weight_norm import WeightNormalizer
from src.utils.seeding import set_seed

# Try to import TensorBoard; degrade gracefully if unavailable.
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore
    _HAS_TB = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainRegressorConfig:
    """Hyperparameters for the effective-map regressor training loop.

    Attributes:
        lr: Peak AdamW learning rate (cosine-decayed to ``lr_min``).
        weight_decay: AdamW weight decay.
        lr_min: Final LR for the cosine schedule.
        batch_size: Training batch size (number of configs per step).
        max_steps: Total optimizer steps. If None, train for ``epochs``.
        epochs: Full passes over the train configs (used if max_steps is None).
        grad_clip: Max gradient norm (0 disables).
        log_every: Log train loss every N steps.
        eval_every: Evaluate val loss every N steps.
        save_every: Save a checkpoint every N steps.
        patience: Early-stopping patience (evals without improvement). 0 = off.
        device: torch device ("cuda"/"cpu"/"auto").
        seed: RNG seed.
        checkpoint_dir: Directory for checkpoints + logs.
        log_dir: TensorBoard log dir (None = <checkpoint_dir>/tensorboard).
        resume: Optional checkpoint path to resume from.
        # Data
        targets_dir: Directory with eff_maps.pt + configs.json.
        val_configs: Number of configs held out for validation (deterministic
            split from ``seed``).
        # Model
        hidden_dims: Hidden layer widths.
        activation: "gelu" or "silu".
        use_residual: Residual connection across equal-width hidden layers.
        use_layer_norm: LayerNorm before activations.
        dropout: Dropout probability (0 = off).
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_min: float = 1e-6
    batch_size: int = 64
    max_steps: Optional[int] = None
    epochs: Optional[int] = 20000
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 2000
    patience: int = 50
    device: str = "cuda"
    seed: int = 0
    checkpoint_dir: str = "results/regressor_sanity"
    log_dir: Optional[str] = None
    resume: Optional[str] = None
    # Data
    targets_dir: str = "data/processed/targets_eff"
    val_configs: int = 50
    target_mode: str = "learned"
    # Model
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 512])
    activation: str = "gelu"
    use_residual: bool = True
    use_layer_norm: bool = True
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Data loading + split
# ---------------------------------------------------------------------------

@dataclass
class RegressorDataBundle:
    """Loaded, split, and standardized regressor corpus (one target per config).

    Attributes:
        features_all: ``[n_configs, 14]`` config features for all configs.
        targets_all: ``[n_configs, 1056]`` per-config effective maps used as the
            regression target. In ``"learned"`` mode these are the MLP-averaged
            maps; in ``"oracle"`` mode these are the exact kernel-derived oracle
            maps.
        targets_learned_all: ``[n_configs, 1056]`` MLP-averaged learned maps
            (always available, for comparison / Toeplitz analysis). In
            ``"learned"`` mode this equals ``targets_all``.
        train_features: ``[n_train_configs, 14]``.
        train_targets_raw: ``[n_train_configs, 1056]`` raw effective maps.
        val_features: ``[n_val_configs, 14]``.
        val_targets_raw: ``[n_val_configs, 1056]`` raw effective maps.
        train_targets_norm: standardized train targets.
        val_targets_norm: standardized val targets.
        normalizer: ``WeightNormalizer`` fit on train targets only.
        train_config_indices: sorted list of train config indices.
        val_config_indices: sorted list of val config indices.
        configs: full list of config dicts.
        n_configs: total number of configs.
        n_mlp: number of MLPs per config (50).
        D_eff: effective-map dimension (1056).
        split_source: "deterministic_seed" (split provenance for artifacts).
        target_mode: "learned" or "oracle".
    """

    features_all: torch.Tensor
    targets_all: torch.Tensor
    targets_learned_all: torch.Tensor
    train_features: torch.Tensor
    train_targets_raw: torch.Tensor
    val_features: torch.Tensor
    val_targets_raw: torch.Tensor
    train_targets_norm: torch.Tensor
    val_targets_norm: torch.Tensor
    normalizer: WeightNormalizer
    train_config_indices: List[int]
    val_config_indices: List[int]
    configs: List[dict]
    n_configs: int
    n_mlp: int
    D_eff: int
    split_source: str
    target_mode: str


def deterministic_split(n_configs: int, val_configs: int,
                        seed: int) -> Tuple[List[int], List[int]]:
    """Reproduce the deterministic config-level split.

    Uses ``np.random.default_rng(seed)`` so the split is identical across runs
    and across callers (e.g. the evaluation CLI).

    Returns:
        ``(train_indices, val_indices)``, each sorted.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_configs)
    n_val = min(int(val_configs), max(0, n_configs - 1))
    val_idx = sorted(int(i) for i in perm[:n_val])
    train_idx = sorted(int(i) for i in perm[n_val:])
    return train_idx, val_idx


def load_regressor_data(config: TrainRegressorConfig) -> RegressorDataBundle:
    """Load effective maps, build targets, split by config, normalize.

    Steps:
      1. Load ``eff_maps.pt`` ``[n_configs, n_mlp, D_eff]`` and ``configs.json``.
      2. Average over the MLP axis -> ``[n_configs, D_eff]`` (the *learned*
         target, one per config).
      3. Select the regression target via ``config.target_mode``:
         - ``"learned"`` (default): use the MLP-averaged maps.
         - ``"oracle"``: build the exact kernel-derived oracle effective map
           per config via :func:`kernel_to_oracle_effective_map` (MLP
           convention, ``b_eff = 0``). The learned maps are still retained in
           ``targets_learned_all`` for comparison.
      4. Compute 14-dim config features for all configs.
      5. Split configs into train/val deterministically from ``seed``.
      6. Fit ``WeightNormalizer`` on train targets only.
      7. Standardize train and val targets.

    Args:
        config: Training config.

    Returns:
        A ``RegressorDataBundle``.
    """
    if config.target_mode not in ("learned", "oracle"):
        raise ValueError(
            f"target_mode must be 'learned' or 'oracle', got "
            f"{config.target_mode!r}")
    tdir = config.targets_dir
    eff_path = os.path.join(tdir, "eff_maps.pt")
    cfg_path = os.path.join(tdir, "configs.json")
    for p in (eff_path, cfg_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing required file: {p}")

    eff_maps = torch.load(eff_path, map_location="cpu").float()
    if eff_maps.dim() != 3:
        raise ValueError(
            f"eff_maps.pt must be 3-D [n_configs, n_mlp, D_eff], got "
            f"{tuple(eff_maps.shape)}")
    n_configs, n_mlp, D_eff = eff_maps.shape
    if D_eff != DEFAULT_EFF_D:
        print(f"[load_regressor_data] WARNING: D_eff={D_eff} != "
              f"DEFAULT_EFF_D={DEFAULT_EFF_D}")

    with open(cfg_path, "r") as f:
        configs = json.load(f)
    if len(configs) != n_configs:
        raise ValueError(
            f"configs.json has {len(configs)} entries != eff_maps axis 0 "
            f"{n_configs}")

    # Average over the MLP axis -> one learned target per config.
    targets_learned_all = eff_maps.mean(dim=1).contiguous()  # [n_configs, D_eff]
    print(f"[load_regressor_data] eff_maps {tuple(eff_maps.shape)} -> "
          f"per-config learned targets {tuple(targets_learned_all.shape)} "
          f"(averaged over {n_mlp} MLPs)")

    # Select the regression target.
    if config.target_mode == "oracle":
        # Build the exact oracle effective map from each config's kernel.
        oracle_targets = torch.stack([
            kernel_to_oracle_effective_map(
                torch.tensor(cfg["kernel"], dtype=torch.float32),
                L=int(cfg.get("L", DEFAULT_L)),
            )
            for cfg in configs
        ], dim=0).contiguous()  # [n_configs, D_eff]
        targets_all = oracle_targets
        print(f"[load_regressor_data] target_mode='oracle': built exact "
              f"kernel-derived oracle targets {tuple(targets_all.shape)} "
              f"(b_eff = 0)")
    else:
        targets_all = targets_learned_all
        print(f"[load_regressor_data] target_mode='learned': using "
              f"MLP-averaged targets {tuple(targets_all.shape)}")

    # Config features for all configs.
    features_all = configs_to_features(configs, dtype=torch.float32)
    if features_all.shape != (n_configs, CONFIG_FEATURE_DIM):
        raise ValueError(
            f"config features shape {tuple(features_all.shape)} != "
            f"({n_configs}, {CONFIG_FEATURE_DIM})")

    # Split by config (deterministic from seed).
    train_idx, val_idx = deterministic_split(
        n_configs, config.val_configs, config.seed)
    split_source = (f"deterministic_seed={config.seed},"
                    f"val_configs={config.val_configs}")
    print(f"[load_regressor_data] split: {split_source} "
          f"({len(train_idx)} train / {len(val_idx)} val configs)")

    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)

    train_features = features_all[train_idx_t].contiguous()
    val_features = features_all[val_idx_t].contiguous()
    train_targets_raw = targets_all[train_idx_t].contiguous()
    val_targets_raw = targets_all[val_idx_t].contiguous()

    # Fit normalizer on TRAIN targets only.
    normalizer = WeightNormalizer.fit(train_targets_raw)
    train_targets_norm = normalizer.standardize(train_targets_raw)
    val_targets_norm = normalizer.standardize(val_targets_raw)
    print(f"[load_regressor_data] normalizer: {normalizer}")

    return RegressorDataBundle(
        features_all=features_all,
        targets_all=targets_all,
        targets_learned_all=targets_learned_all,
        train_features=train_features,
        train_targets_raw=train_targets_raw,
        val_features=val_features,
        val_targets_raw=val_targets_raw,
        train_targets_norm=train_targets_norm,
        val_targets_norm=val_targets_norm,
        normalizer=normalizer,
        train_config_indices=train_idx,
        val_config_indices=val_idx,
        configs=configs,
        n_configs=n_configs,
        n_mlp=n_mlp,
        D_eff=D_eff,
        split_source=split_source,
        target_mode=config.target_mode,
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model: EffectiveMapRegressor,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    normalizer: WeightNormalizer, step: int,
                    config: TrainRegressorConfig,
                    train_config_indices: List[int],
                    val_config_indices: List[int],
                    best_val_mse: float,
                    split_source: str) -> None:
    """Save a full regressor checkpoint (resumable)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "step": int(step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "normalizer_state": normalizer.state_dict(),
        "config": asdict(config),
        "train_config_indices": train_config_indices,
        "val_config_indices": val_config_indices,
        "best_val_mse": float(best_val_mse),
        "split_source": split_source,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: torch.device) -> Dict:
    """Load a checkpoint dict (mapping tensors to ``device``)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(nn.functional.mse_loss(pred, target).item())


@torch.no_grad()
def eval_mse(model: EffectiveMapRegressor, features: torch.Tensor,
             targets_norm: torch.Tensor, device: torch.device) -> float:
    """Mean MSE in normalized effective-map space over a feature set."""
    model.eval()
    pred = model(features.to(device))
    return _mse(pred, targets_norm.to(device))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_regressor(config: TrainRegressorConfig,
                    show_progress: bool = True) -> Dict:
    """Run the effective-map regressor training loop.

    Args:
        config: Training hyperparameters.
        show_progress: Print progress to stdout.

    Returns:
        Dict with final metrics: ``step``, ``train_mse_norm``, ``val_mse_norm``,
        ``best_val_mse``, ``checkpoint_path``, ``split_source``.
    """
    set_seed(config.seed)
    device = _resolve_device(config.device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    bundle = load_regressor_data(config)
    normalizer = bundle.normalizer

    train_feats = bundle.train_features.to(device)
    train_tnorm = bundle.train_targets_norm.to(device)
    val_feats = bundle.val_features.to(device)
    val_tnorm = bundle.val_targets_norm.to(device)
    n_train = train_feats.shape[0]
    n_val = val_feats.shape[0]

    steps_per_epoch = max(1, math.ceil(n_train / config.batch_size))
    if config.max_steps is not None:
        total_steps = int(config.max_steps)
    elif config.epochs is not None:
        total_steps = int(config.epochs) * steps_per_epoch
    else:
        raise ValueError("either max_steps or epochs must be set")

    if show_progress:
        print(f"[train_regressor] device: {device}")
        print(f"[train_regressor] train configs: {n_train}, val configs: {n_val}")
        print(f"[train_regressor] D_eff={bundle.D_eff}, "
              f"batch_size={config.batch_size}, "
              f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
        print(f"[train_regressor] split_source: {bundle.split_source}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = EffectiveMapRegressor(
        RegressorConfig(
            feature_dim=CONFIG_FEATURE_DIM,
            hidden_dims=list(config.hidden_dims),
            output_dim=bundle.D_eff,
            activation=config.activation,
            use_residual=config.use_residual,
            use_layer_norm=config.use_layer_norm,
            dropout=config.dropout,
        )
    ).to(device)
    if show_progress:
        print(f"[train_regressor] model: {model}")
        print(f"[train_regressor] params: {model.n_params():,} "
              f"({model.n_params() / 1e6:.3f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.lr_min)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    best_val_mse = float("inf")
    if config.resume is not None and os.path.exists(config.resume):
        ckpt = load_checkpoint(config.resume, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = int(ckpt["step"])
        best_val_mse = float(ckpt.get("best_val_mse", float("inf")))
        if show_progress:
            print(f"[train_regressor] resumed from {config.resume} at step "
                  f"{start_step}, best_val_mse={best_val_mse:.6f}")
    else:
        # Initialize best with a quick eval.
        best_val_mse = eval_mse(model, val_feats, val_tnorm, device)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    log_csv = os.path.join(config.checkpoint_dir, "train_log.csv")
    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])
    csv_file.flush()

    # Save split metadata.
    with open(os.path.join(config.checkpoint_dir, "split.json"), "w") as f:
        json.dump({
            "train_config_indices": bundle.train_config_indices,
            "val_config_indices": bundle.val_config_indices,
            "split_source": bundle.split_source,
            "n_train_configs": len(bundle.train_config_indices),
            "n_val_configs": len(bundle.val_config_indices),
            "n_configs": bundle.n_configs,
            "n_mlp": bundle.n_mlp,
            "D_eff": bundle.D_eff,
            "target_mode": bundle.target_mode,
        }, f, indent=2)

    tb_writer = None
    if _HAS_TB:
        tb_log_dir = (config.log_dir if config.log_dir is not None
                      else os.path.join(config.checkpoint_dir, "tensorboard"))
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        if show_progress:
            print(f"[train_regressor] tensorboard log_dir: {tb_log_dir}")

    final_ckpt = os.path.join(config.checkpoint_dir, "regressor_final.pt")
    best_ckpt = os.path.join(config.checkpoint_dir, "regressor_best.pt")
    periodic_ckpt = os.path.join(config.checkpoint_dir, "regressor_step.pt")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    step = start_step
    train_loss_smoothed = 0.0
    t0 = time.time()
    no_improve = 0
    model.train()

    # Precompute a shuffled index iterator for mini-batching configs.
    rng = np.random.default_rng(config.seed)

    def _next_batch() -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.from_numpy(
            rng.choice(n_train, size=min(config.batch_size, n_train),
                       replace=False)).long().to(device)
        return train_feats[idx], train_tnorm[idx]

    try:
        while step < total_steps:
            optimizer.zero_grad()
            xb, yb = _next_batch()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            train_loss_smoothed = (
                0.95 * train_loss_smoothed + 0.05 * float(loss.item())
                if train_loss_smoothed > 0 else float(loss.item()))

            if step % config.log_every == 0 or step == total_steps:
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                if show_progress:
                    print(f"  step {step:>6d}/{total_steps}  "
                          f"loss={float(loss.item()):.6f}  "
                          f"smoothed={train_loss_smoothed:.6f}  "
                          f"lr={lr_now:.2e}  t={elapsed:.1f}s")
                csv_writer.writerow(
                    [step, float(loss.item()), "", lr_now, elapsed])
                csv_file.flush()
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", float(loss.item()), step)
                    tb_writer.add_scalar("train/loss_smoothed",
                                         train_loss_smoothed, step)
                    tb_writer.add_scalar("train/lr", lr_now, step)

            if step % config.eval_every == 0 or step == total_steps:
                val_mse = eval_mse(model, val_feats, val_tnorm, device)
                tr_mse = eval_mse(model, train_feats, train_tnorm, device)
                lr_now = scheduler.get_last_lr()[0]
                if show_progress:
                    print(f"  [eval] step {step}: val_mse={val_mse:.6f}  "
                          f"train_mse={tr_mse:.6f}")
                csv_writer.writerow(
                    [step, tr_mse, val_mse, lr_now, time.time() - t0])
                csv_file.flush()
                if tb_writer is not None:
                    tb_writer.add_scalar("val/mse", val_mse, step)
                    tb_writer.add_scalar("train/mse", tr_mse, step)

                # Early stopping + best checkpoint.
                improved = val_mse < best_val_mse - 1e-7
                if improved:
                    best_val_mse = val_mse
                    no_improve = 0
                    save_checkpoint(
                        best_ckpt, model, optimizer, scheduler, normalizer,
                        step, config, bundle.train_config_indices,
                        bundle.val_config_indices, best_val_mse,
                        bundle.split_source)
                    if show_progress:
                        print(f"  [best] new best val_mse={best_val_mse:.6f} "
                              f"-> {best_ckpt}")
                else:
                    no_improve += 1
                    if config.patience > 0 and no_improve >= config.patience:
                        if show_progress:
                            print(f"  [early-stop] no improvement for "
                                  f"{config.patience} evals; stopping at "
                                  f"step {step}")
                        break

            if config.save_every > 0 and step % config.save_every == 0:
                save_checkpoint(
                    periodic_ckpt, model, optimizer, scheduler, normalizer,
                    step, config, bundle.train_config_indices,
                    bundle.val_config_indices, best_val_mse,
                    bundle.split_source)
                if show_progress:
                    print(f"  [ckpt] saved periodic checkpoint at step {step}")
    finally:
        csv_file.close()
        if tb_writer is not None:
            tb_writer.close()

    # Final checkpoint (always the last model, not necessarily the best).
    save_checkpoint(
        final_ckpt, model, optimizer, scheduler, normalizer, step, config,
        bundle.train_config_indices, bundle.val_config_indices,
        best_val_mse, bundle.split_source)
    if show_progress:
        print(f"[train_regressor] saved final checkpoint -> {final_ckpt}")

    final_val = eval_mse(model, val_feats, val_tnorm, device)
    final_train = eval_mse(model, train_feats, train_tnorm, device)
    if show_progress:
        print(f"[train_regressor] done. step={step} "
              f"train_mse(norm)={final_train:.6f} "
              f"val_mse(norm)={final_val:.6f} "
              f"best_val_mse={best_val_mse:.6f}")

    return {
        "step": step,
        "train_mse_norm": final_train,
        "val_mse_norm": final_val,
        "best_val_mse": best_val_mse,
        "checkpoint_path": final_ckpt,
        "best_checkpoint_path": best_ckpt,
        "split_source": bundle.split_source,
        "n_params": model.n_params(),
    }
