"""Train a single MLP to convergence on a 1D regression dataset."""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.mlp import MLPModel
from src.models.weight_codec import WeightCodec
from src.utils.seeding import set_seed

# Device resolved lazily: a module-level torch.cuda.is_available() call would
# initialize CUDA in spawned workers before they pin CUDA_VISIBLE_DEVICES.
_DEFAULT_DEVICE: Optional[torch.device] = None


def _default_device() -> torch.device:
    global _DEFAULT_DEVICE
    if _DEFAULT_DEVICE is None:
        _DEFAULT_DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    return _DEFAULT_DEVICE


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for training one MLP to convergence."""

    lr: float = 3e-3
    weight_decay: float = 0.0
    steps: int = 5000
    min_steps: int = 500
    patience: int = 300
    tol: float = 1e-6
    val_frac: float = 0.1
    lr_min_ratio: float = 0.01
    grad_clip: float = 1.0
    L: int = 32
    H: int = 128
    seed: int = 0
    device: str = "auto"
    eval_every: int = 25

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return _default_device()
        return torch.device(self.device)


@dataclass
class TrainResult:
    """Outcome of training one MLP to convergence."""

    theta: torch.Tensor
    train_mse: float
    val_mse: float
    test_mse: float
    n_steps: int
    converged: bool
    best_step: int


def _split_train_val(
    x: torch.Tensor, y: torch.Tensor, val_frac: float, rng: np.random.Generator
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    n_val = max(1, int(round(n * val_frac)))
    n_val = min(n_val, n - 1)  # keep at least one train sample
    perm = rng.permutation(n)
    val_idx = torch.from_numpy(perm[:n_val]).long()
    tr_idx = torch.from_numpy(perm[n_val:]).long()
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def train_mlp_to_convergence(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    config: Optional[TrainConfig] = None,
) -> TrainResult:
    """Train one MLP (random init from config.seed) to convergence."""
    cfg = config if config is not None else TrainConfig()
    device = cfg.resolve_device()

    set_seed(cfg.seed)

    model = MLPModel(L=cfg.L, H=cfg.H).to(device)
    codec = WeightCodec(L=cfg.L, H=cfg.H)
    loss_fn = nn.MSELoss()

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    # Val split uses a numpy RNG seeded by the training seed.
    split_rng = np.random.default_rng(cfg.seed + 1)
    x_tr, y_tr, x_va, y_va = _split_train_val(x_train, y_train, cfg.val_frac, split_rng)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.steps, eta_min=cfg.lr * cfg.lr_min_ratio)

    best_val = float("inf")
    best_state: Optional[dict] = None
    best_step = 0
    steps_since_best = 0
    converged = False

    for step in range(1, cfg.steps + 1):
        model.train()
        opt.zero_grad()
        pred = model(x_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        opt.step()
        scheduler.step()

        if step >= cfg.min_steps and (step % cfg.eval_every == 0 or step == cfg.steps):
            model.eval()
            with torch.no_grad():
                val_mse = loss_fn(model(x_va), y_va).item()
            improved = (best_val - val_mse) > cfg.tol * max(1.0, abs(best_val))
            if improved:
                best_val = val_mse
                best_step = step
                steps_since_best = 0
                # Snapshot best state (CPU copy to save GPU memory).
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            else:
                steps_since_best += cfg.eval_every
                if steps_since_best >= cfg.patience:
                    converged = True
                    break

    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_step = cfg.steps if not converged else best_step

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_mse = loss_fn(model(x_tr), y_tr).item()
        val_mse = loss_fn(model(x_va), y_va).item()
        test_mse = loss_fn(model(x_test), y_test).item()

    theta = codec.pack_model(model).detach().cpu().float()

    return TrainResult(
        theta=theta,
        train_mse=train_mse,
        val_mse=val_mse,
        test_mse=test_mse,
        n_steps=step,
        converged=converged,
        best_step=best_step,
    )
