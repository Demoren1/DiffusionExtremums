"""Train a single MLP to convergence on a 1D regression dataset (Phase 2, Approach B).

Phase 2 collects *converged MLP weights* as regression targets. For each dataset
ID we train multiple MLPs (different random initializations) to convergence and
save their flattened weight vectors.

Why this works as a target distribution (Approach B):
- The ground-truth map y = T_k x is linear, and the MLP is linear
  (``Linear(L,H) -> Linear(H,L)``), so the effective map is the matrix
  ``W2 @ W1`` (plus biases). With n_train=1024 >> L=32 the per-output
  least-squares system is well-determined -> the MLP generalizes well (low test
  MSE, close to the conv/oracle baseline). This makes the converged weights
  *good solutions* worth collecting as targets.
- The factorization ``W2 @ W1`` of the (unique) effective matrix is *non-unique*
  (gauge freedom: ``W1 -> G W1``, ``W2 -> W2 G^{-1}`` for invertible G). So
  different random initializations converge to *different* weight vectors that
  implement (approximately) the same map. This gives the target collection a
  rich weight distribution, not a single point.

Convergence strategy:
- AdamW + full-batch MSE (datasets are tiny: n_train=1024, L=32).
- Cosine LR schedule from ``lr`` to ``lr * lr_min_ratio``.
- Early stopping on a held-out validation split of the training set: we split
  ``n_train`` into ``train`` / ``val`` (default 90/10), track the best val loss,
  and stop after ``patience`` steps without improvement. A ``min_steps`` floor
  ensures we don't stop prematurely. This guarantees convergence (low train
  loss) while avoiding wasted compute.
- Gradient clipping (max norm 1.0) keeps the linear optimization stable.

The returned weight vector is flattened in the canonical order
(``fc1.weight, fc1.bias, fc2.weight, fc2.bias``, C-order) via ``WeightCodec``.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.mlp import MLPModel
from src.models.weight_codec import WeightCodec
from src.utils.seeding import set_seed

# Default device: use CUDA when available so the GPU does the work.
#
# IMPORTANT: this is computed *lazily* (on first call to ``resolve_device`` with
# ``device="auto"``), NOT at module import time. A module-level
# ``DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() ...)``
# would call ``torch.cuda.is_available()`` during import, which initializes the
# CUDA runtime in the importing process. In the multi-GPU target-collection
# pipeline (``src.training.collect_targets._collect_shard_worker``) each spawned
# worker re-imports this module; if that import initialized CUDA *before* the
# worker set ``CUDA_VISIBLE_DEVICES``, the CUDA context would bind to physical
# GPU 0 and the per-worker env-var pinning would silently fail (every worker
# would run on GPU 0). Deferring the call to first use keeps the import
# CUDA-free, so the worker can pin itself first.
_DEFAULT_DEVICE: Optional[torch.device] = None


def _default_device() -> torch.device:
    """Return (and cache) the default device: CUDA if available, else CPU.

    Computed lazily so importing this module does not initialize CUDA.
    """
    global _DEFAULT_DEVICE
    if _DEFAULT_DEVICE is None:
        _DEFAULT_DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    return _DEFAULT_DEVICE


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for training one MLP to convergence.

    Defaults are tuned for n_train=1024, L=32, H=128 (8352 params): an
    over-determined linear least-squares problem that converges quickly with
    AdamW. A100 GPUs make even large step budgets cheap.

    Attributes:
        lr: Peak AdamW learning rate (cosine-decayed to ``lr * lr_min_ratio``).
        weight_decay: AdamW weight decay. 0.0 by default: we want the MLP free
            to find *any* factorization of the effective matrix (the gauge
            freedom is the source of weight diversity); L2 would bias all inits
            toward the same minimum-norm solution and collapse the target
            distribution. Set >0 only if optimization is unstable.
        steps: Maximum number of full-batch gradient steps.
        min_steps: Minimum steps before early stopping is allowed (convergence
            floor). Ensures the MLP actually converges even if val loss plateaus
            early due to a lucky init.
        patience: Early-stopping patience: stop if val loss has not improved
            below the best-so-far by ``tol`` for this many consecutive steps.
        tol: Relative improvement on val loss required to reset the patience
            counter.
        val_frac: Fraction of the training set held out for validation / early
            stopping. The MLP is trained on the remaining ``1 - val_frac``.
        lr_min_ratio: Final LR = ``lr * lr_min_ratio`` (cosine schedule endpoint).
        grad_clip: Max gradient norm for clipping (0.0 disables).
        L: Input length (must match the dataset).
        H: Hidden width (must match the target MLP architecture).
        seed: RNG seed for the MLP initialization (different seeds -> different
            inits -> different converged weights).
        device: torch device for training ("cuda" / "cpu" / torch.device).
        eval_every: How often (in steps) to evaluate val/train loss for early
            stopping. Smaller = more responsive but slower.
    """

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
    """Outcome of training one MLP to convergence.

    Attributes:
        theta: Flattened converged weight vector, shape ``[D]`` (8352), on CPU,
            float32, in the canonical order (fc1.weight, fc1.bias, fc2.weight,
            fc2.bias, C-order). This is the regression target.
        train_mse: Final MSE on the (internal) training split.
        val_mse: Best validation MSE observed (used for early stopping).
        test_mse: MSE on the provided test set (generalization to new inputs).
        n_steps: Number of steps actually run (<= ``steps``; may be less if early
            stopping triggered).
        converged: True if early stopping triggered (val loss plateaued), False
            if the step budget was exhausted.
        best_step: Step at which the best val loss was observed.
    """

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
    """Randomly split (x, y) into train/val along the batch dim.

    Args:
        x: ``[N, L]``.
        y: ``[N, L]``.
        val_frac: Fraction held out for validation.
        rng: NumPy ``Generator`` for the permutation (reproducible).

    Returns:
        ``(x_tr, y_tr, x_va, y_va)``.
    """
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
    """Train one MLP (random init from ``config.seed``) to convergence.

    The MLP is the linear ``MLPModel`` (``Linear(L,H) -> Linear(H,L)``, 8352
    params). Training uses full-batch AdamW + cosine LR + early stopping on a
    held-out validation split of the training set.

    Args:
        x_train: ``[n_train, L]`` training inputs.
        y_train: ``[n_train, L]`` training targets.
        x_test: ``[n_test, L]`` test inputs (generalization to new inputs).
        y_test: ``[n_test, L]`` test targets.
        config: Training hyperparameters. If None, uses ``TrainConfig()`` with
            ``seed=0``.

    Returns:
        ``TrainResult`` with the flattened converged weights (``theta``) and
        train/val/test MSE. ``theta`` is on CPU, float32, in the canonical
        flatten order.
    """
    cfg = config if config is not None else TrainConfig()
    device = cfg.resolve_device()

    # Reproducible init: the seed selects the random initialization, so
    # different seeds -> different inits -> different converged weights.
    set_seed(cfg.seed)

    model = MLPModel(L=cfg.L, H=cfg.H).to(device)
    codec = WeightCodec(L=cfg.L, H=cfg.H)
    loss_fn = nn.MSELoss()

    # Move data to device. Split train into train/val for early stopping.
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    # Val split uses a numpy RNG seeded by the training seed (deterministic,
    # independent of the torch init RNG state).
    split_rng = np.random.default_rng(cfg.seed + 1)
    x_tr, y_tr, x_va, y_va = _split_train_val(x_train, y_train, cfg.val_frac, split_rng)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine LR from cfg.lr to cfg.lr * lr_min_ratio over cfg.steps.
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

        # Periodic eval for early stopping (after the min_steps floor).
        if step >= cfg.min_steps and (step % cfg.eval_every == 0 or step == cfg.steps):
            model.eval()
            with torch.no_grad():
                val_mse = loss_fn(model(x_va), y_va).item()
            # Relative improvement check.
            improved = (best_val - val_mse) > cfg.tol * max(1.0, abs(best_val))
            if improved:
                best_val = val_mse
                best_step = step
                steps_since_best = 0
                # Snapshot the best model state (CPU copy to save GPU memory).
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            else:
                steps_since_best += cfg.eval_every
                if steps_since_best >= cfg.patience:
                    converged = True
                    break

    # If we never beat inf (e.g. min_steps==0 edge case), snapshot the final state.
    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_step = cfg.steps if not converged else best_step

    # Load the best (lowest val) model and compute final metrics.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_mse = loss_fn(model(x_tr), y_tr).item()
        val_mse = loss_fn(model(x_va), y_va).item()
        test_mse = loss_fn(model(x_test), y_test).item()

    # Flatten converged weights in the canonical order (CPU, float32).
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
