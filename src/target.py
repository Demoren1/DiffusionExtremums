"""Target periodic function and the small MLP that approximates it.

The diffusion model (later) will learn to *generate the weights* of ``TargetMLP``
so that the resulting network reproduces the target function and its extrema.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import TargetConfig, TargetMLPConfig


# --------------------------------------------------------------------------- #
# Target function  y = sin(k0 x + b0) + sin(k1 x + b1)
# --------------------------------------------------------------------------- #
def target_function(x: torch.Tensor, cfg: TargetConfig) -> torch.Tensor:
    """Evaluate the periodic target function.

    x: tensor of shape (..., 1) or (...,) — input coordinate(s).
    Returns tensor of the same leading shape.
    """
    if x.dim() >= 1 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    (k0, k1), (b0, b1) = cfg.k, cfg.b
    return torch.sin(k0 * x + b0) + torch.sin(k1 * x + b1)


def sample_target_grid(cfg: TargetConfig, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (x, y) tensors sampled uniformly on [x_min, x_max].

    x: (n_samples, 1), y: (n_samples, 1).
    """
    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.n_samples, device=device).unsqueeze(1)
    y = target_function(x, cfg).unsqueeze(1)
    return x, y


# --------------------------------------------------------------------------- #
# Target MLP  1 -> hidden (tanh) -> 1
# --------------------------------------------------------------------------- #
class TargetMLP(nn.Module):
    """A tiny MLP whose *weights* are the object the diffusion model generates.

    Architecture: Linear(in_dim, hidden) -> activation -> Linear(hidden, out_dim).
    The hidden width is a free parameter (configurable, default > 10).
    """

    def __init__(self, cfg: TargetMLPConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        self.act = nn.Tanh() if cfg.activation == "tanh" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    # -- convenience wrappers around src.utils (kept here for locality) --------
    def flatten(self) -> torch.Tensor:
        from .utils import flatten_params
        return flatten_params(self)

    def load_flat(self, flat: torch.Tensor) -> None:
        from .utils import unflatten_params
        unflatten_params(flat, self)


def canonicalize_mlp(mlp: TargetMLP) -> TargetMLP:
    """Remove permutation and sign-flip symmetries of the hidden neurons (in-place).

    For a ``1 -> hidden(tanh) -> 1`` network each hidden neuron ``j`` is defined by
    incoming weight ``w1[:, j]``, bias ``b1[j]`` and outgoing weight ``w2[j, :]``.
    Two symmetries produce *equivalent* networks with different weight vectors:

    * **Sign-flip** — for tanh, negating ``(w1[:, j], b1[j])`` *and* ``w2[j, :]``
      leaves the network output unchanged.  We canonicalise by making the first
      non-zero element of ``w1[:, j]`` positive.
    * **Permutation** — reordering hidden neurons swaps columns of ``w1`` and rows
      of ``w2``.  We sort neurons by a deterministic key (incoming weight norm,
      then bias, then outgoing weight).

    After canonicalisation, equivalent networks map to the *same* weight vector,
    which reduces spurious multimodality in the diffusion training data.

    Works for any ``in_dim`` / ``out_dim``; returns ``mlp`` for convenience.
    """
    w1 = mlp.fc1.weight.data   # (hidden, in_dim)
    b1 = mlp.fc1.bias.data     # (hidden,)
    w2 = mlp.fc2.weight.data   # (out_dim, hidden)
    b2 = mlp.fc2.bias.data     # (out_dim,)
    H = w1.shape[0]

    # --- sign-flip canonicalisation --------------------------------------- #
    for j in range(H):
        col = w1[j]  # (in_dim,)
        # Find the first non-zero element (with a small tolerance).
        nonzero = col.abs() > 1e-12
        if nonzero.any():
            idx = int(nonzero.nonzero(as_tuple=False)[0].item())
            if col[idx] < 0:
                w1[j] = -w1[j]
                b1[j] = -b1[j]
                w2[:, j] = -w2[:, j]

    # --- permutation canonicalisation ------------------------------------- #
    # Deterministic sort key per neuron: (||w1[j]||, b1[j], ||w2[:, j]||, ...).
    # Using the full flattened incoming weight as tie-breaker for stability.
    keys = []
    for j in range(H):
        keys.append((
            float(w1[j].norm().item()),
            float(b1[j].item()),
            float(w2[:, j].norm().item()),
            *[float(v) for v in w1[j].tolist()],
        ))
    order = sorted(range(H), key=lambda j: keys[j])

    w1[:] = w1[order]
    b1[:] = b1[order]
    # w2 has shape (out_dim, hidden) — reorder columns.
    w2[:] = w2[:, order]
    # b2 is unaffected by hidden-neuron reordering.

    return mlp


def build_target_mlp(cfg: TargetMLPConfig, device=None) -> TargetMLP:
    mlp = TargetMLP(cfg)
    if device is not None:
        mlp = mlp.to(device)
    return mlp
