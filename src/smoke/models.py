"""Models for the conv-vs-MLP smoke test (Phase 1).

The shared target MLP architecture now lives in ``src/models/`` (single source
of truth for Phase 2+). This module re-exports ``MLPModel`` from there and keeps
the Phase-1-specific ``Conv1dModel`` (the inductive-bias baseline) here.

Two models, both mapping R^L -> R^L:

- ``Conv1dModel``: a single 1D convolution with kernel size K (a few params).
  This is the inductive-bias model: weight sharing + locality.

- ``MLPModel``: a linear two-layer MLP ``Linear(L, H) -> Linear(H, L)`` with
  H=128 (8352 params, ~1000x more than the conv). This is the
  over-parameterized baseline: enough capacity to represent the convolution
  (it is a linear map), but no inductive bias, so it overfits at small n_train.

Per ``plans/plan.md`` Section 2.2, the v1 MLP is *linear* (no activation): the
ground-truth map y = T_k x is linear, a linear MLP can represent it exactly,
and a linear MLP trained from scratch on few samples is a rank-deficient /
overfit least-squares problem -> poor generalization. This isolates the
inductive-bias question from capacity.
"""
import torch
import torch.nn as nn

# Re-export the shared target MLP so there is a single source of truth.
from src.models.mlp import MLPModel  # noqa: F401  (re-exported)


class Conv1dModel(nn.Module):
    """1D convolution baseline: a single conv1d layer with 'same' padding.

    The kernel size K is chosen to match (or exceed) the ground-truth kernel
    size of the dataset. The model has only K parameters (the conv weights)
    plus a bias, so it is the inductive-bias model.
    """

    def __init__(self, L: int = 32, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        # in_channels=1, out_channels=1, groups=1. 'same' padding for odd K.
        pad = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=kernel_size,
            padding=pad, padding_mode="zeros", bias=bias,
        )
        self.L = L
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] -> add channel dim -> [B, 1, L] -> conv -> [B, 1, L] -> [B, L].
        h = self.conv(x.unsqueeze(1))
        return h.squeeze(1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
