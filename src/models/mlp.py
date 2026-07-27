"""The target MLP architecture: a linear two-layer MLP R^L -> R^L.

This is the architecture whose weights the diffusion model learns to generate
(Phase 2, Approach B: train many MLPs to convergence, collect their weights as
diffusion targets).

Architecture (plan Section 2.2, confirmed for Phase 2):
    fc1: Linear(L, H)   # L=32, H=128
    fc2: Linear(H, L)
    # NO nonlinearity (linear MLP). This is a linear map R^L -> R^L.
    # Capacity: H*(L+1) + L*(H+1) = 128*33 + 32*129 = 4224 + 4128 = 8352 params.

Why linear (no activation):
- The ground-truth map y = T_k x is linear, so a linear MLP can represent it
  exactly. With n_train=1024 >> L=32 the per-output least-squares system is
  well-determined, so the MLP generalizes well (unlike the small-n_train
  Phase 1 case). This makes the converged weights good diffusion targets.
- The (W1, W2) factorization of the effective matrix W2 @ W1 is non-unique
  (gauge freedom), so different random initializations converge to *different*
  weight vectors that implement (approximately) the same map. This gives the
  diffusion model a rich target distribution to learn.
"""
import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """Over-parameterized linear MLP: Linear(L, H) -> Linear(H, L).

    No nonlinearity (v1 design, plan Section 2.2): the ground-truth map is
    linear, so a linear MLP can represent it exactly. With H=128 it has
    8352 parameters (~1000x the conv), so capacity is NOT the bottleneck;
    inductive bias is. Trained from scratch on few samples it overfits, but
    with n_train=1024 it generalizes well (Phase 2 target regime).

    The parameter ordering for flattening is fixed and documented in
    ``src/models/weight_codec.py``: fc1.weight, fc1.bias, fc2.weight, fc2.bias
    (C-order / row-major, matching PyTorch ``nn.Linear`` storage).
    """

    def __init__(self, L: int = 32, H: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(L, H)
        self.fc2 = nn.Linear(H, L)
        self.L = L
        self.H = H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] -> [B, H] -> [B, L]. Linear (no activation) per v1.
        return self.fc2(self.fc1(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
