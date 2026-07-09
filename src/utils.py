"""Shared utilities: device selection, seeding, weight (un)flattening."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Device helper (cuda if available, else cpu)
# --------------------------------------------------------------------------- #
def get_device(pref: str = "auto") -> torch.device:
    """Return a torch device.

    pref: "auto" | "cuda" | "cpu".
    "auto" picks cuda when available, otherwise cpu.
    """
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Weight (un)flattening
# --------------------------------------------------------------------------- #
def flatten_params(model: nn.Module) -> torch.Tensor:
    """Return a 1-D tensor with all trainable parameters concatenated."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def unflatten_params(flat: torch.Tensor, model: nn.Module) -> None:
    """Load a 1-D tensor back into ``model`` in-place (matching parameter order)."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        # ``flat`` may live on a different device / require grad; copy data only.
        p.data.copy_(flat[idx:idx + n].reshape_as(p).to(p.device))
        idx += n


def param_shapes(model: nn.Module) -> List[Tuple[int, ...]]:
    """Return the shape of every parameter tensor (used for (un)flattening)."""
    return [tuple(p.shape) for p in model.parameters()]


def num_params(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())
