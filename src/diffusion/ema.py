"""Exponential Moving Average (EMA) of model parameters.

Maintains a shadow copy of a model's parameters updated as

    p_ema <- decay * p_ema + (1 - decay) * p_model

EMA weights typically produce higher-quality samples at inference time.
The shadow parameters live on the same device as the model.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average over the parameters of an ``nn.Module``.

    Usage::

        ema = EMA(model, decay=0.9999)
        for ...:
            loss.backward(); opt.step()
            ema.update(model, step)

        # swap in EMA weights for sampling, then restore:
        with ema.swap(model):
            samples = ddpm.sample(...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Shadow parameters — detached clones, no grad.
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int = 0, update_after: int = 0) -> None:
        """Update shadow params from ``model``.  No-op before ``update_after`` steps."""
        if step < update_after:
            return
        d = self.decay
        for name, p in model.named_parameters():
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy shadow params into ``model`` (overwriting current weights)."""
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name].to(p.device))

    class _SwapContext:
        """Context manager: store original params, copy EMA in, restore on exit."""

        def __init__(self, ema: "EMA", model: nn.Module):
            self.ema = ema
            self.model = model
            self.backup: dict[str, torch.Tensor] = {}

        def __enter__(self):
            for name, p in self.model.named_parameters():
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.ema.shadow[name].to(p.device))
            return self.model

        def __exit__(self, *exc):
            for name, p in self.model.named_parameters():
                p.data.copy_(self.backup[name].to(p.device))
            return False

    def swap(self, model: nn.Module) -> "EMA._SwapContext":
        """Return a context manager that temporarily loads EMA weights into ``model``."""
        return self._SwapContext(self, model)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        for name in self.shadow:
            if name in sd:
                self.shadow[name] = sd[name].to(self.shadow[name].device).clone()
