"""Denoiser network: an MLP with sinusoidal time embedding and optional conditioning.

Signature:  denoiser(x_t, t, cond=None) -> pred   (same dim as x_t)

- ``x_t``: (B, data_dim) noisy sample.
- ``t``:   (B,) integer timestep in [0, T).
- ``cond``: (B, cond_dim) optional conditioning vector (None for unconditional).
- output:  (B, data_dim) predicted noise (epsilon) or clean sample (x0).

The time embedding is sinusoidal (Transformer-style) and projected to the hidden
width, then added at every layer.  Conditioning (if any) is concatenated to the
input.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..config import DenoiserConfig


# --------------------------------------------------------------------------- #
# Sinusoidal time embedding
# --------------------------------------------------------------------------- #
class SinusoidalTimeEmbedding(nn.Module):
    """Project integer timesteps into a sinusoidal embedding of dim ``emb_dim``."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        if emb_dim % 2 != 0:
            raise ValueError("emb_dim must be even for sinusoidal embedding")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )  # (half,)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, emb_dim)
        return emb


# --------------------------------------------------------------------------- #
# Denoiser MLP
# --------------------------------------------------------------------------- #
class MLPDenoiser(nn.Module):
    """Time-conditioned MLP denoiser with optional conditioning."""

    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        self.cfg = cfg
        self.cond_dim = cfg.cond_dim

        # Time embedding: sinusoidal -> MLP -> hidden width.
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.time_emb_dim),
            nn.Linear(cfg.time_emb_dim, cfg.hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dims[0], cfg.hidden_dims[0]),
        )

        # Input projection: data (+ optional cond) -> first hidden.
        in_dim = cfg.data_dim + (cfg.cond_dim if cfg.cond_dim > 0 else 0)
        self.input_proj = nn.Linear(in_dim, cfg.hidden_dims[0])

        # Hidden layers (with time embedding added before activation).
        self.hidden_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(cfg.hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(cfg.hidden_dims[i], cfg.hidden_dims[i + 1]))
            self.norms.append(nn.LayerNorm(cfg.hidden_dims[i + 1]))

        self.dropout = nn.Dropout(cfg.dropout)
        self.act = nn.SiLU()

        # Output projection back to data_dim.
        self.out_proj = nn.Linear(cfg.hidden_dims[-1], cfg.data_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Time embedding -> hidden width.
        temb = self.time_embed(t)  # (B, H0)

        # Input.
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond_dim > 0 but cond is None")
            inp = torch.cat([x_t, cond], dim=-1)
        else:
            inp = x_t
        h = self.input_proj(inp) + temb
        h = self.act(h)

        for layer, norm in zip(self.hidden_layers, self.norms):
            h = layer(h) + temb
            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)

        return self.out_proj(h)


def build_denoiser(cfg: DenoiserConfig, device: Optional[torch.device] = None) -> MLPDenoiser:
    net = MLPDenoiser(cfg)
    if device is not None:
        net = net.to(device)
    return net
