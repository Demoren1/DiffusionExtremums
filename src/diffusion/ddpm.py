"""DDPM core: noise schedules, forward process, and (ancestral) sampling.

Implements a standard Denoising Diffusion Probabilistic Model (Ho et al. 2020)
with linear or cosine beta schedule.  The module is **data-agnostic**: it operates
on arbitrary tensors ``x_0`` of shape (B, d) and a denoiser network that maps
``(x_t, t[, cond]) -> prediction``.

Device-agnostic: all buffers are moved to the device of the first forward call
via ``to(device)``; sampling works on both cuda and cpu.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import DDPMConfig


# --------------------------------------------------------------------------- #
# Beta schedules
# --------------------------------------------------------------------------- #
def linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0, 0.999)


# --------------------------------------------------------------------------- #
# DDPM
# --------------------------------------------------------------------------- #
class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model wrapper.

    The denoiser network is provided externally (see ``denoiser.py``) and is
    expected to have signature ``denoiser(x_t, t, cond=None) -> pred`` where
    ``pred`` is either the noise epsilon or the clean sample x0 (per ``objective``).
    """

    def __init__(self, denoiser: nn.Module, cfg: DDPMConfig):
        super().__init__()
        self.denoiser = denoiser
        self.cfg = cfg
        self.objective = cfg.objective
        self.variance_type = cfg.variance_type

        # Build schedule
        if cfg.schedule == "linear":
            betas = linear_beta_schedule(cfg.num_timesteps, cfg.beta_start, cfg.beta_end)
        elif cfg.schedule == "cosine":
            betas = cosine_beta_schedule(cfg.num_timesteps)
        else:
            raise ValueError(f"unknown schedule: {cfg.schedule}")

        betas = betas.clone()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register as buffers (float64 for numerical stability, cast at use).
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]]))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Posterior q(x_{t-1} | x_t, x_0) variance (fixed_small default).
        posterior_var = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_var.clamp_min(1e-20))
        self.register_buffer("posterior_log_variance", torch.log(self.posterior_variance))
        # Coefficients for posterior mean.
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------ #
    # Forward process  q(x_t | x_0)
    # ------------------------------------------------------------------ #
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t ~ q(x_t | x_0) = N(sqrt(a_t) x_0, (1-a_t) I).

        Returns (x_t, noise).
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sa = self._gather(self.sqrt_alphas_cumprod, t, x_0)
        s1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_0)
        x_t = sa * x_0 + s1ma * noise
        return x_t, noise

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #
    def training_loss(self, x_0: torch.Tensor,
                      cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the simplified DDPM loss for a batch x_0 (B, d)."""
        B = x_0.shape[0]
        t = torch.randint(0, self.cfg.num_timesteps, (B,), device=x_0.device)
        x_t, noise = self.q_sample(x_0, t)

        pred = self.denoiser(x_t, t, cond=cond)

        if self.objective == "epsilon":
            target = noise
        elif self.objective == "x0":
            target = x_0
        else:
            raise ValueError(f"unknown objective: {self.objective}")

        return nn.functional.mse_loss(pred, target)

    # ------------------------------------------------------------------ #
    # Sampling (ancestral / DDPM)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, shape: Tuple[int, int], device: torch.device,
              cond: Optional[torch.Tensor] = None,
              n_steps: Optional[int] = None,
              clip_denoised: bool = False) -> torch.Tensor:
        """Generate samples by reversing the diffusion process.

        shape: (B, d).  n_steps: if given and < T, uses a sub-sequence of timesteps
        (still ancestral).  If None, uses all T steps.
        """
        T = self.cfg.num_timesteps
        if n_steps is None or n_steps >= T:
            timesteps = list(range(T - 1, -1, -1))
        else:
            timesteps = list(torch.linspace(T - 1, 0, n_steps, dtype=torch.long).tolist())

        x = torch.randn(shape, device=device)
        for i, t in enumerate(timesteps):
            tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred = self.denoiser(x, tt, cond=cond)

            # Recover predicted x0 / epsilon depending on objective.
            if self.objective == "x0":
                x0_pred = pred
                eps = self._predict_eps_from_x0(x, tt, x0_pred)
            else:
                eps = pred
                x0_pred = self._predict_x0_from_eps(x, tt, eps)

            if clip_denoised:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Posterior mean  mu = coef1 * x0 + coef2 * x_t
            c1 = self._gather(self.posterior_mean_coef1, tt, x)
            c2 = self._gather(self.posterior_mean_coef2, tt, x)
            mean = c1 * x0_pred + c2 * x

            if t > 0:
                # Add stochastic noise (except at t=0).
                if self.variance_type == "fixed_large":
                    var = self._gather(self.betas, tt, x)
                    log_var = torch.log(var.clamp_min(1e-20))
                else:  # fixed_small
                    log_var = self._gather(self.posterior_log_variance, tt, x)
                noise = torch.randn_like(x)
                x = mean + torch.exp(0.5 * log_var) * noise
            else:
                x = mean
        return x

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sa = self._gather(self.sqrt_alphas_cumprod, t, x_t)
        s1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        return (x_t - s1ma * eps) / sa.clamp_min(1e-8)

    def _predict_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sa = self._gather(self.sqrt_alphas_cumprod, t, x_t)
        s1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        return (x_t - sa * x0) / s1ma.clamp_min(1e-8)

    @staticmethod
    def _gather(src: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Index ``src`` at ``t`` and reshape to broadcast over ``ref``.

        ``src`` is a float64 buffer; the result is cast to ``ref``'s dtype/device
        and reshaped to (B, 1, 1, ...) for broadcasting.
        """
        out = src.to(ref.device).gather(0, t.to(ref.device))
        out = out.to(ref.dtype)
        return out.view(-1, *([1] * (ref.dim() - 1)))
