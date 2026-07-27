"""DDPM diffusion hypernetwork over MLP weight space (plan Section 3).

A **denoising diffusion probabilistic model (DDPM)** operating over the
8352-dimensional weight space, conditioned on the dataset configuration. The
core hypothesis of the project is that a **small** diffusion model can act as a
meta-learner / hypernetwork: conditioned on a dataset config, it generates the
weights of an MLP that implements a convolution-like (local,
translation-equivariant) map. The diffusion model is deliberately compact
(~4.6M params, plan Section 3.4) — small relative to the space it models and
shared across the whole corpus.

This module implements:

1. **Noise schedule** (``NoiseSchedule``): linear ``beta_t in [1e-4, 0.02]``,
   ``T = 1000`` timesteps, with the closed-form ``alpha_t``, ``bar_alpha_t``,
   and the forward-process coefficients ``sqrt(bar_alpha_t)`` and
   ``sqrt(1 - bar_alpha_t)``.
2. **Timestep embedding** (``SinusoidalTimeEmbedding``): standard sinusoidal
   positional embedding -> MLP -> per-layer modulation signal.
3. **Denoiser network** (``DenoiserNet``): a compact MLP
   ``8352 -> hidden -> ... -> 8352`` with **FiLM-style** conditioning: the time
   embedding and config embedding are projected to per-layer (scale, shift)
   modulation factors applied to each hidden layer.
4. **DiffusionModel**: the full DDPM. ``forward()`` computes the training loss
   (simple noise-prediction MSE); ``sample()`` runs the reverse-process
   ancestral sampling loop to generate weights from noise given a condition.

Forward process (plan Section 3.2)::

    z_t = sqrt(bar_alpha_t) * z_0 + sqrt(1 - bar_alpha_t) * eps,  eps ~ N(0, I_D)

Reverse process (DDPM ancestral sampling, plan Section 3.6)::

    z_{t-1} = (1/sqrt(alpha_t)) * (z_t - (beta_t / sqrt(1 - bar_alpha_t)) * eps_phi(z_t, t, c))
              + sigma_t * z,   z ~ N(0, I) if t > 0 else 0
    sigma_t = sqrt(beta_t)            (eta=1, standard DDPM)

The model operates on **standardized** weight vectors ``z`` (see
``src/models/weight_norm.py``); the caller is responsible for standardizing
targets before training and destandardizing samples after generation.
"""
import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs.base import DatasetConfig
from src.models.config_encoder import (
    CONFIG_FEATURE_DIM,
    ConfigEncoder,
    configs_to_features,
)

# Default diffusion data dimension (the MLP weight vector length).
DEFAULT_D: int = 8352


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class NoiseSchedule:
    """Linear DDPM noise schedule (non-learnable; precomputed buffers).

    Linear ``beta_t`` interpolation from ``beta_start`` to ``beta_end`` over
    ``T`` timesteps (plan Section 3.2). Provides the closed-form forward-process
    coefficients and the reverse-process posterior variance.

    Args:
        num_timesteps: Number of diffusion steps ``T`` (default 1000).
        beta_start: Initial beta (default 1e-4).
        beta_end: Final beta (default 0.02).
        schedule: ``"linear"`` (only supported schedule in v1).

    Attributes:
        T: Number of timesteps.
        betas: ``[T]`` beta_t.
        alphas: ``[T]`` alpha_t = 1 - beta_t.
        alphas_cumprod: ``[T]`` bar_alpha_t = prod_{s<=t} alpha_s.
        sqrt_alphas_cumprod: ``[T]`` sqrt(bar_alpha_t).
        sqrt_one_minus_alphas_cumprod: ``[T]`` sqrt(1 - bar_alpha_t).
        posterior_variance: ``[T]`` reverse-process noise variance
            (``beta_t`` for standard DDPM eta=1; index 0 unused).
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, schedule: str = "linear"):
        if schedule != "linear":
            raise ValueError(f"unsupported schedule {schedule!r}; use 'linear'")
        self.T = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.schedule = schedule

        betas = torch.linspace(beta_start, beta_end, self.T,
                               dtype=torch.float64)  # high precision for cumprod
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
        # Posterior variance for ancestral sampling (standard DDPM, eta=1):
        #   q(x_{t-1}|x_t,x_0) variance = beta_tilde_t =
        #       beta_t * (1 - bar_alpha_{t-1}) / (1 - bar_alpha_t).
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Clamp the t=0 entry (undefined; not used in sampling) to beta_0.
        posterior_variance[0] = betas[0]
        self.posterior_variance = posterior_variance.float()
        self.posterior_log_variance = torch.log(posterior_variance.clamp_min(1e-20)).float()

    def to(self, device: torch.device) -> "NoiseSchedule":
        """Move all buffers to ``device``."""
        for name in ("betas", "alphas", "alphas_cumprod", "sqrt_alphas_cumprod",
                     "sqrt_one_minus_alphas_cumprod", "posterior_variance",
                     "posterior_log_variance"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def device(self) -> torch.device:
        return self.betas.device


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding of the timestep -> MLP -> embedding.

    Standard transformer-style sinusoidal embedding (plan Section 3.5)::

        te(t)_{2i}   = sin(t * 10000^{-2i/d})
        te(t)_{2i+1} = cos(t * 10000^{-2i/d})

    followed by a 2-layer MLP (Linear -> SiLU -> Linear) to ``embed_dim``.

    Args:
        embed_dim: Output embedding dimension (default 128).
        max_period: Frequency base (default 10000).
    """

    def __init__(self, embed_dim: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_period = float(max_period)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed integer timesteps.

        Args:
            t: ``[B]`` integer/long timesteps in ``[0, T)`` (or any non-negative
                ints; the embedding is continuous in the value).

        Returns:
            ``[B, embed_dim]`` time embedding.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.float()
        device = t.device
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=device) / half)
        args = t[:, None].float() * freqs[None, :]  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]
        if self.embed_dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# FiLM modulation block
# ---------------------------------------------------------------------------

class FiLMBlock(nn.Module):
    """A residual MLP block with FiLM (scale + shift) conditioning.

    Computes ``h = SiLU(LN_or_none(Linear(h)) * (1 + gamma) + beta + residual)``
    where ``(gamma, beta)`` are per-layer modulation vectors produced from the
    time and config embeddings. Following the plan (Section 3.4) we use
    **additive FiLM** (shift-only) as the v1 default for simplicity, but this
    block supports full scale+shift FiLM via ``use_scale``.

    Args:
        dim: Block width (input == output).
        cond_dim: Dimension of the combined time+config conditioning vector.
        use_scale: If True, produce a per-layer scale (gamma) in addition to
            shift (beta). Default False (additive FiLM, per plan v1).
    """

    def __init__(self, dim: int, cond_dim: int, use_scale: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.cond_dim = int(cond_dim)
        self.use_scale = bool(use_scale)
        self.fc = nn.Linear(dim, dim)
        # cond -> (gamma, beta) modulation. Two outputs: scale and shift.
        out = 2 if self.use_scale else 1
        self.cond_proj = nn.Linear(cond_dim, out * dim)
        # Zero-init the modulation projection so the block starts as a plain
        # residual MLP (stable training start).
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply the block: linear + FiLM modulation + SiLU + residual.

        Args:
            h: ``[B, dim]`` hidden activations.
            cond: ``[B, cond_dim]`` combined time+config conditioning.

        Returns:
            ``[B, dim]`` updated activations.
        """
        out = self.fc(h)
        mod = self.cond_proj(cond)
        if self.use_scale:
            gamma, beta = mod.chunk(2, dim=-1)
            out = out * (1.0 + gamma) + beta
        else:
            beta = mod
            out = out + beta
        out = F.silu(out)
        return out + h  # residual


# ---------------------------------------------------------------------------
# Denoiser network
# ---------------------------------------------------------------------------

class DenoiserNet(nn.Module):
    """Compact MLP denoiser with FiLM conditioning (plan Section 3.4).

    Takes ``(z_t, t, c)`` and predicts the noise ``eps_hat in R^D``.

    Architecture::

        fc_in:  Linear(D, hidden)
        blocks: N x FiLMBlock(hidden, cond_dim=2*embed_dim)
        fc_out: Linear(hidden, D)

    The time embedding (``SinusoidalTimeEmbedding``) and config embedding
    (``ConfigEncoder``) are each projected to ``embed_dim`` and concatenated
    into a ``2*embed_dim`` conditioning vector that drives the FiLM blocks.

    Args:
        D: Data (weight-space) dimension (default 8352).
        hidden: Hidden width (default 256; the plan's ~4.6M-param config).
        n_blocks: Number of FiLM blocks (default 3).
        embed_dim: Time/config embedding dimension (default 128).
        feature_dim: Config feature dimension (default ``CONFIG_FEATURE_DIM``=14).
        cond_hidden: Config encoder hidden width (default 64).
        use_scale: Use full scale+shift FiLM (default False = additive, plan v1).

    The config encoder is owned by the denoiser so the whole model is a single
    module with one parameter set / optimizer.
    """

    def __init__(self, D: int = DEFAULT_D, hidden: int = 256, n_blocks: int = 3,
                 embed_dim: int = 128, feature_dim: int = CONFIG_FEATURE_DIM,
                 cond_hidden: int = 64, use_scale: bool = False):
        super().__init__()
        self.D = int(D)
        self.hidden = int(hidden)
        self.n_blocks = int(n_blocks)
        self.embed_dim = int(embed_dim)
        self.feature_dim = int(feature_dim)
        self.use_scale = bool(use_scale)

        # Conditioning: time embedding + config encoder.
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=embed_dim)
        self.config_encoder = ConfigEncoder(
            feature_dim=feature_dim, hidden_dim=cond_hidden, embed_dim=embed_dim)
        cond_dim = 2 * embed_dim  # [time_emb ; config_emb]

        # Main trunk.
        self.fc_in = nn.Linear(D, hidden)
        self.blocks = nn.ModuleList(
            [FiLMBlock(hidden, cond_dim, use_scale=use_scale)
             for _ in range(n_blocks)])
        self.fc_out = nn.Linear(hidden, D)
        # Zero-init the output projection so the denoiser starts predicting
        # ~zero noise (identity-ish reverse process at init -> stable start).
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                cond_emb: torch.Tensor) -> torch.Tensor:
        """Predict the noise added to ``z_t`` at timestep ``t``.

        Args:
            z_t: ``[B, D]`` noised standardized weights.
            t: ``[B]`` integer timesteps in ``[0, T)``.
            cond_emb: ``[B, embed_dim]`` config embedding (from
                ``config_encoder``). Passing the precomputed embedding lets the
                training loop compute it once per batch and reuse it.

        Returns:
            ``[B, D]`` predicted noise ``eps_hat``.
        """
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        B = z_t.shape[0]
        # Time embedding from the raw timestep.
        t_emb = self.time_embed(t)                 # [B, embed_dim]
        # cond_emb is already [B, embed_dim]; concat -> [B, 2*embed_dim].
        if cond_emb.dim() == 1:
            cond_emb = cond_emb.unsqueeze(0)
        cond = torch.cat([t_emb, cond_emb], dim=-1)  # [B, 2*embed_dim]

        h = self.fc_in(z_t)
        for block in self.blocks:
            h = block(h, cond)
        out = self.fc_out(h)
        return out

    # ------------------------------------------------------------------
    # Convenience: encode configs internally
    # ------------------------------------------------------------------
    def encode_configs(self, configs: Sequence[Union[DatasetConfig, Dict]]
                       ) -> torch.Tensor:
        """Compute config embeddings (delegated to the owned ``ConfigEncoder``)."""
        return self.config_encoder.encode_configs(configs)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Full DDPM
# ---------------------------------------------------------------------------

class DiffusionModel(nn.Module):
    """DDPM over MLP weight space, conditioned on the dataset config.

    Wraps the noise schedule, denoiser, and (optional) weight normalizer into a
    single module. ``forward()`` computes the training loss; ``sample()``
    generates weights via ancestral sampling.

    Args:
        D: Weight-space dimension (default 8352).
        num_timesteps: Diffusion steps ``T`` (default 1000).
        beta_start, beta_end: Linear schedule endpoints (default 1e-4, 0.02).
        hidden: Denoiser hidden width (default 256 -> ~4.6M params).
        n_blocks: Number of FiLM blocks (default 3).
        embed_dim: Time/config embedding dim (default 128).
        use_scale: Full scale+shift FiLM (default False, additive per plan v1).
        normalizer: Optional fitted ``WeightNormalizer`` for standardization.
            If provided, ``forward`` accepts *raw* weights and standardizes
            internally, and ``sample`` returns *raw* weights (destandardized).
            If None, the caller handles standardization (the smoke test uses
            this path with already-standardized random tensors).

    Attributes:
        D, T: Dimensions.
        schedule: ``NoiseSchedule``.
        denoiser: ``DenoiserNet`` (owns the config encoder).
        normalizer: The ``WeightNormalizer`` (or None).
    """

    def __init__(self, D: int = DEFAULT_D, num_timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 hidden: int = 256, n_blocks: int = 3, embed_dim: int = 128,
                 use_scale: bool = False,
                 normalizer: Optional["object"] = None):
        super().__init__()
        self.D = int(D)
        self.T = int(num_timesteps)
        self.schedule = NoiseSchedule(num_timesteps=num_timesteps,
                                      beta_start=beta_start, beta_end=beta_end)
        self.denoiser = DenoiserNet(
            D=D, hidden=hidden, n_blocks=n_blocks, embed_dim=embed_dim,
            use_scale=use_scale)
        self.normalizer = normalizer  # may be None

    # ------------------------------------------------------------------
    # Device handling for the (non-nn.Module) schedule
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):  # type: ignore[override]
        """Move the module AND the noise-schedule buffers to the device."""
        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.schedule.to(device)
        if self.normalizer is not None:
            self.normalizer.to(device)
        return self

    def _ensure_schedule_device(self) -> None:
        """Ensure the noise-schedule buffers live on the model's device."""
        device = next(self.parameters()).device
        if self.schedule.betas.device != device:
            self.schedule.to(device)

    # ------------------------------------------------------------------
    # Forward (noisy) process: q(z_t | z_0)
    # ------------------------------------------------------------------
    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-form forward process: sample ``z_t`` from ``z_0``.

        ``z_t = sqrt(bar_alpha_t) * z_0 + sqrt(1 - bar_alpha_t) * eps``.

        Args:
            z_0: ``[B, D]`` clean (standardized) weights.
            t: ``[B]`` integer timesteps in ``[0, T)``.
            noise: Optional ``[B, D]`` noise; if None, sampled ~ N(0, I).

        Returns:
            ``(z_t, eps)`` where ``z_t`` is ``[B, D]`` and ``eps`` is the noise
            used (``[B, D]``).
        """
        self._ensure_schedule_device()
        if noise is None:
            noise = torch.randn_like(z_0)
        sqrt_bar = self.schedule.sqrt_alphas_cumprod.to(z_0.device, z_0.dtype)
        sqrt_om = self.schedule.sqrt_one_minus_alphas_cumprod.to(z_0.device, z_0.dtype)
        s1 = sqrt_bar[t].view(-1, 1)
        s2 = sqrt_om[t].view(-1, 1)
        z_t = s1 * z_0 + s2 * noise
        return z_t, noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def forward(self, z_0: torch.Tensor, t: torch.Tensor,
                cond_emb: torch.Tensor,
                noise: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Compute the DDPM training loss (simple noise-prediction MSE).

        ``L = E_{t, z_0, eps} || eps - eps_phi(z_t, t, cond) ||_2^2`` (plan 3.2/4.2),
        averaged over the batch and the D dimensions.

        Args:
            z_0: ``[B, D]`` clean weights. If a ``normalizer`` is attached, these
                are treated as **raw** weights and standardized internally;
                otherwise they are assumed already standardized.
            t: ``[B]`` integer timesteps in ``[0, T)``.
            cond_emb: ``[B, embed_dim]`` precomputed config embeddings.
            noise: Optional fixed noise ``[B, D]`` (for reproducibility/tests).

        Returns:
            Scalar loss tensor (mean MSE).
        """
        if self.normalizer is not None:
            z_0 = self.normalizer.standardize(z_0)
        z_t, eps = self.q_sample(z_0, t, noise=noise)
        eps_hat = self.denoiser(z_t, t, cond_emb)
        loss = F.mse_loss(eps_hat, eps)
        return loss

    # ------------------------------------------------------------------
    # Reverse (sampling) process: ancestral DDPM
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, cond_emb: torch.Tensor, n_samples: Optional[int] = None,
               device: Optional[torch.device] = None,
               clip_denoised: bool = False,
               return_trajectory: bool = False) -> torch.Tensor:
        """Generate weights from noise via DDPM ancestral sampling.

        Standard reverse process (plan Section 3.6), iterating ``t = T-1 ... 0``::

            z_{t-1} = (1/sqrt(alpha_t)) * (z_t - (beta_t/sqrt(1-bar_alpha_t)) * eps_phi)
                      + sigma_t * z,   z ~ N(0, I) if t > 0 else 0

        Args:
            cond_emb: ``[B, embed_dim]`` config embeddings to condition on.
            n_samples: Number of samples if ``cond_emb`` is a single 1-D vector.
                Ignored if ``cond_emb`` is already batched (2-D).
            device: Device to sample on (defaults to the model's device).
            clip_denoised: If True, clip ``z_t`` to ``[-1, 1]`` at each step
                (standardized space). Default False (the target manifold may
                legitimately exceed unit range).
            return_trajectory: If True, return the full ``[T+1, B, D]`` trajectory
                (for visualization); else return only the final ``[B, D]``.

        Returns:
            Generated weights ``[B, D]``. If a ``normalizer`` is attached, the
            output is **destandardized** to raw weight space; otherwise it is
            in standardized space.
        """
        self._ensure_schedule_device()
        if device is None:
            device = next(self.parameters()).device
        if cond_emb.dim() == 1:
            cond_emb = cond_emb.unsqueeze(0)
        if n_samples is not None and cond_emb.shape[0] == 1:
            cond_emb = cond_emb.expand(n_samples, -1)
        B = cond_emb.shape[0]
        cond_emb = cond_emb.to(device)

        # Buffers on the sampling device/dtype.
        betas = self.schedule.betas.to(device)
        alphas = self.schedule.alphas.to(device)
        sqrt_alphas_cumprod = self.schedule.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod.to(device)
        posterior_var = self.schedule.posterior_variance.to(device)
        posterior_log_var = self.schedule.posterior_log_variance.to(device)

        z = torch.randn(B, self.D, device=device)
        traj = [z] if return_trajectory else None

        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps_hat = self.denoiser(z, t, cond_emb)
            # Coefficients.
            b_t = betas[i]
            a_t = alphas[i]
            sqrt_a = math.sqrt(1.0 / float(a_t.item()))
            coef = b_t / sqrt_one_minus[i]
            mean = sqrt_a * (z - coef * eps_hat)
            if clip_denoised:
                mean = mean.clamp(-1.0, 1.0)
            if i > 0:
                # Standard DDPM (eta=1): sigma_t = sqrt(posterior_variance_t).
                # Use the log-variance for numerical stability.
                noise = torch.randn_like(z)
                sigma = torch.sqrt(posterior_var[i])
                z = mean + sigma * noise
            else:
                z = mean
            if return_trajectory:
                traj.append(z)

        if self.normalizer is not None:
            z = self.normalizer.destandardize(z)
        if return_trajectory:
            traj = torch.stack(traj, dim=0)  # [T+1, B, D]
            if self.normalizer is not None:
                traj = self.normalizer.destandardize(traj)
            return traj
        return z

    # ------------------------------------------------------------------
    # High-level: condition on configs and sample
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_configs(self, configs: Sequence[Union[DatasetConfig, Dict]],
                       device: Optional[torch.device] = None,
                       clip_denoised: bool = False) -> torch.Tensor:
        """Generate raw weight vectors for a batch of dataset configs.

        Convenience wrapper: encodes the configs, runs ancestral sampling, and
        destandardizes (if a normalizer is attached) to raw weight space.

        Args:
            configs: Sequence of ``DatasetConfig`` or dicts.
            device: Sampling device (defaults to the model's device).
            clip_denoised: Clip standardized samples to ``[-1, 1]``.

        Returns:
            ``[B, D]`` generated (raw, destandardized) weight vectors.
        """
        if device is None:
            device = next(self.parameters()).device
        cond_emb = self.denoiser.encode_configs(configs).to(device)
        return self.sample(cond_emb, device=device, clip_denoised=clip_denoised)

    # ------------------------------------------------------------------
    # Parameter accounting
    # ------------------------------------------------------------------
    def n_params(self) -> int:
        """Total learnable parameter count of the diffusion model."""
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> Dict[str, int]:
        """Per-submodule parameter counts (for documentation/debugging)."""
        breakdown: Dict[str, int] = {}
        for name, mod in self.named_modules():
            if name == "":
                continue
            if "." in name:
                continue  # only top-level submodules
            n = sum(p.numel() for p in mod.parameters(recurse=True))
            if n > 0:
                breakdown[name] = n
        breakdown["total"] = self.n_params()
        return breakdown
