"""Per-dimension weight standardization for diffusion targets.

Diffusion models assume data ``z_0 ~ N(0, 1)`` (roughly). Raw converged MLP
weight vectors ``theta in R^D`` (``D = 8352``) are not standardized: their
per-dimension mean and scale vary widely, and some dimensions are (near-)
constant (e.g. the identity/zeros blocks in the analytic Toeplitz target, or
gauge-frozen dimensions of the learned MLP targets).

This module computes per-dimension mean ``mu`` and std ``sigma`` over a corpus
of target weight vectors and provides the standardize / destandardize
transforms (plan Section 2.5)::

    z = (theta - mu) / sigma        # standardize  (diffusion operates on z)
    theta = sigma * z + mu          # destandardize (decode sampled z back to weights)

Constant dimensions (``std == 0``) are handled by setting ``sigma = 1`` and
``mu = <constant>`` so that ``z = 0`` there (the diffusion model learns these
are trivial, always-zero dimensions). This both avoids division by zero and
reduces the effective dimensionality the model must learn.

The statistics are computed once from the training corpus (Phase 4) and stored
alongside the model checkpoint. This module is a pure, device-agnostic
container (no learnable parameters) so it can be saved/loaded trivially.
"""
from typing import Dict, Optional

import torch

# A small floor to avoid division by zero for (near-)constant dimensions.
# Dimensions whose empirical std is below this are treated as constant.
STD_FLOOR: float = 1e-6


class WeightNormalizer:
    """Per-dimension standardization of weight vectors.

    Holds non-learnable statistics ``mu, sigma in R^D`` and transforms between
    raw weight space ``theta`` and standardized diffusion space ``z``.

    Args:
        D: Weight-space dimension (default 8352, the MLP param count).
        mu: Optional precomputed per-dim mean ``[D]``.
        sigma: Optional precomputed per-dim std ``[D]`` (constant dims set to 1).
        std_floor: Dimensions with empirical std below this are treated as
            constant (``sigma := 1``, ``mu := <constant>``).

    Attributes:
        D: The weight-space dimension.
        mu: ``[D]`` per-dim mean (float32).
        sigma: ``[D]`` per-dim std (float32; 1.0 for constant dims).
        constant_mask: ``[D]`` bool tensor, True where the dim is constant.
    """

    def __init__(self, D: int = 8352,
                 mu: Optional[torch.Tensor] = None,
                 sigma: Optional[torch.Tensor] = None,
                 std_floor: float = STD_FLOOR):
        self.D = int(D)
        self.std_floor = float(std_floor)
        if mu is not None:
            mu = mu.to(torch.float32).reshape(self.D)
        else:
            mu = torch.zeros(self.D, dtype=torch.float32)
        if sigma is not None:
            sigma = sigma.to(torch.float32).reshape(self.D)
        else:
            sigma = torch.ones(self.D, dtype=torch.float32)
        if mu.shape != (self.D,) or sigma.shape != (self.D,):
            raise ValueError(
                f"mu/sigma must have shape ({self.D},), got {tuple(mu.shape)} "
                f"and {tuple(sigma.shape)}")
        self.mu = mu
        self.sigma = sigma
        # A dim is "constant" iff its std is at/below the floor. When a
        # normalizer is constructed directly (not via fit()), we approximate
        # this from sigma: dims with sigma <= std_floor are treated as constant.
        # fit() overwrites this with the exact empirical mask.
        self.constant_mask = (self.sigma <= self.std_floor)

    # ------------------------------------------------------------------
    # Fit (compute statistics from a corpus)
    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, weights: torch.Tensor, std_floor: float = STD_FLOOR,
            eps: float = 1e-8) -> "WeightNormalizer":
        """Compute per-dim mean/std from a corpus of weight vectors.

        Args:
            weights: ``[..., D]`` float tensor of target weight vectors. The
                leading dimensions are flattened, so e.g. a Phase 2
                ``[n_datasets, n_mlp, D]`` tensor is handled directly (all
                leading axes are pooled together to estimate the per-dim
                statistics).
            std_floor: Dimensions with empirical std below this are treated
                as constant (``sigma := 1``).
            eps: Numerical floor added to std before the constant check is
                applied (avoids exact-zero std from finite samples).

        Returns:
            A fitted ``WeightNormalizer``.
        """
        w = weights.to(torch.float32).reshape(-1, weights.shape[-1])
        D = w.shape[-1]
        mu = w.mean(dim=0)
        std = w.std(dim=0, unbiased=False)
        constant = std < std_floor
        sigma = torch.where(constant, torch.ones_like(std), std + eps)
        # For constant dims, mu is already the constant value; sigma=1 -> z=0.
        norm = cls(D=D, mu=mu, sigma=sigma, std_floor=std_floor)
        norm.constant_mask = constant
        return norm

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None
           ) -> "WeightNormalizer":
        """Move the stored statistics to ``device`` (and optional ``dtype``)."""
        self.mu = self.mu.to(device=device, dtype=dtype or torch.float32)
        self.sigma = self.sigma.to(device=device, dtype=dtype or torch.float32)
        self.constant_mask = self.constant_mask.to(device=device)
        return self

    def standardize(self, theta: torch.Tensor) -> torch.Tensor:
        """Map raw weights ``theta`` to standardized diffusion space ``z``.

        Args:
            theta: ``[..., D]`` raw weight vectors.

        Returns:
            ``[..., D]`` standardized vectors ``z = (theta - mu) / sigma``.
        """
        mu = self.mu.to(theta.device, dtype=theta.dtype)
        sigma = self.sigma.to(theta.device, dtype=theta.dtype)
        return (theta - mu) / sigma

    def destandardize(self, z: torch.Tensor) -> torch.Tensor:
        """Map standardized ``z`` back to raw weight space ``theta``.

        Args:
            z: ``[..., D]`` standardized vectors.

        Returns:
            ``[..., D]`` raw weights ``theta = sigma * z + mu``.
        """
        mu = self.mu.to(z.device, dtype=z.dtype)
        sigma = self.sigma.to(z.device, dtype=z.dtype)
        return sigma * z + mu

    # ------------------------------------------------------------------
    # (De)serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a serializable state dict (mu, sigma, constant_mask)."""
        return {
            "D": torch.tensor(self.D),
            "mu": self.mu.detach().cpu(),
            "sigma": self.sigma.detach().cpu(),
            "constant_mask": self.constant_mask.detach().cpu(),
            "std_floor": torch.tensor(float(self.std_floor)),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "WeightNormalizer":
        """Reconstruct a ``WeightNormalizer`` from a ``state_dict``."""
        norm = cls(
            D=int(state["D"].item()),
            mu=state["mu"].to(torch.float32),
            sigma=state["sigma"].to(torch.float32),
            std_floor=float(state["std_floor"].item()),
        )
        norm.constant_mask = state["constant_mask"].to(torch.bool)
        return norm

    def __repr__(self) -> str:
        n_const = int(self.constant_mask.sum().item())
        return (f"WeightNormalizer(D={self.D}, "
                f"constant_dims={n_const}/{self.D}, "
                f"std_floor={self.std_floor})")
