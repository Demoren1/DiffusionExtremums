"""Kernel-generation functions for the 5 dataset families.

Each family defines a distribution over local, translation-equivariant FIR kernels.
See ``plans/plan.md`` Section 1.3 (the family table).

All functions return a 1D float32 numpy array of length ``K = 2*radius + 1``.
"""
from typing import Dict, Callable

import numpy as np

# Canonical family names.
FAMILIES = ("MA", "DIFF", "GAUSS", "MATCH", "RAND")


def _normalize_sum_one(k: np.ndarray) -> np.ndarray:
    """Scale ``k`` so its entries sum to 1 (for low-pass / averaging kernels)."""
    s = k.sum()
    if abs(s) < 1e-12:
        return k
    return k / s


def _normalize_l2(k: np.ndarray) -> np.ndarray:
    """Scale ``k`` to unit L2 norm (for derivative / general FIR kernels)."""
    n = np.linalg.norm(k)
    if n < 1e-12:
        return k
    return k / n


# ---------------------------------------------------------------------------
# Per-family kernel samplers: sample_kernel(radius, rng) -> np.ndarray[K]
# ---------------------------------------------------------------------------

def _kernel_ma(radius: int, rng: np.random.Generator) -> np.ndarray:
    """MA (moving average / low-pass): box or Gaussian-normalized kernel.

    With prob 0.5 use a uniform box (1/K); otherwise sample a Gaussian-shaped
    window and normalize to sum 1. Both are local smoothers; an MLP overfits
    the high-frequency noise that the box removes.
    """
    K = 2 * radius + 1
    if rng.random() < 0.5:
        k = np.ones(K, dtype=np.float32) / K
    else:
        sigma_k = float(rng.uniform(0.5, 2.0))
        js = np.arange(K, dtype=np.float32) - radius
        k = np.exp(-(js ** 2) / (2.0 * sigma_k ** 2)).astype(np.float32)
        k = _normalize_sum_one(k)
    return k


def _kernel_diff(radius: int, rng: np.random.Generator) -> np.ndarray:
    """DIFF (difference / edge detector): local derivative kernels.

    radius=1 -> first difference [-1, 0, 1] or Laplacian [-1, 2, -1];
    radius>=2 -> Sobel-like / higher-order differences. Scaled and L2-normalized.
    A position-specific MLP fails to generalize on shifted edges.
    """
    K = 2 * radius + 1
    js = np.arange(K, dtype=np.float32) - radius
    if radius == 1:
        choice = rng.integers(0, 2)
        if choice == 0:
            k = np.array([-1.0, 0.0, 1.0], dtype=np.float32)  # first difference
        else:
            k = np.array([-1.0, 2.0, -1.0], dtype=np.float32)  # Laplacian
    else:
        # Higher-order central difference: coefficients of the (radius)-th
        # finite-difference operator, placed symmetrically.
        k = np.zeros(K, dtype=np.float32)
        # first-difference coefficients via repeated convolution of [1,-1]
        diff = np.array([1.0, -1.0], dtype=np.float32)
        coeffs = np.array([1.0], dtype=np.float32)
        for _ in range(radius):
            coeffs = np.convolve(coeffs, diff)
        # coeffs has length radius+1; place them at the start of the kernel
        k[: coeffs.shape[0]] = coeffs
    scale = float(rng.uniform(0.5, 2.0))
    k = k * scale
    return _normalize_l2(k)


def _kernel_gauss(radius: int, rng: np.random.Generator) -> np.ndarray:
    """GAUSS (Gaussian blur): Gaussian window with varying width, sum-normalized.

    Local smoothing with varying width; the locality is the conv inductive bias.
    """
    K = 2 * radius + 1
    sigma_k = float(rng.uniform(0.5, 2.0))
    js = np.arange(K, dtype=np.float32) - radius
    k = np.exp(-(js ** 2) / (2.0 * sigma_k ** 2)).astype(np.float32)
    return _normalize_sum_one(k)


def _kernel_match(radius: int, rng: np.random.Generator) -> np.ndarray:
    """MATCH (local pattern matching): sparse motif kernels.

    Either a shift (delta at a random offset within the window) or a 2nd
    difference [1, -2, 1] (when radius>=1). Detects a specific local motif.
    """
    K = 2 * radius + 1
    choice = rng.integers(0, 2)
    if choice == 0 and K >= 3:
        # 2nd difference motif
        k = np.zeros(K, dtype=np.float32)
        k[radius - 1] = 1.0
        k[radius] = -2.0
        k[radius + 1] = 1.0
    else:
        # sparse shift: delta at a random offset (not the center, to be non-trivial)
        k = np.zeros(K, dtype=np.float32)
        offsets = [o for o in range(-radius, radius + 1) if o != 0]
        off = int(rng.choice(offsets))
        k[radius + off] = 1.0
    return k.astype(np.float32)


def _kernel_rand(radius: int, rng: np.random.Generator) -> np.ndarray:
    """RAND (random local FIR): k ~ N(0, I_K), optionally L2-normalized.

    A general local linear filter; tests meta-generalization across kernels.
    """
    K = 2 * radius + 1
    k = rng.standard_normal(K).astype(np.float32)
    if rng.random() < 0.5:
        k = _normalize_l2(k)
    return k


# Registry of family -> sampler.
KERNEL_SAMPLERS: Dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "MA": _kernel_ma,
    "DIFF": _kernel_diff,
    "GAUSS": _kernel_gauss,
    "MATCH": _kernel_match,
    "RAND": _kernel_rand,
}


def sample_kernel(family: str, radius: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a kernel for ``family`` with the given ``radius``.

    Args:
        family: One of MA, DIFF, GAUSS, MATCH, RAND.
        radius: Kernel radius r in {1, 2, 3}.
        rng: NumPy ``Generator`` for reproducibility.

    Returns:
        float32 array of length ``2*radius + 1``.
    """
    if family not in KERNEL_SAMPLERS:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}")
    if radius not in (1, 2, 3):
        raise ValueError(f"radius must be in {{1,2,3}}, got {radius}")
    return KERNEL_SAMPLERS[family](radius, rng)
