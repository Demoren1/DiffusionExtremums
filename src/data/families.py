"""Kernel-generation functions for the 5 dataset families."""
from typing import Dict, Callable

import numpy as np

FAMILIES = ("MA", "DIFF", "GAUSS", "MATCH", "RAND")


def _normalize_sum_one(k: np.ndarray) -> np.ndarray:
    s = k.sum()
    if abs(s) < 1e-12:
        return k
    return k / s


def _normalize_l2(k: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(k)
    if n < 1e-12:
        return k
    return k / n


def _kernel_ma(radius: int, rng: np.random.Generator) -> np.ndarray:
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
    K = 2 * radius + 1
    js = np.arange(K, dtype=np.float32) - radius
    if radius == 1:
        choice = rng.integers(0, 2)
        if choice == 0:
            k = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        else:
            k = np.array([-1.0, 2.0, -1.0], dtype=np.float32)
    else:
        k = np.zeros(K, dtype=np.float32)
        diff = np.array([1.0, -1.0], dtype=np.float32)
        coeffs = np.array([1.0], dtype=np.float32)
        for _ in range(radius):
            coeffs = np.convolve(coeffs, diff)
        k[: coeffs.shape[0]] = coeffs
    scale = float(rng.uniform(0.5, 2.0))
    k = k * scale
    return _normalize_l2(k)


def _kernel_gauss(radius: int, rng: np.random.Generator) -> np.ndarray:
    K = 2 * radius + 1
    sigma_k = float(rng.uniform(0.5, 2.0))
    js = np.arange(K, dtype=np.float32) - radius
    k = np.exp(-(js ** 2) / (2.0 * sigma_k ** 2)).astype(np.float32)
    return _normalize_sum_one(k)


def _kernel_match(radius: int, rng: np.random.Generator) -> np.ndarray:
    K = 2 * radius + 1
    choice = rng.integers(0, 2)
    if choice == 0 and K >= 3:
        k = np.zeros(K, dtype=np.float32)
        k[radius - 1] = 1.0
        k[radius] = -2.0
        k[radius + 1] = 1.0
    else:
        k = np.zeros(K, dtype=np.float32)
        offsets = [o for o in range(-radius, radius + 1) if o != 0]
        off = int(rng.choice(offsets))
        k[radius + off] = 1.0
    return k.astype(np.float32)


def _kernel_rand(radius: int, rng: np.random.Generator) -> np.ndarray:
    K = 2 * radius + 1
    k = rng.standard_normal(K).astype(np.float32)
    if rng.random() < 0.5:
        k = _normalize_l2(k)
    return k


KERNEL_SAMPLERS: Dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "MA": _kernel_ma,
    "DIFF": _kernel_diff,
    "GAUSS": _kernel_gauss,
    "MATCH": _kernel_match,
    "RAND": _kernel_rand,
}


def sample_kernel(family: str, radius: int, rng: np.random.Generator) -> np.ndarray:
    if family not in KERNEL_SAMPLERS:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}")
    if radius not in (1, 2, 3):
        raise ValueError(f"radius must be in {{1,2,3}}, got {radius}")
    return KERNEL_SAMPLERS[family](radius, rng)
