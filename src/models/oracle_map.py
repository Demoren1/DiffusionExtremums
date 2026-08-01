"""Exact oracle effective map from a known convolution kernel.

This module constructs the **exact** effective linear map that a 1D
'same'-padding zero-padded convolution with kernel ``k`` (length ``K = 2r+1``)
implements, expressed in the **MLP convention** used throughout the project:

    y = M @ x + b_eff

where ``M`` is the ``[L, L]`` effective matrix and ``b_eff = 0`` (the
convolution has no bias). The flat effective-map vector is
``[M.flatten() ; b_eff]`` of length ``L*L + L = 1056`` (for ``L = 32``),
matching the layout in :mod:`src.models.effective_map`.

Convention derivation
---------------------
The repository's :func:`src.data.dataset.conv1d_same` computes, for output
position ``p`` and input ``x`` of length ``L`` (zero-padded):

    y[p] = sum_{j=0}^{K-1} k[j] * x_pad[p + j - r]
         = sum_{q=0}^{L-1} k[q - p + r] * x[q]      (where 0 <= q-p+r < K)

The MLP computes ``y = M @ x``, i.e. ``y[p] = sum_q M[p, q] * x[q]``. Matching
the two gives the **MLP-convention** effective matrix:

    M[p, q] = k[q - p + r]      when 0 <= q - p + r < K, else 0.

This is the **transpose** of the convolution-operator matrix
(``M_conv[p, q] = k[p - q + r]``), because the MLP applies ``M @ x`` (column
vector convention) while the convolution operator is naturally written as
``y = x @ M_conv`` (row vector convention). The two are related by
``M = M_conv.T``.

The existing :func:`src.models.effective_map.kernel_to_effective_map` builds
``M_conv`` (the row-vector / ``x @ M`` convention). For oracle targets that
must round-trip through the MLP factorization (``effective_map_to_weights`` ->
``instantiate_mlp`` -> ``y = M @ x``), we need the MLP convention here.

Validation
----------
:func:`kernel_to_oracle_effective_map` is validated against
:func:`conv1d_same` on random batched inputs to tight tolerance (see the smoke
test in :mod:`src.smoke.test_oracle_map` and the self-check at the bottom of
this module).
"""
from typing import Tuple

import numpy as np
import torch

from src.models.effective_map import DEFAULT_EFF_D, DEFAULT_L

__all__ = [
    "kernel_to_oracle_effective_map",
    "kernel_to_oracle_matrix",
    "oracle_effective_map_matches_conv1d_same",
]


def kernel_to_oracle_matrix(kernel: torch.Tensor, L: int = DEFAULT_L) -> torch.Tensor:
    """Build the exact oracle effective matrix ``M`` [L, L] (MLP convention).

    ``M[p, q] = k[q - p + r]`` when ``0 <= q - p + r < K``, else 0. This is the
    matrix such that ``M @ x == conv1d_same(x, k)`` for all ``x``.

    Args:
        kernel: ``[K]`` ground-truth FIR filter (K = 2*radius + 1, odd).
        L: Input/output length (default 32).

    Returns:
        ``[L, L]`` float32 matrix ``M`` (MLP convention, ``y = M @ x``).
    """
    k = kernel.reshape(-1).to(torch.float32)
    K = k.shape[0]
    if K % 2 == 0:
        raise ValueError(f"kernel size K must be odd, got {K}")
    r = K // 2
    M = torch.zeros(L, L, dtype=torch.float32)
    # Vectorized Toeplitz construction. For each kernel tap j (offset d = j - r
    # in [-r, r]), M[p, p + d] = k[j] for valid p. Equivalently M[p, q] = k[q-p+r].
    for j in range(K):
        d = j - r  # diagonal offset: q - p = d
        # M[p, p+d] = k[j] for p in [max(0,-d), min(L, L-d)).
        if d >= 0:
            p_start, p_end = 0, L - d
        else:
            p_start, p_end = -d, L
        if p_end > p_start:
            idx = torch.arange(p_start, p_end)
            M[idx, idx + d] = k[j]
    return M


def kernel_to_oracle_effective_map(
    kernel: torch.Tensor,
    L: int = DEFAULT_L,
) -> torch.Tensor:
    """Build the exact oracle effective map vector ``[L*L + L]`` (1056 for L=32).

    The effective map is ``[M.flatten() ; b_eff]`` where ``M`` is the
    MLP-convention oracle matrix (:func:`kernel_to_oracle_matrix`) and
    ``b_eff = 0`` (the convolution has no bias). The flat layout matches
    :mod:`src.models.effective_map` (row-major ``M.flatten()`` then ``b_eff``).

    A map produced here, when factorized via
    :func:`src.models.effective_map.effective_map_to_weights` and instantiated
    as an MLP, computes exactly ``conv1d_same(x, kernel)`` (to numerical
    precision).

    Args:
        kernel: ``[K]`` ground-truth FIR filter (K = 2*radius + 1, odd).
        L: Input/output length (default 32).

    Returns:
        ``[D_eff]`` float32 effective map vector (1056 for L=32).
    """
    M = kernel_to_oracle_matrix(kernel, L=L)
    b_eff = torch.zeros(L, dtype=torch.float32)
    return torch.cat([M.reshape(-1), b_eff], dim=0)


def oracle_effective_map_matches_conv1d_same(
    kernel: torch.Tensor,
    L: int = DEFAULT_L,
    n_batches: int = 8,
    seed: int = 0,
    atol: float = 1e-5,
) -> Tuple[float, bool]:
    """Validate the oracle map against :func:`conv1d_same` on random inputs.

    Builds the oracle effective map, factorizes it to an MLP via SVD, and
    compares the MLP output to :func:`conv1d_same` on random batched inputs.

    Args:
        kernel: ``[K]`` ground-truth FIR filter.
        L: Input/output length.
        n_batches: Number of random input rows to test.
        seed: RNG seed.
        atol: Absolute tolerance for the max-error check.

    Returns:
        ``(max_err, ok)`` where ``max_err`` is the max abs difference between
        the MLP output and ``conv1d_same``, and ``ok`` is True iff
        ``max_err < atol``.
    """
    # Local import to avoid a circular import at module load time.
    from src.data.dataset import conv1d_same
    from src.models.effective_map import instantiate_mlp_from_eff_map

    k_np = kernel.reshape(-1).to(torch.float32).numpy()
    eff = kernel_to_oracle_effective_map(kernel, L=L)
    mlp = instantiate_mlp_from_eff_map(eff, L=L, H=128)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_batches, L)).astype(np.float32)
    y_conv = conv1d_same(x, k_np)
    with torch.no_grad():
        y_mlp = mlp(torch.from_numpy(x)).numpy()
    max_err = float(np.abs(y_conv - y_mlp).max())
    return max_err, bool(max_err < atol)


# ---------------------------------------------------------------------------
# Self-check on import (cheap; catches convention regressions early).
# ---------------------------------------------------------------------------
def _self_check() -> None:
    """Internal sanity check: oracle map matches conv1d_same for K in {3,5,7}.

    Uses a fixed RNG seed for the test kernels so the check is deterministic
    across runs (the SVD factorization is float32, so an unseeded kernel can
    occasionally land just above the 1e-5 tolerance).
    """
    gen = torch.Generator().manual_seed(12345)
    for K in (3, 5, 7):
        k = torch.randn(K, generator=gen)
        max_err, ok = oracle_effective_map_matches_conv1d_same(k, seed=K)
        if not ok:
            raise RuntimeError(
                f"oracle map self-check FAILED for K={K}: max_err={max_err:.3e} "
                f"(>= atol). The kernel->effective-map convention is wrong.")


# Run the self-check at import time. It is cheap (3 small SVDs + matmuls) and
# guards against silent convention regressions.
_self_check()