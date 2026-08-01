"""Effective linear-map codec for the target MLP (Strategy B).

Because the MLP is **linear** (no activation), the 50 MLPs per dataset that
compute the same function have weights spread across a large **gauge-freedom
manifold** (~7264 dims): infinitely many ``(W1, W2)`` factorizations give the
same effective map.

Strategy B removes this confound by representing only the **effective linear
map**, which has no gauge freedom:

    y = W2 @ (W1 @ x + b1) + b2 = (W2 @ W1) @ x + (W2 @ b1 + b2) = M @ x + b_eff

- **Effective map**     M = W2 @ W1, shape [L, L] = [32, 32]  -> 1024 params
- **Effective bias**    b_eff = W2 @ b1 + b2, shape [L] = [32] -> 32 params
- **Total effective dimension: 1056** (no gauge freedom)

All 50 MLPs per dataset compute the same function -> the same ``(M, b_eff)`` ->
the target distribution per config is a tight cluster (low entropy), much
easier to model.

Effective-map layout (fixed, documented)::

    eff_map = concat([
        flatten(M),     # L*L = 32*32 = 1024  (row-major, M[i,j] at index i*L+j)
        flatten(b_eff), # L    = 32
    ])
    # D_eff = 1024 + 32 = 1056

SVD factorization (M -> W1, W2) to instantiate an MLP from (M, b_eff)::

    SVD: M = U @ diag(S) @ V^T   (U [L,L], S [L], V [L,L])
    W1 [H, L]: first L rows = diag(sqrt(S)) @ V^T, remaining (H-L) rows = 0
    W2 [L, H]: first L cols = U @ diag(sqrt(S)), remaining (H-L) cols = 0
    b1 [H] = 0, b2 [L] = b_eff

    Verify: W2 @ W1 = U @ diag(sqrt(S)) @ diag(sqrt(S)) @ V^T
                       = U @ diag(S) @ V^T = M  ✓

The SVD factorization is numerically stable: near-zero singular values are
clamped to a small positive floor so ``sqrt(S)`` is well-defined.
"""
import torch

from src.models.weight_codec import WeightCodec, instantiate_mlp

# Default MLP dimensions (must match src/models/mlp.py and weight_codec.py).
DEFAULT_L: int = 32
DEFAULT_H: int = 128
# Effective-map dimension: M_flat (L*L) + b_eff (L) = 1024 + 32 = 1056.
DEFAULT_EFF_D: int = DEFAULT_L * DEFAULT_L + DEFAULT_L  # 1056

# Floor for singular values to keep sqrt(S) numerically stable.
_SVD_S_FLOOR: float = 1e-10


def weights_to_effective_map(weights: torch.Tensor, L: int = DEFAULT_L,
                             H: int = DEFAULT_H) -> torch.Tensor:
    """Convert full MLP weight vectors to the effective linear map.

    Unpacks each weight vector via ``WeightCodec``, computes the effective map
    ``M = fc2.weight @ fc1.weight`` [L, L] and effective bias
    ``b_eff = fc2.weight @ fc1.bias + fc2.bias`` [L], and concatenates them
    into a flat ``[L*L + L]`` vector.

    Args:
        weights: ``[..., D]`` tensor of full MLP weight vectors (D = 8352 for
            the default L=32, H=128 architecture). Leading dims are batched.
        L: MLP input/output dimension (default 32).
        H: MLP hidden width (default 128).

    Returns:
        ``[..., D_eff]`` tensor of effective maps (D_eff = 1056 for the
        default architecture), with the same leading dims as ``weights``.
    """
    codec = WeightCodec(L=L, H=H)
    D = codec.D
    lead = weights.shape[:-1]
    w = weights.reshape(-1, D)  # [N, D]
    N = w.shape[0]

    # Unpack each weight vector into the four parameter tensors.
    # We slice the flat vector directly (faster than calling unpack per row).
    off_w1 = codec.offsets["fc1.weight"]
    off_b1 = codec.offsets["fc1.bias"]
    off_w2 = codec.offsets["fc2.weight"]
    off_b2 = codec.offsets["fc2.bias"]
    sz_w1 = codec.sizes["fc1.weight"]   # H*L
    sz_b1 = codec.sizes["fc1.bias"]     # H
    sz_w2 = codec.sizes["fc2.weight"]   # L*H
    sz_b2 = codec.sizes["fc2.bias"]     # L

    W1 = w[:, off_w1:off_w1 + sz_w1].view(N, H, L)   # [N, H, L]
    b1 = w[:, off_b1:off_b1 + sz_b1].view(N, H)       # [N, H]
    W2 = w[:, off_w2:off_w2 + sz_w2].view(N, L, H)   # [N, L, H]
    b2 = w[:, off_b2:off_b2 + sz_b2].view(N, L)       # [N, L]

    # Effective map: M = W2 @ W1  -> [N, L, L]
    M = torch.bmm(W2, W1)  # [N, L, L]
    # Effective bias: b_eff = W2 @ b1 + b2 -> [N, L]
    b_eff = torch.bmm(W2, b1.unsqueeze(-1)).squeeze(-1) + b2  # [N, L]

    M_flat = M.reshape(N, L * L)        # [N, L*L]
    eff = torch.cat([M_flat, b_eff], dim=-1)  # [N, L*L + L]
    return eff.reshape(*lead, L * L + L)


def effective_map_to_weights(eff_map: torch.Tensor, L: int = DEFAULT_L,
                             H: int = DEFAULT_H,
                             s_floor: float = _SVD_S_FLOOR) -> torch.Tensor:
    """Convert an effective linear map back to full MLP weights via SVD.

    Splits the effective map into ``M`` [L, L] and ``b_eff`` [L], computes the
    SVD ``M = U @ diag(S) @ V^T``, and constructs a valid ``(W1, W2, b1, b2)``
    factorization (see module docstring). The result is packed into the
    canonical 8352-dim weight vector via ``WeightCodec``.

    Args:
        eff_map: ``[..., D_eff]`` tensor of effective maps (D_eff = 1056 for
            the default architecture). Leading dims are batched.
        L: MLP input/output dimension (default 32).
        H: MLP hidden width (default 128).
        s_floor: Floor for singular values (clamped to >= s_floor) to keep
            ``sqrt(S)`` numerically stable. Default 1e-10.

    Returns:
        ``[..., D]`` tensor of full MLP weight vectors (D = 8352 for the
        default architecture), with the same leading dims as ``eff_map``.
    """
    codec = WeightCodec(L=L, H=H)
    D = codec.D
    D_eff = L * L + L
    lead = eff_map.shape[:-1]
    e = eff_map.reshape(-1, D_eff)  # [N, D_eff]
    N = e.shape[0]

    M_flat = e[:, :L * L]            # [N, L*L]
    b_eff = e[:, L * L:]             # [N, L]
    M = M_flat.view(N, L, L)         # [N, L, L]

    # SVD: M = U @ diag(S) @ Vh.  torch.linalg.svd returns full matrices.
    #   U: [N, L, L], S: [N, L], Vh: [N, L, L]  (Vh = V^T)
    U, S, Vh = torch.linalg.svd(M, full_matrices=True)
    # Clamp singular values for numerical stability of sqrt(S).
    S_clamped = S.clamp_min(s_floor)
    sqrtS = torch.sqrt(S_clamped)     # [N, L]

    # W1 [H, L]: first L rows = diag(sqrt(S)) @ Vh, remaining (H-L) rows = 0.
    #   diag(sqrt(S)) @ Vh -> [N, L, L]
    W1_top = torch.bmm(torch.diag_embed(sqrtS), Vh)  # [N, L, L]
    W1 = torch.zeros(N, H, L, dtype=eff_map.dtype, device=eff_map.device)
    W1[:, :L, :] = W1_top

    # W2 [L, H]: first L cols = U @ diag(sqrt(S)), remaining (H-L) cols = 0.
    #   U @ diag(sqrt(S)) -> [N, L, L]
    W2_left = torch.bmm(U, torch.diag_embed(sqrtS))  # [N, L, L]
    W2 = torch.zeros(N, L, H, dtype=eff_map.dtype, device=eff_map.device)
    W2[:, :, :L] = W2_left

    # Biases: b1 [H] = 0, b2 [L] = b_eff.
    b1 = torch.zeros(N, H, dtype=eff_map.dtype, device=eff_map.device)
    b2 = b_eff  # [N, L]

    # Pack into the canonical flat vector [N, D].
    chunks = [
        W1.reshape(N, H * L),
        b1,
        W2.reshape(N, L * H),
        b2,
    ]
    weights = torch.cat(chunks, dim=-1)  # [N, D]
    assert weights.shape[-1] == D
    return weights.reshape(*lead, D)


class EffectiveMapCodec:
    """Codec wrapping the effective-map conversion with a fixed ``WeightCodec``.

    Provides ``weights_to_effective_map`` and ``effective_map_to_weights`` bound
    to a specific ``(L, H)`` architecture, plus convenience helpers for
    round-trip verification and MLP instantiation.

    Args:
        L: MLP input/output dimension (default 32).
        H: MLP hidden width (default 128).

    Attributes:
        L, H: The MLP dimensions.
        weight_codec: The underlying ``WeightCodec`` (for pack/unpack/instantiate).
        D: Full weight-space dimension (8352 for the default architecture).
        D_eff: Effective-map dimension (1056 for the default architecture).
    """

    def __init__(self, L: int = DEFAULT_L, H: int = DEFAULT_H):
        self.L = int(L)
        self.H = int(H)
        self.weight_codec = WeightCodec(L=self.L, H=self.H)
        self.D: int = self.weight_codec.D
        self.D_eff: int = self.L * self.L + self.L

    def to_effective(self, weights: torch.Tensor) -> torch.Tensor:
        """Full weights ``[..., D]`` -> effective map ``[..., D_eff]``."""
        return weights_to_effective_map(weights, L=self.L, H=self.H)

    def to_weights(self, eff_map: torch.Tensor,
                   s_floor: float = _SVD_S_FLOOR) -> torch.Tensor:
        """Effective map ``[..., D_eff]`` -> full weights ``[..., D]`` (via SVD)."""
        return effective_map_to_weights(eff_map, L=self.L, H=self.H,
                                        s_floor=s_floor)

    def round_trip(self, weights: torch.Tensor) -> torch.Tensor:
        """Full weights -> effective map -> full weights (SVD factorization)."""
        return self.to_weights(self.to_effective(weights))

    def __repr__(self) -> str:
        return (f"EffectiveMapCodec(L={self.L}, H={self.H}, "
                f"D={self.D}, D_eff={self.D_eff})")


def kernel_to_effective_map(
    kernel: torch.Tensor,
    L: int = DEFAULT_L,
) -> torch.Tensor:
    """Build the oracle effective map from a ground-truth convolution kernel.

    The effective map of a 1D 'same'-padding convolution with zero-padding is a
    Toeplitz matrix ``M`` [L, L] where ``M[i, j] = k[i - j + r]`` (when the index
    is in range, else 0), and ``b_eff = 0`` (the convolution has no bias).

    Args:
        kernel: ``[K]`` ground-truth FIR filter (K = 2*radius + 1, odd).
        L: Input/output length (default 32).

    Returns:
        ``[D_eff]`` effective map vector (1056-dim): ``M_flat ++ b_eff``.
    """
    k = kernel.reshape(-1).to(torch.float32)
    K = k.shape[0]
    if K % 2 == 0:
        raise ValueError(f"kernel size K must be odd, got {K}")
    r = K // 2
    M = torch.zeros(L, L, dtype=torch.float32)
    for i in range(L):
        for j in range(L):
            d = i - j  # diagonal offset
            idx = d + r  # index into kernel
            if 0 <= idx < K:
                M[i, j] = k[idx]
    b_eff = torch.zeros(L, dtype=torch.float32)
    return torch.cat([M.reshape(-1), b_eff], dim=0)


def effective_map_to_matrix(
    eff_map: torch.Tensor,
    L: int = DEFAULT_L,
) -> torch.Tensor:
    """Extract the L×L effective matrix M from a 1056-dim effective map.

    Args:
        eff_map: ``[D_eff]`` or ``[..., D_eff]`` effective map.
        L: Matrix dimension (default 32).

    Returns:
        ``[L, L]`` (or ``[..., L, L]``) matrix M.
    """
    e = eff_map.reshape(*eff_map.shape[:-1], L * L + L)
    M_flat = e[..., :L * L]
    return M_flat.reshape(*e.shape[:-1], L, L)


def instantiate_mlp_from_eff_map(
    eff_map: torch.Tensor,
    L: int = DEFAULT_L,
    H: int = DEFAULT_H,
) -> "MLPModel":
    """Instantiate an ``MLPModel`` from an effective map (1056-dim) via SVD.

    Convenience: combines ``effective_map_to_weights`` and ``instantiate_mlp``.

    Args:
        eff_map: ``[D_eff]`` or ``[1, D_eff]`` effective map vector (1056-dim).
        L, H: MLP dimensions.

    Returns:
        An ``MLPModel`` in eval mode.
    """
    e = eff_map.reshape(1, -1)
    weights = effective_map_to_weights(e, L=L, H=H)
    return instantiate_mlp(weights[0], L=L, H=H)
