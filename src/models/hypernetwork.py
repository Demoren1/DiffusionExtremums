"""Functional forward of the target MLP (design doc sections 4-5)."""
import torch

from src.models.mlp_encoder import unpack_batch
from src.models.weight_codec import WeightCodec


def functional_forward(
    theta: torch.Tensor, x: torch.Tensor, L: int = 32, H: int = 128,
) -> torch.Tensor:
    """Differentiable forward of the target MLP y = W2 @ relu(W1 @ x + b1) + b2."""
    codec = WeightCodec(L=L, H=H)
    W1, b1, W2, b2 = unpack_batch(theta, codec)  # W1 [B,H,L], W2 [B,L,H]
    h = torch.relu(
        torch.einsum("bnl,bhl->bnh", x, W1) + b1[:, None, :])  # [B,n,H]
    y_hat = torch.einsum("bnh,blh->bnl", h, W2) + b2[:, None, :]
    return y_hat
