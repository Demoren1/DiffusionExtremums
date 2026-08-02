"""Models for the conv-vs-MLP smoke test (Phase 1)."""
import torch
import torch.nn as nn

from src.models.mlp import MLPModel  # noqa: F401  (re-exported)


class Conv1dModel(nn.Module):
    """1D convolution baseline: a single conv1d layer with 'same' padding."""

    def __init__(self, L: int = 32, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=kernel_size,
            padding=pad, padding_mode="zeros", bias=bias,
        )
        self.L = L
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))
        return h.squeeze(1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
