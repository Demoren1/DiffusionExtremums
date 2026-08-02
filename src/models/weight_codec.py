"""Canonical weight vectorization for the target MLP."""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.models.mlp import MLPModel

# Canonical parameter names in the fixed flatten order.
PARAM_NAMES: Tuple[str, ...] = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")


class WeightCodec:
    """Pack/unpack/instantiate the target MLP weights <-> flat theta vector."""

    def __init__(self, L: int = 32, H: int = 128):
        self.L = int(L)
        self.H = int(H)
        self.shapes: Dict[str, Tuple[int, ...]] = {
            "fc1.weight": (self.H, self.L),
            "fc1.bias": (self.H,),
            "fc2.weight": (self.L, self.H),
            "fc2.bias": (self.L,),
        }
        self.sizes: Dict[str, int] = {
            name: int(torch.tensor(shape).prod().item())
            for name, shape in self.shapes.items()
        }
        self.offsets: Dict[str, int] = {}
        off = 0
        for name in PARAM_NAMES:
            self.offsets[name] = off
            off += self.sizes[name]
        self.D: int = off  # total dimension = 8352

    def pack(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if set(params.keys()) != set(PARAM_NAMES):
            raise ValueError(
                f"params keys {set(params.keys())} != expected {set(PARAM_NAMES)}")
        chunks: List[torch.Tensor] = []
        for name in PARAM_NAMES:
            t = params[name]
            if tuple(t.shape) != self.shapes[name]:
                raise ValueError(
                    f"{name} has shape {tuple(t.shape)}, expected {self.shapes[name]}")
            chunks.append(t.reshape(-1).contiguous())
        return torch.cat(chunks, dim=0)

    def pack_model(self, model: nn.Module) -> torch.Tensor:
        params = {name: p.detach() for name, p in model.named_parameters()}
        return self.pack(params)

    def unpack(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        theta = theta.reshape(-1)
        if theta.numel() != self.D:
            raise ValueError(f"theta has {theta.numel()} elements, expected {self.D}")
        params: Dict[str, torch.Tensor] = {}
        for name in PARAM_NAMES:
            off = self.offsets[name]
            size = self.sizes[name]
            params[name] = theta[off:off + size].view(self.shapes[name]).contiguous()
        return params

    def instantiate(self, theta: torch.Tensor) -> MLPModel:
        """Build an MLPModel and load theta, in eval mode with gradients disabled."""
        model = MLPModel(L=self.L, H=self.H)
        params = self.unpack(theta)
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(params[name].to(p.device, dtype=p.dtype))
        model.to(theta.device, dtype=theta.dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model


def instantiate_mlp(
    weights: torch.Tensor,
    L: int = 32,
    H: int = 128,
) -> MLPModel:
    codec = WeightCodec(L=L, H=H)
    w = weights.reshape(-1)
    if w.numel() != codec.D:
        raise ValueError(
            f"weights has {w.numel()} elements, expected D={codec.D}")
    return codec.instantiate(w)
