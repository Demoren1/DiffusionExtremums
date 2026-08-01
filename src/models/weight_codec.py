"""Canonical weight vectorization for the target MLP.

The regressor pipeline operates on a single flat vector ``theta in R^D`` that
is reshaped into the MLP's parameters. This module defines the **fixed,
documented ordering** (plan Section 2.4) and provides ``pack`` / ``unpack`` /
``instantiate``.

Canonical ordering (C-order / row-major, matching PyTorch ``nn.Linear`` storage):

    theta = concat([
        flatten(fc1.weight),   # H*L  = 128*32 = 4096
        flatten(fc1.bias),     # H    = 128
        flatten(fc2.weight),   # L*H  = 32*128 = 4096
        flatten(fc2.bias),     # L    = 32
    ])
    # D = 4096 + 128 + 4096 + 32 = 8352

Reshape spec (row-major / C-order):
- ``fc1.weight`` shape ``[H, L]`` -> first ``H*L`` entries,
  ``weight[i, j] = theta[i*L + j]``.
- ``fc1.bias``   shape ``[H]``    -> next ``H`` entries.
- ``fc2.weight`` shape ``[L, H]`` -> next ``L*H`` entries,
  ``weight[i, j] = theta[offset + i*H + j]``.
- ``fc2.bias``   shape ``[L]``    -> last ``L`` entries.

A single ``WeightCodec`` handles ``pack(params_dict) -> theta`` and
``unpack(theta) -> params_dict`` and ``instantiate(theta) -> MLPModel``.
``D = 8352`` is the MLP weight-vector dimension.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.models.mlp import MLPModel

# Canonical parameter names in the fixed flatten order.
PARAM_NAMES: Tuple[str, ...] = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")


class WeightCodec:
    """Pack/unpack/instantiate the target MLP weights <-> flat theta vector.

    The ordering is fixed (see module docstring) and independent of any
    particular model instance, so targets saved by Phase 2 are loadable by
    Phase 3 without ambiguity.

    Args:
        L: Input length (default 32).
        H: Hidden width (default 128).
    """

    def __init__(self, L: int = 32, H: int = 128):
        self.L = int(L)
        self.H = int(H)
        # Per-parameter shapes in the canonical order.
        self.shapes: Dict[str, Tuple[int, ...]] = {
            "fc1.weight": (self.H, self.L),
            "fc1.bias": (self.H,),
            "fc2.weight": (self.L, self.H),
            "fc2.bias": (self.L,),
        }
        # Per-parameter flat sizes and cumulative offsets.
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

    # ------------------------------------------------------------------
    # pack / unpack
    # ------------------------------------------------------------------
    def pack(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten a parameter dict (name -> tensor) into a single ``[D]`` vector.

        Args:
            params: Mapping with keys ``fc1.weight`` (``[H,L]``), ``fc1.bias``
                (``[H]``), ``fc2.weight`` (``[L,H]``), ``fc2.bias`` (``[L]``).

        Returns:
            1-D float tensor of length ``D`` (8352), C-order concatenation.
        """
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
        """Convenience: pack the parameters of an ``MLPModel`` (or compatible)."""
        params = {name: p.detach() for name, p in model.named_parameters()}
        return self.pack(params)

    def unpack(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split a flat ``[D]`` vector back into the parameter dict (reshaped).

        Args:
            theta: 1-D tensor of length ``D`` (8352).

        Returns:
            Dict ``name -> tensor`` with the original ``nn.Linear`` shapes.
        """
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
        """Build an ``MLPModel`` and load ``theta`` into its parameters.

        The returned model is on the same device/dtype as ``theta`` and in eval
        mode. Gradients are disabled on its parameters (``requires_grad_(False)``)
        since it is meant for evaluation, not further training.
        """
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
    """Instantiate an ``MLPModel`` from a flat weight vector.

    Uses ``WeightCodec.instantiate`` to build the model and load the weights.
    The returned model is in eval mode with ``requires_grad=False`` (ready for
    evaluation, not further training).

    Args:
        weights: ``[D]`` or ``[1, D]`` flat weight vector (8352-dim).
        L, H: MLP dimensions (default 32, 128).

    Returns:
        An ``MLPModel`` on the same device/dtype as ``weights``, in eval mode.
    """
    codec = WeightCodec(L=L, H=H)
    w = weights.reshape(-1)
    if w.numel() != codec.D:
        raise ValueError(
            f"weights has {w.numel()} elements, expected D={codec.D}")
    return codec.instantiate(w)
