"""Models package: the target MLP architecture and weight codec.

Phase 2 introduces the shared ``MLPModel`` (the diffusion target architecture)
and ``WeightCodec`` (canonical flatten/unflatten of MLP weights into the
8352-dim vector the diffusion model operates on). The Phase 1 smoke models
re-export ``MLPModel`` from here so there is a single source of truth.
"""
from src.models.mlp import MLPModel
from src.models.weight_codec import WeightCodec

__all__ = ["MLPModel", "WeightCodec"]
