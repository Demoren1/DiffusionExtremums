"""Models package: target MLP architecture and weight codec.

- ``MLPModel``: the shared target MLP architecture.
- ``WeightCodec``: canonical flatten/unflatten of MLP weights into a flat
  ``theta`` vector, plus ``instantiate_mlp``.
"""
from src.models.mlp import MLPModel
from src.models.weight_codec import WeightCodec

__all__ = [
    "MLPModel",
    "WeightCodec",
]
