"""Models package: target MLP architecture and weight codec."""
from src.models.mlp import MLPModel
from src.models.weight_codec import WeightCodec

__all__ = [
    "MLPModel",
    "WeightCodec",
]
