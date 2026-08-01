"""Models package: target MLP architecture, weight codec, and effective map.

- ``MLPModel``: the shared target MLP architecture.
- ``WeightCodec``: canonical flatten/unflatten of MLP weights into a flat
  ``theta`` vector, plus ``instantiate_mlp``.
- ``ConfigEncoder``: dataset config -> 14-dim feature vector -> 128-dim
  embedding (used as the regressor conditioning path).
- ``WeightNormalizer``: per-dimension standardization of target vectors.
"""
from src.models.mlp import MLPModel
from src.models.weight_codec import WeightCodec
from src.models.config_encoder import (
    ConfigEncoder,
    CONFIG_FEATURE_DIM,
    config_to_features,
    configs_to_features,
)
from src.models.weight_norm import WeightNormalizer

__all__ = [
    "MLPModel",
    "WeightCodec",
    "ConfigEncoder",
    "CONFIG_FEATURE_DIM",
    "config_to_features",
    "configs_to_features",
    "WeightNormalizer",
]
