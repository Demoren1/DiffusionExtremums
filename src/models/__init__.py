"""Models package: the target MLP architecture, weight codec, and diffusion hypernetwork.

Phase 2 introduces the shared ``MLPModel`` (the diffusion target architecture)
and ``WeightCodec`` (canonical flatten/unflatten of MLP weights into the
8352-dim vector the diffusion model operates on). The Phase 1 smoke models
re-export ``MLPModel`` from here so there is a single source of truth.

Phase 3 adds the diffusion hypernetwork:
- ``ConfigEncoder``: dataset config -> 14-dim feature vector -> 128-dim embedding
  (the conditioning path; enables zero-shot on held-out dataset IDs).
- ``WeightNormalizer``: per-dimension standardization of target weights so the
  diffusion model operates on ~N(0,1) data.
- ``DiffusionModel``: a compact DDPM over the 8352-dim weight space with
  FiLM-conditioned denoiser, forward-process training loss, and ancestral
  sampling. The denoiser is deliberately small (~4.6M params) — the core
  hypothesis is that a compact meta-learner discovers the convolution inductive
  bias in weight space.
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
from src.models.diffusion import (
    DiffusionModel,
    DenoiserNet,
    NoiseSchedule,
    SinusoidalTimeEmbedding,
    FiLMBlock,
    DEFAULT_D,
)

__all__ = [
    "MLPModel",
    "WeightCodec",
    # Phase 3: diffusion hypernetwork
    "ConfigEncoder",
    "CONFIG_FEATURE_DIM",
    "config_to_features",
    "configs_to_features",
    "WeightNormalizer",
    "DiffusionModel",
    "DenoiserNet",
    "NoiseSchedule",
    "SinusoidalTimeEmbedding",
    "FiLMBlock",
    "DEFAULT_D",
]
