"""Config-feature encoder: dataset config -> fixed-dim feature vector -> embedding.

This is the **conditioning path** of the effective-map regressor: the model
predicts the effective map from the *dataset configuration* (family, kernel,
radius, noise_std). Encoding the config as a fixed feature vector (rather than
a learned per-ID embedding table) enables **zero-shot generalization to
held-out dataset IDs**: any config, seen or unseen, maps to a conditioning
embedding through the same encoder.

Feature layout (14-dim, per plan Section 3.3, ``dim = 5 + 7 + 1 + 1 = 14``)::

    index       content                      range / notes
    --------    -------------------------    --------------------------------
    [0:5]       family one-hot               order = FAMILIES = (MA, DIFF,
                                                 GAUSS, MATCH, RAND); exactly
                                                 one entry is 1.0, rest 0.0.
    [5:12]      kernel values, zero-padded   the K=2*radius+1 kernel taps are
                 to K_max = 7                CENTER-aligned in a length-7
                                                 window (the center tap is
                                                 always at slot index 3 = MAX
                                                 radius); unused side slots are
                                                 0.0. Raw kernel values (no
                                                 normalization): they carry the
                                                 actual convolution rule and
                                                 their scale is informative.
    [12]        radius  / 3.0                 radius in {1,2,3} -> {0.33,0.67,1.0}
                                                 (normalized by MAX_RADIUS so
                                                 the feature is O(1)).
    [13]        noise_std / 0.2              noise_std in {0,0.05,0.1,0.2} ->
                                                 {0,0.25,0.5,1.0} (normalized by
                                                 the grid maximum NOISE_STD_MAX).

Fields NOT used as conditioning (and why):
- ``n_train``: fixed at 1024 for all Phase 2 collected targets, so it carries
  no per-dataset conditioning information. (If a future phase varies it, add a
  15th feature.)
- ``n_test``: fixed at 512; does not affect the target weight distribution.
- ``seed``: determines the input/noise *draws*, not the rule; the target weight
  distribution is conditionally independent of ``seed`` given (family, kernel,
  radius, noise_std).
- ``L``: fixed at 32 (the MLP input length); constant, no conditioning signal.

The encoder is a small MLP ``[14, 64, 128]`` (SiLU activations) producing a
128-dim conditioning embedding ``c``.
"""
from typing import Dict, Sequence, Union

import torch
import torch.nn as nn

from src.configs.base import DatasetConfig
from src.data.families import FAMILIES

# Maximum kernel radius / size in v1 (radius in {1,2,3} -> K in {3,5,7}).
MAX_RADIUS: int = 3
K_MAX: int = 2 * MAX_RADIUS + 1  # 7
# Normalization constants for the scalar features (the v1 discrete grids).
RADIUS_MAX: float = float(MAX_RADIUS)          # 3.0
NOISE_STD_MAX: float = 0.2                      # max of {0.0,0.05,0.1,0.2}
# Dimensionality of the config feature vector (documented layout above).
CONFIG_FEATURE_DIM: int = len(FAMILIES) + K_MAX + 2  # 5 + 7 + 1 + 1 = 14

# Family name -> one-hot index (stable, matches FAMILIES order).
_FAMILY_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FAMILIES)}


# ---------------------------------------------------------------------------
# Config -> feature vector
# ---------------------------------------------------------------------------

def _as_config_dict(config: Union[DatasetConfig, Dict]) -> Dict:
    """Coerce a ``DatasetConfig`` or a raw dict (from ``configs.json``) to a dict.

    The dict form (written by ``src/training/collect_targets.py``) has keys:
    ``family, kernel, radius, noise_std, n_train, n_test, seed, L, dataset_id``.
    """
    if isinstance(config, DatasetConfig):
        return {
            "family": config.family,
            "kernel": list(config.kernel),
            "radius": config.radius,
            "noise_std": config.noise_std,
        }
    if isinstance(config, dict):
        # Tolerate missing optional keys; only the 4 conditioning fields matter.
        return {
            "family": str(config["family"]),
            "kernel": [float(v) for v in config["kernel"]],
            "radius": int(config["radius"]),
            "noise_std": float(config["noise_std"]),
        }
    raise TypeError(f"config must be DatasetConfig or dict, got {type(config)!r}")


def config_to_features(config: Union[DatasetConfig, Dict],
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Encode a single dataset config into the 14-dim feature vector.

    Args:
        config: A ``DatasetConfig`` or its dict form (e.g. one entry of
            ``configs.json``).
        dtype: Output tensor dtype (default float32).

    Returns:
        1-D tensor of length ``CONFIG_FEATURE_DIM`` (14).
    """
    d = _as_config_dict(config)
    family = d["family"]
    if family not in _FAMILY_INDEX:
        raise ValueError(
            f"unknown family {family!r}; expected one of {FAMILIES}")
    kernel = list(d["kernel"])
    radius = int(d["radius"])
    if radius < 0 or radius > MAX_RADIUS:
        raise ValueError(
            f"radius {radius} out of range [0, {MAX_RADIUS}]")
    K = 2 * radius + 1
    if len(kernel) != K:
        raise ValueError(
            f"kernel length {len(kernel)} != 2*radius+1 = {K}")

    # Center-align the kernel in a length-K_MAX window: the center tap (index
    # `radius`) lands at slot MAX_RADIUS (=3), so kernels of different radii
    # share a common center position. Unused side slots are zero.
    padded = torch.zeros(K_MAX, dtype=dtype)
    offset = MAX_RADIUS - radius  # left padding
    for j, v in enumerate(kernel):
        padded[offset + j] = float(v)

    # Family one-hot.
    onehot = torch.zeros(len(FAMILIES), dtype=dtype)
    onehot[_FAMILY_INDEX[family]] = 1.0

    # Normalized scalars.
    radius_norm = torch.tensor([radius / RADIUS_MAX], dtype=dtype)
    noise_norm = torch.tensor([float(d["noise_std"]) / NOISE_STD_MAX],
                              dtype=dtype)

    return torch.cat([onehot, padded, radius_norm, noise_norm], dim=0)


def configs_to_features(
        configs: Sequence[Union[DatasetConfig, Dict]],
        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Encode a batch of configs into a ``[B, CONFIG_FEATURE_DIM]`` tensor.

    Args:
        configs: Sequence of ``DatasetConfig`` or dicts.
        dtype: Output tensor dtype.

    Returns:
        2-D tensor ``[B, 14]``.
    """
    rows = [config_to_features(c, dtype=dtype) for c in configs]
    if not rows:
        return torch.zeros(0, CONFIG_FEATURE_DIM, dtype=dtype)
    return torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# ConfigEncoder module: features -> 128-dim embedding
# ---------------------------------------------------------------------------

class ConfigEncoder(nn.Module):
    """MLP that maps the 14-dim config feature vector to a conditioning embedding.

    Architecture (plan Section 3.3): ``MLP([14, 64, 128])`` with SiLU
    activations. Produces a 128-dim conditioning embedding ``c``.

    Args:
        feature_dim: Input feature dimension (default ``CONFIG_FEATURE_DIM``=14).
        hidden_dim: Hidden width of the encoder MLP (default 64).
        embed_dim: Output embedding dimension (default 128; the conditioning
            dimension consumed by the downstream model).
    """

    def __init__(self, feature_dim: int = CONFIG_FEATURE_DIM,
                 hidden_dim: int = 64, embed_dim: int = 128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map config features to a conditioning embedding.

        Args:
            features: ``[B, feature_dim]`` tensor from ``config_to_features`` /
                ``configs_to_features``.

        Returns:
            ``[B, embed_dim]`` conditioning embedding.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features last dim {features.shape[-1]} != "
                f"feature_dim {self.feature_dim}")
        return self.net(features)

    def encode_configs(self, configs: Sequence[Union[DatasetConfig, Dict]]
                       ) -> torch.Tensor:
        """Convenience: configs -> embedding (features computed internally).

        Args:
            configs: Sequence of ``DatasetConfig`` or dicts.

        Returns:
            ``[B, embed_dim]`` conditioning embedding.
        """
        feats = configs_to_features(configs, dtype=next(self.parameters()).dtype)
        return self.forward(feats.to(next(self.parameters()).device))
