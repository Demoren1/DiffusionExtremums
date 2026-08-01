"""Deterministic effective-map regressor (sanity-check baseline).

A supervised baseline that maps the 14-dim config feature vector directly to
the 1056-dim effective linear map. Its purpose is to isolate whether the
config/target pipeline is sound: if this simple deterministic regressor learns
the held-out effective maps and yields convolution-like functional outputs,
then the config/target alignment is correct.

Architecture (default): ``14 -> 256 -> 512 -> 512 -> 1056`` with GELU
activations and a residual connection across the two 512-wide hidden layers
(justified: the 14-dim input is information-poor relative to the 1056-dim
output, so a residual highway stabilizes optimization of the deep 512->512
block). Layer norms are placed before each activation (pre-norm) to keep the
1056-dim output well-scaled for the per-dimension WeightNormalizer.

The model consumes the *same* 14-dim feature tensor produced by
``src.models.config_encoder.configs_to_features`` — it does **not** duplicate
the feature construction.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from src.configs.base import DatasetConfig
from src.models.config_encoder import CONFIG_FEATURE_DIM, configs_to_features
from src.models.effective_map import DEFAULT_EFF_D


@dataclass(frozen=True)
class RegressorConfig:
    """Architecture configuration for ``EffectiveMapRegressor``.

    Attributes:
        feature_dim: Input dimension (the 14-dim config feature vector).
            Default ``CONFIG_FEATURE_DIM`` (=14).
        hidden_dims: List of hidden layer widths. Default ``[256, 512, 512]``
            giving the suggested ``14 -> 256 -> 512 -> 512 -> 1056``.
        output_dim: Output dimension (the effective map, 1056).
        activation: Activation name, ``"gelu"`` or ``"silu"``.
        use_residual: If True, add a residual connection across the last two
            hidden layers (requires them to have equal width).
        use_layer_norm: If True, apply LayerNorm before each activation.
        dropout: Dropout probability (0 = disabled). Default 0 (the corpus is
            small; dropout is not needed and would hurt the sanity signal).
    """

    feature_dim: int = CONFIG_FEATURE_DIM
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 512])
    output_dim: int = DEFAULT_EFF_D
    activation: str = "gelu"
    use_residual: bool = True
    use_layer_norm: bool = True
    dropout: float = 0.0


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation {name!r}; expected gelu/silu/relu")


class _ResidualBlock(nn.Module):
    """Pre-norm residual block: ``x + f(LN(x))`` with equal in/out width."""

    def __init__(self, dim: int, activation: str, use_layer_norm: bool,
                 dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.linear = nn.Linear(dim, dim)
        self.act = _make_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.drop(h)
        return x + h


class EffectiveMapRegressor(nn.Module):
    """Deterministic MLP regressor: config features [14] -> effective map [1056].

    This is a standard supervised regression network. It is a sanity-check
    baseline: if it learns the held-out effective maps, the config/target
    pipeline is sound.

    Args:
        config: A ``RegressorConfig``. If None, uses the default
            ``14 -> 256 -> 512 -> 512 -> 1056`` architecture.
        feature_dim: Override the input dimension (convenience; ignored if
            ``config`` is given).
        output_dim: Override the output dimension (convenience; ignored if
            ``config`` is given).
        hidden_dims: Override the hidden widths (convenience; ignored if
            ``config`` is given).
    """

    def __init__(
        self,
        config: Optional[RegressorConfig] = None,
        *,
        feature_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if config is None:
            config = RegressorConfig(
                feature_dim=feature_dim if feature_dim is not None
                else CONFIG_FEATURE_DIM,
                hidden_dims=list(hidden_dims) if hidden_dims is not None
                else [256, 512, 512],
                output_dim=output_dim if output_dim is not None
                else DEFAULT_EFF_D,
            )
        self.config = config
        self.feature_dim = config.feature_dim
        self.output_dim = config.output_dim

        dims = [config.feature_dim] + list(config.hidden_dims)
        act = _make_activation(config.activation)

        # Build the trunk: Linear -> [LN -> act -> [Dropout]] per layer.
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            # Residual block only valid for equal-width interior layers.
            is_residual_candidate = (
                config.use_residual and i >= 1 and in_d == out_d)
            if is_residual_candidate:
                layers.append(_ResidualBlock(
                    out_d, config.activation,
                    config.use_layer_norm, config.dropout))
            else:
                if config.use_layer_norm:
                    layers.append(nn.LayerNorm(out_d))
                layers.append(act)
                if config.dropout > 0.0:
                    layers.append(nn.Dropout(config.dropout))
        self.trunk = nn.Sequential(*layers)

        # Output head (no activation: regression in normalized effective-map
        # space, which is approximately N(0,1) per dimension).
        self.head = nn.Linear(dims[-1], config.output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map config features to an effective map.

        Args:
            features: ``[B, feature_dim]`` tensor from ``configs_to_features``.

        Returns:
            ``[B, output_dim]`` predicted (normalized) effective map.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features last dim {features.shape[-1]} != "
                f"feature_dim {self.feature_dim}")
        h = self.trunk(features)
        return self.head(h)

    def predict(self, configs: Sequence[Union[DatasetConfig, dict]]) -> torch.Tensor:
        """Convenience: configs -> predicted effective map.

        Args:
            configs: Sequence of config dicts or ``DatasetConfig``.

        Returns:
            ``[B, output_dim]`` predicted effective map (on the model's device).
        """
        feats = configs_to_features(configs, dtype=next(self.parameters()).dtype)
        return self.forward(feats.to(next(self.parameters()).device))

    def n_params(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (f"EffectiveMapRegressor(feature_dim={self.feature_dim}, "
                f"hidden_dims={self.config.hidden_dims}, "
                f"output_dim={self.output_dim}, "
                f"params={self.n_params():,})")
