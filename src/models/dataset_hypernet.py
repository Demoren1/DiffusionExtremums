"""Dataset-conditioned hypernetwork: few examples → weight vector.

DatasetEncoder: takes K example (x, y) pairs, passes through a small transformer
(1-2 self-attention layers) to extract the convolution signature, then pools to
an embedding.

WeightDecoder: embedding → MLP → flat weight vector θ [D].

Training: encoder sees K enc pairs, loss on different L loss pairs.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hypernetwork import functional_forward
from src.models.weight_codec import WeightCodec


class DatasetEncoder(nn.Module):
    """Extract dataset signature from a few (x, y) example pairs.

    Args:
        L: Input/output dimension (32).
        K_enc: Number of example pairs for the encoder (default 32).
        d_model: Transformer width (default 128).
        d_emb: Output embedding dimension (default 128).
        n_layers: Transformer layers (default 1).
        n_heads: Attention heads (default 4).
    """

    def __init__(
        self, L: int = 32, K_enc: int = 32,
        d_model: int = 128, d_emb: int = 128,
        n_layers: int = 1, n_heads: int = 4,
    ):
        super().__init__()
        self.K_enc = int(K_enc)
        self.d_model = int(d_model)

        # Project (x, y) concatenated → d_model.
        self.input_proj = nn.Linear(2 * L, d_model)

        # Transformer layers (pre-LN, self-attention over K examples).
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
            }))

        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_emb)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Encode K example pairs to dataset embedding.

        Args:
            x: [B, K_enc, L] input examples.
            y: [B, K_enc, L] target examples.

        Returns:
            emb: [B, d_emb].
        """
        # Concat (x, y) per example.
        tokens = torch.cat([x, y], dim=-1)               # [B, K, 2L]
        tokens = self.input_proj(tokens)                  # [B, K, d_model]

        for layer in self.layers:
            # Pre-LN self-attention
            normed = layer["ln1"](tokens)
            attn_out, _ = layer["attn"](normed, normed, normed, need_weights=False)
            tokens = tokens + attn_out
            # Pre-LN FFN
            normed = layer["ln2"](tokens)
            tokens = tokens + layer["ffn"](normed)

        tokens = self.final_ln(tokens)
        emb = tokens.mean(dim=1)                          # [B, d_model] mean pool
        return self.out_proj(emb)                         # [B, d_emb]


class WeightDecoder(nn.Module):
    """Embedding → MLP weights.

    Args:
        d_emb: Input embedding dimension.
        D: Output weight vector dimension (from WeightCodec).
        hidden: Hidden layer widths (default [256, 512]).
    """

    def __init__(self, d_emb: int = 128, D: int = 1072,
                 hidden: Tuple[int, ...] = (256, 512)):
        super().__init__()
        layers = []
        in_dim = d_emb
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(h))
            in_dim = h
        layers.append(nn.Linear(in_dim, D))
        self.net = nn.Sequential(*layers)

        # Zero-init last layer so initial output is near-zero.
        nn.init.normal_(self.net[-1].weight, std=1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Map embedding to flat weight vector.

        Args:
            emb: [B, d_emb].

        Returns:
            theta: [B, D].
        """
        return self.net(emb)


class DatasetHypernet(nn.Module):
    """Full pipeline: examples → embedding → weights.

    Args:
        encoder: DatasetEncoder instance.
        decoder: WeightDecoder instance.
        mlp_hidden: MLP hidden dimension H (for functional_forward).
    """

    def __init__(self, encoder: DatasetEncoder, decoder: WeightDecoder,
                 mlp_hidden: int = 16):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mlp_hidden = int(mlp_hidden)

    def forward(self, x_enc: torch.Tensor, y_enc: torch.Tensor) -> torch.Tensor:
        """Encode examples, decode to weights.

        Args:
            x_enc: [B, K_enc, L].
            y_enc: [B, K_enc, L].

        Returns:
            theta: [B, D].
        """
        emb = self.encoder(x_enc, y_enc)
        return self.decoder(emb)

    def compute_loss(
        self, x_enc: torch.Tensor, y_enc: torch.Tensor,
        x_loss: torch.Tensor, y_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward + functional MSE loss.

        Args:
            x_enc, y_enc: [B, K_enc, L] examples for the encoder.
            x_loss, y_loss: [B, N_loss, L] points for the loss.

        Returns:
            Scalar MSE loss.
        """
        theta = self.forward(x_enc, y_enc)
        y_pred = functional_forward(
            theta, x_loss, L=32, H=self.mlp_hidden)
        return F.mse_loss(y_pred, y_loss)
