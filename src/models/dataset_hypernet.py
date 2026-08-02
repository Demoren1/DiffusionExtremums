"""Dataset-conditioned hypernetwork: few examples → weight vector."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hypernetwork import functional_forward
from src.models.weight_codec import WeightCodec


class DatasetEncoder(nn.Module):
    """Extract dataset signature from a few (x, y) example pairs."""

    def __init__(
        self, L: int = 32, K_enc: int = 32,
        d_model: int = 128, d_emb: int = 128,
        n_layers: int = 1, n_heads: int = 4,
    ):
        super().__init__()
        self.K_enc = int(K_enc)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(2 * L, d_model)

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
        tokens = torch.cat([x, y], dim=-1)               # [B, K, 2L]
        tokens = self.input_proj(tokens)                  # [B, K, d_model]

        for layer in self.layers:
            normed = layer["ln1"](tokens)
            attn_out, _ = layer["attn"](normed, normed, normed, need_weights=False)
            tokens = tokens + attn_out
            normed = layer["ln2"](tokens)
            tokens = tokens + layer["ffn"](normed)

        tokens = self.final_ln(tokens)
        emb = tokens.mean(dim=1)                          # [B, d_model] mean pool
        return self.out_proj(emb)                         # [B, d_emb]


class WeightDecoder(nn.Module):
    """Embedding → MLP weights."""

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

        nn.init.normal_(self.net[-1].weight, std=1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class DatasetHypernet(nn.Module):
    """Full pipeline: examples → embedding → weights."""

    def __init__(self, encoder: DatasetEncoder, decoder: WeightDecoder,
                 mlp_hidden: int = 16):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mlp_hidden = int(mlp_hidden)

    def forward(self, x_enc: torch.Tensor, y_enc: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x_enc, y_enc)
        return self.decoder(emb)

    def compute_loss(
        self, x_enc: torch.Tensor, y_enc: torch.Tensor,
        x_loss: torch.Tensor, y_loss: torch.Tensor,
    ) -> torch.Tensor:
        theta = self.forward(x_enc, y_enc)
        y_pred = functional_forward(
            theta, x_loss, L=32, H=self.mlp_hidden)
        return F.mse_loss(y_pred, y_loss)
