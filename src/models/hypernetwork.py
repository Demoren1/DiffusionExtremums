"""Low-rank weight decoder + functional forward (design doc sections 4-5)."""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mlp_encoder import (
    neuron_token_types,
    svd_token_types,
    theta_to_neuron_tokens,
    theta_to_svd_tokens,
    unpack_batch,
)
from src.models.weight_codec import WeightCodec

LOG_S_INIT = -3.0


class _FactorHead(nn.Module):
    """Two-layer GELU head producing one factor component from z [B, d_model]."""

    def __init__(self, d_model: int, out_dim: int,
                 init_mode: str = "normal", hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        last = self.net[-1]
        if init_mode == "zero":
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif init_mode == "log_s":
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, LOG_S_INIT)
        else:  # "normal": small-std directions
            nn.init.normal_(last.weight, std=0.02)
            nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LowRankWeightDecoder(nn.Module):
    """Generate low-rank MLP weight factors from encoder outputs."""

    def __init__(self, cond_dim: int = 512, d_model: int = 256,
                 L: int = 32, H: int = 128, r1: int = 16, r2: int = 16,
                 head_hidden: int = 256):
        super().__init__()
        self.L = int(L)
        self.H = int(H)
        self.r1 = int(r1)
        self.r2 = int(r2)

        n_queries = r1 + r2 + 2
        self.queries = nn.Parameter(torch.zeros(n_queries, d_model))
        nn.init.normal_(self.queries, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            d_model, 8, dropout=0.0, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

        self.head_a1 = _FactorHead(d_model, H, init_mode="normal", hidden=head_hidden)
        self.head_b1 = _FactorHead(d_model, L, init_mode="normal", hidden=head_hidden)
        self.head_a2 = _FactorHead(d_model, L, init_mode="normal", hidden=head_hidden)
        self.head_b2 = _FactorHead(d_model, H, init_mode="normal", hidden=head_hidden)
        self.head_s = _FactorHead(d_model, 1, init_mode="log_s", hidden=head_hidden)
        self.head_b1b = _FactorHead(d_model, H, init_mode="zero", hidden=head_hidden)
        self.head_b2b = _FactorHead(d_model, L, init_mode="zero", hidden=head_hidden)

        self.cond_proj = nn.Linear(cond_dim, d_model)

    def forward(
        self, enc: torch.Tensor, cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B = enc.shape[0]
        n_queries = self.r1 + self.r2 + 2

        q = self.queries.unsqueeze(0).expand(B, n_queries, -1)  # [B, nq, d_model]
        z, _ = self.cross_attn(q, enc, enc, need_weights=False)
        z = self.ln(z + q)

        r1, r2 = self.r1, self.r2
        z1 = z[:, :r1]                      # W1 factors
        z2 = z[:, r1:r1 + r2]               # W2 factors
        zb1 = z[:, r1 + r2]                 # b1
        zb2 = z[:, r1 + r2 + 1]             # b2

        A1 = F.normalize(self.head_a1(z1), dim=-1)
        B1 = F.normalize(self.head_b1(z1), dim=-1)
        s1 = torch.exp(self.head_s(z1).squeeze(-1))
        A2 = F.normalize(self.head_a2(z2), dim=-1)
        B2 = F.normalize(self.head_b2(z2), dim=-1)
        s2 = torch.exp(self.head_s(z2).squeeze(-1))

        c = self.cond_proj(cond)
        b1 = self.head_b1b(zb1 + c)
        b2 = self.head_b2b(zb2 + c)

        return {
            "A1": A1, "B1": B1, "s1": s1,
            "A2": A2, "B2": B2, "s2": s2,
            "b1": b1, "b2": b2,
        }


def assemble_weights(
    factors: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    A1, B1, s1 = factors["A1"], factors["B1"], factors["s1"]
    A2, B2, s2 = factors["A2"], factors["B2"], factors["s2"]
    W1 = torch.einsum("bkh,bkl,bk->bhl", A1, B1, s1)
    W2 = torch.einsum("bkl,bkh,bk->blh", A2, B2, s2)
    return W1, factors["b1"], W2, factors["b2"]


def batched_pack(
    W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor,
    codec: WeightCodec,
) -> torch.Tensor:
    B = W1.shape[0]
    if (tuple(W1.shape[1:]), tuple(W2.shape[1:])) != (
            codec.shapes["fc1.weight"], codec.shapes["fc2.weight"]):
        raise ValueError("W1/W2 shapes do not match the codec")
    return torch.cat(
        [W1.reshape(B, -1), b1.reshape(B, -1),
         W2.reshape(B, -1), b2.reshape(B, -1)], dim=1)


def functional_forward(
    theta: torch.Tensor, x: torch.Tensor, L: int = 32, H: int = 128,
) -> torch.Tensor:
    """Differentiable forward of the target MLP y = W2 @ relu(W1 @ x + b1) + b2."""
    codec = WeightCodec(L=L, H=H)
    W1, b1, W2, b2 = unpack_batch(theta, codec)  # W1 [B,H,L], W2 [B,L,H]
    h = torch.relu(
        torch.einsum("bnl,bhl->bnh", x, W1) + b1[:, None, :])  # [B,n,H]
    y_hat = torch.einsum("bnh,blh->bnl", h, W2) + b2[:, None, :]
    return y_hat


class Hypernetwork(nn.Module):
    """Full hypernetwork: encoder (weights -> cond) + decoder (cond -> weights)."""

    def __init__(self, encoder: nn.Module, decoder: LowRankWeightDecoder,
                 codec: WeightCodec = None, token_mode: str = "neuron"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codec = codec if codec is not None else WeightCodec(
            L=decoder.L, H=decoder.H)
        if token_mode not in ("neuron", "svd"):
            raise ValueError(f"token_mode must be 'neuron' or 'svd', got {token_mode!r}")
        self.token_mode = token_mode

    def forward(self, theta_in: torch.Tensor) -> torch.Tensor:
        if theta_in.dim() == 1:
            theta_in = theta_in.unsqueeze(0)
        device = theta_in.device
        if self.token_mode == "neuron":
            tokens = theta_to_neuron_tokens(theta_in, self.codec)
            types = neuron_token_types(theta_in, self.codec, device)
        else:
            tokens = theta_to_svd_tokens(theta_in, self.codec)
            types = svd_token_types(theta_in, self.codec, device)
        enc, cond = self.encoder(tokens, types)
        factors = self.decoder(enc, cond)
        W1, b1, W2, b2 = assemble_weights(factors)
        return batched_pack(W1, b1, W2, b2, self.codec)
