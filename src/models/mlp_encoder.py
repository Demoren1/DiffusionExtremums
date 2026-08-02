"""Input MLP encoder: weight vector -> permutation-equivariant token embeddings.

Implements the input side of the hypernetwork (design doc
``plans/plan_hypernet_ru.md``, section 2-3):

- ``theta_to_neuron_tokens`` (primary): each hidden neuron becomes one token
  ``concat(W1[i,:], b1[i], W2[:,i])`` (dim ``2*L+1 = 65``) plus one bias token
  for ``b2`` -> ``T = H + 1 = 129`` tokens covering all ``D = 8352`` params
  losslessly. The hidden-layer permutation symmetry becomes a token
  permutation, which the transformer (no positional encodings) handles
  structurally.
- ``theta_to_svd_tokens`` (ablation): per-matrix SVD triples ``(u_k, s_k, v_k)``
  -> ``T = 66`` tokens of dim 161.

``MLPEncoder`` is a set-transformer (pre-LN, no positional encodings, type
embeddings, CLS token) producing ``(enc, cond)`` where
``cond = concat(enc_cls, mean_pool(enc_tokens))``.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.weight_codec import PARAM_NAMES, WeightCodec

# Token layout for the neuron representation (per plan section 2.1).
NEURON_TOKEN_DIM = 2 * 32 + 1  # L + 1 + L = 65 (with default L=32)
N_TOKEN_TYPES = 2              # hidden neuron token (0), bias token (1)


# ---------------------------------------------------------------------------
# Tokenization (lossless / lossy weight -> token-tensor transforms)
# ---------------------------------------------------------------------------

def unpack_batch(
    theta: torch.Tensor, codec: WeightCodec,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched ``WeightCodec.unpack``: ``[B, D]`` -> ``(W1, b1, W2, b2)``.

    Args:
        theta: ``[B, D]`` flat weight vectors (canonical order).
        codec: Codec defining the canonical layout.

    Returns:
        ``W1 [B, H, L]``, ``b1 [B, H]``, ``W2 [B, L, H]``, ``b2 [B, L]``.
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    if theta.shape[1] != codec.D:
        raise ValueError(
            f"theta has dim {theta.shape[1]}, expected D={codec.D}")
    B = theta.shape[0]
    H, L = codec.H, codec.L
    o1, s1 = codec.offsets["fc1.weight"], codec.sizes["fc1.weight"]
    ob, sb = codec.offsets["fc1.bias"], codec.sizes["fc1.bias"]
    o2, s2 = codec.offsets["fc2.weight"], codec.sizes["fc2.weight"]
    ob2, sb2 = codec.offsets["fc2.bias"], codec.sizes["fc2.bias"]
    W1 = theta[:, o1:o1 + s1].view(B, H, L)
    b1 = theta[:, ob:ob + sb].view(B, H)
    W2 = theta[:, o2:o2 + s2].view(B, L, H)
    b2 = theta[:, ob2:ob2 + sb2].view(B, L)
    return W1, b1, W2, b2


def theta_to_neuron_tokens(
    theta: torch.Tensor, codec: WeightCodec,
) -> torch.Tensor:
    """Convert flat weights to neuron-centric tokens.

    Args:
        theta: ``[B, D]`` (or ``[D]``) flat weight vectors.
        codec: Codec defining ``L``/``H`` and the canonical layout.

    Returns:
        Tokens ``[B, H+1, 2L+1]``. First ``H`` rows are the hidden-neuron
        tokens ``concat(W1[i,:], b1[i], W2[:,i])``; the last row is the bias
        token ``concat(0, 0, b2)``.
    """
    W1, b1, W2, b2 = unpack_batch(theta, codec)
    B, H, L = W1.shape[0], codec.H, codec.L

    # Hidden-neuron tokens: concat(W1[i,:], b1[i], W2[:,i]) -> [B, H, 2L+1].
    tok_hidden = torch.cat(
        [W1, b1.unsqueeze(-1), W2.transpose(1, 2)], dim=-1)  # [B, H, 2L+1]

    # Bias token: concat(0^L, 0, b2) -> [B, 1, 2L+1].
    tok_bias = torch.cat(
        [torch.zeros(B, 1, L, dtype=theta.dtype, device=theta.device),
         torch.zeros(B, 1, 1, dtype=theta.dtype, device=theta.device),
         b2.unsqueeze(1)], dim=-1)  # [B, 1, 2L+1]

    return torch.cat([tok_hidden, tok_bias], dim=1)  # [B, H+1, 2L+1]


def theta_to_svd_tokens(
    theta: torch.Tensor, codec: WeightCodec,
) -> torch.Tensor:
    """Convert flat weights to SVD-triple tokens (ablation, plan section 2.2).

    Each singular triple of ``W1`` (``[H, L]``) and ``W2`` (``[L, H]``) becomes
    a token ``concat(u_k, s_k, v_k)`` of dim ``H + 1 + L = 161``. Biases are
    emitted as zero-padded tokens of the same dim. Singular values are sorted
    descending (``torch.linalg.svd`` default); signs are canonicalized so the
    largest-magnitude entry of each ``u_k`` is positive.

    Args:
        theta: ``[B, D]`` (or ``[D]``) flat weight vectors.
        codec: Codec defining ``L``/``H`` and the canonical layout.

    Returns:
        Tokens ``[B, H + L + 2, H + L + 1]`` = ``[B, 66, 161]``:
        ``H`` triples of W1, ``L`` triples of W2, one b1 token, one b2 token.
    """
    W1, b1, W2, b2 = unpack_batch(theta, codec)
    B, H, L = W1.shape[0], codec.H, codec.L

    def _canon_sign(u: torch.Tensor) -> torch.Tensor:
        """Flip the sign of each u row so its max-magnitude entry is positive."""
        idx = u.abs().argmax(dim=-1, keepdim=True)
        sgn = torch.gather(u, -1, idx).sign().clamp_min(0.0) * 2 - 1
        # sgn is [..., 1] (keepdim); sgn = +1 if entry>0 else -1 (0 -> +1).
        return u * sgn

    U1, s1, Vh1 = torch.linalg.svd(W1, full_matrices=False)
    # Singular vectors are COLUMNS of U1 [B, H, 32]; rows of Vh1 [B, 32, L].
    # Token rows = (u_k, s_k, v_k): transpose U1 to [B, 32, H] and canonicalize
    # the sign per u_k (along its H components).
    U1 = _canon_sign(U1.transpose(1, 2))                    # [B, 32, H]
    tok1 = torch.cat([U1, s1.unsqueeze(-1), Vh1], dim=-1)   # [B, 32, H+1+L]

    U2, s2, Vh2 = torch.linalg.svd(W2, full_matrices=False)
    # U2 [B, L, 32] (u_k columns of length L), Vh2 [B, 32, H] (v_k rows).
    U2 = _canon_sign(U2.transpose(1, 2))                    # [B, 32, L]
    tok2 = torch.cat([U2, s2.unsqueeze(-1), Vh2], dim=-1)   # [B, 32, L+1+H]

    d = H + 1 + L
    tok_b1 = F.pad(b1.unsqueeze(1), (0, d - H))  # [B, 1, d]
    tok_b2 = F.pad(b2.unsqueeze(1), (0, d - L))  # [B, 1, d]

    return torch.cat([tok1, tok2, tok_b1, tok_b2], dim=1)  # [B, 66, 161]


# ---------------------------------------------------------------------------
# Transformer building blocks (pre-LN)
# ---------------------------------------------------------------------------

class _PreLNBlock(nn.Module):
    """Pre-LN transformer block: MHA + FFN with residuals."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class MLPEncoder(nn.Module):
    """Set-transformer encoder over MLP weight tokens.

    No positional encodings: token permutation equivariance is structural (the
    hidden-layer gauge symmetry becomes token permutation). A learned CLS token
    aggregates the set; ``cond = concat(cls, mean_pool(tokens))``.

    Args:
        token_dim: Input token dimension (65 for neuron tokens, 161 for SVD).
        d_model: Transformer width.
        n_layers: Number of pre-LN transformer layers.
        n_heads: Attention heads per layer.
        ffn_hidden: FFN hidden width.
        n_types: Number of token types for type embeddings (default 2).
        dropout: Dropout probability.
    """

    def __init__(self, token_dim: int = NEURON_TOKEN_DIM, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 8, ffn_hidden: int = 1024,
                 n_types: int = N_TOKEN_TYPES, dropout: float = 0.1):
        super().__init__()
        self.token_dim = int(token_dim)
        self.d_model = int(d_model)
        self.proj = nn.Linear(token_dim, d_model)
        self.type_embed = nn.Embedding(n_types, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([
            _PreLNBlock(d_model, n_heads, ffn_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, tokens: torch.Tensor, token_types: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a set of weight tokens.

        Args:
            tokens: ``[B, T, token_dim]`` token features (e.g. neuron tokens).
            token_types: ``[B, T]`` long tensor of token-type ids (e.g. 0 for
                hidden neurons, 1 for the bias token).

        Returns:
            ``enc [B, T+1, d_model]`` (CLS at index 0, tokens after) and
            ``cond [B, 2*d_model]`` = ``concat(enc[:, 0], mean_pool(enc[:, 1:]))``.
        """
        B, T, _ = tokens.shape
        x = self.proj(tokens) + self.type_embed(token_types)  # [B, T, d_model]
        cls = self.cls.expand(B, 1, self.d_model)             # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)                        # [B, T+1, d_model]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]                                     # [B, d_model]
        pooled = x[:, 1:].mean(dim=1)                         # [B, d_model]
        cond = torch.cat([cls_out, pooled], dim=-1)           # [B, 2*d_model]
        return x, cond


def neuron_token_types(
    theta: torch.Tensor, codec: WeightCodec, device: torch.device,
) -> torch.Tensor:
    """Token-type ids for the neuron representation: hidden=0, bias=1.

    Args:
        theta: ``[B, D]`` (used only for batch size).
        codec: Codec (used only for ``H``).
        device: Output device.

    Returns:
        ``[B, H+1]`` long tensor.
    """
    B = theta.shape[0] if theta.dim() == 2 else 1
    types = torch.zeros(B, codec.H + 1, dtype=torch.long, device=device)
    types[:, -1] = 1  # bias token type
    return types


def svd_token_types(
    theta: torch.Tensor, codec: WeightCodec, device: torch.device,
) -> torch.Tensor:
    """Token-type ids for the SVD representation: W1 triples=0, W2 triples=0,
    b1 token=1, b2 token=1.

    The number of singular triples per matrix is ``min(H, L) = L`` (32).

    Args:
        theta: ``[B, D]`` (used only for batch size).
        codec: Codec (used only for ``L``).
        device: Output device.

    Returns:
        ``[B, 2*L+2]`` long tensor.
    """
    B = theta.shape[0] if theta.dim() == 2 else 1
    types = torch.zeros(B, 2 * codec.L + 2, dtype=torch.long, device=device)
    types[:, -2:] = 1  # bias tokens
    return types
