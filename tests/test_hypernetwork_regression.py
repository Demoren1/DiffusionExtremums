"""Regression tests for dead-code removal in src/models/hypernetwork.py (task 1.1).

After removing LowRankWeightDecoder, _FactorHead, Hypernetwork, assemble_weights and
batched_pack, only functional_forward remains. DatasetHypernet (the sole importer)
depends on functional_forward. These tests verify:

1. The import chain is intact (functional_forward pulls in mlp_encoder + weight_codec).
2. functional_forward is functionally equivalent to MLPModel.forward — this is the
   real "no regression" guarantee: the differentiated forward still computes the MLP.
3. DatasetHypernet.compute_loss runs end-to-end and returns a scalar MSE.
"""
import torch

from src.models.dataset_hypernet import DatasetEncoder, DatasetHypernet, WeightDecoder
from src.models.hypernetwork import functional_forward
from src.models.mlp import MLPModel
from src.models.weight_codec import WeightCodec


def _packed_theta(seed: int, L: int = 32, H: int = 16) -> torch.Tensor:
    torch.manual_seed(seed)
    mlp = MLPModel(L=L, H=H)
    codec = WeightCodec(L=L, H=H)
    return codec.pack_model(mlp), mlp, codec


def test_import_chain_intact():
    """DatasetHypernet (the sole consumer) imports cleanly; functional_forward present."""
    # If hypernetwork.py or its dependencies failed to import, this module itself
    # would already have errored at collection. Assert the public symbol exists and
    # is callable.
    assert callable(functional_forward)


def test_functional_forward_matches_mlp_forward():
    """functional_forward(theta, x) == MLPModel(x) for a randomly initialized MLP."""
    L, H, n = 32, 16, 7
    theta, mlp, _codec = _packed_theta(seed=7, L=L, H=H)
    x = torch.randn(n, L)  # unbatched samples [n, L]

    with torch.no_grad():
        y_model = mlp(x)  # [n, L]

    # functional_forward expects a batched theta [B, D]; promote to a single batch.
    y_func = functional_forward(theta.unsqueeze(0), x, L=L, H=H)

    assert y_func.shape == (1, n, L), f"unexpected shape {tuple(y_func.shape)}"
    assert torch.allclose(y_func.squeeze(0), y_model, atol=1e-6), (
        "functional_forward output diverged from MLPModel.forward")


def test_functional_forward_gradients_flow():
    """functional_forward is differentiable w.r.t. theta (needed for compute_loss)."""
    L, H, n = 32, 16, 7
    theta, _mlp, _codec = _packed_theta(seed=3, L=L, H=H)
    theta = theta.detach().clone().requires_grad_(True)
    x = torch.randn(n, L)
    y = torch.randn(n, L)

    y_hat = functional_forward(theta.unsqueeze(0), x, L=L, H=H)
    loss = torch.nn.functional.mse_loss(y_hat.squeeze(0), y)
    loss.backward()

    assert theta.grad is not None, "no gradient flowed to theta"
    assert theta.grad.shape == theta.shape
    assert bool(torch.isfinite(theta.grad).all()), "gradient contains non-finite values"
    assert float(theta.grad.abs().sum()) > 0.0, "gradient is all zeros"


def test_dataset_hypernet_compute_loss_scalar():
    """Full DatasetHypernet pipeline runs and returns a scalar finite MSE."""
    L = 32
    enc = DatasetEncoder(L=L, K_enc=8, d_model=16, d_emb=16,
                         n_layers=1, n_heads=2)
    # D = 2*L*H + H + L with H=16
    H = 16
    dec = WeightDecoder(d_emb=16, D=2 * L * H + H + L, hidden=(32,))
    net = DatasetHypernet(encoder=enc, decoder=dec, mlp_hidden=H)

    B, K, N = 2, 8, 16
    x_enc = torch.randn(B, K, L)
    y_enc = torch.randn(B, K, L)
    x_loss = torch.randn(B, N, L)
    y_loss = torch.randn(B, N, L)

    loss = net.compute_loss(x_enc, y_enc, x_loss, y_loss)

    # compute_loss must return a plain finite tensor (loss is reduced to a scalar);
    # if the batch dimension leaked, it would be a vector.
    assert torch.isfinite(loss).all(), "loss is not finite"
    assert loss.shape == torch.Size([]), f"loss should be scalar, got {tuple(loss.shape)}"


def test_functional_forward_shape_validation():
    """Batched theta of wrong dimension raises a clear error (unpack validates D)."""
    bad_theta = torch.zeros(1, 100)  # wrong D
    x = torch.randn(3, 32)
    try:
        functional_forward(bad_theta, x, L=32, H=16)
    except ValueError as e:
        assert "expected" in str(e).lower() or "dim" in str(e).lower()
        return
    raise AssertionError("expected ValueError for malformed theta")
