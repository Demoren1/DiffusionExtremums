"""Smoke test for Phase 3: the diffusion hypernetwork architecture.

Verifies the DDPM over the 8352-dim MLP weight space (plan Section 3) without
requiring the actual collected targets (Phase 2 outputs). It uses random
tensors with the correct shapes to exercise:

1. **Instantiation + parameter count**: the diffusion model builds and has a
   small, documented parameter count (assert < 10M; the plan targets ~4.6M).
2. **Config encoder**: a ``DatasetConfig`` (and its dict form) maps to a
   14-dim feature vector and a 128-dim embedding with correct shapes.
3. **Forward pass (training loss)**: on a small batch of random standardized
   weights + random timesteps + config embeddings, the loss is a finite scalar
   with correct shape.
4. **Sampling**: ancestral sampling generates weights of shape ``[B, 8352]``
   with finite values, conditioned on a config.
5. **Weight normalization round-trip**: standardize then destandardize returns
   the original (within float tolerance); constant dims map to 0 in z-space.
6. **End-to-end with a normalizer**: the model with an attached
   ``WeightNormalizer`` accepts raw weights in ``forward`` and returns raw
   (destandardized) weights from ``sample``.

Run with::

    python -m src.smoke.test_diffusion
"""
import sys
from typing import List, Tuple

import numpy as np
import torch

from src.configs.base import DatasetConfig
from src.data.families import FAMILIES, sample_kernel
from src.models.config_encoder import (
    CONFIG_FEATURE_DIM,
    ConfigEncoder,
    config_to_features,
    configs_to_features,
)
from src.models.diffusion import (
    DEFAULT_D,
    DenoiserNet,
    DiffusionModel,
    NoiseSchedule,
)
from src.models.weight_codec import WeightCodec
from src.models.weight_norm import WeightNormalizer
from src.utils.seeding import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A reasonable upper bound on the denoiser size for the "small" hypothesis.
# The plan targets ~4.6M; we allow headroom but assert it stays well under 10M.
MAX_PARAMS = 10_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(family: str = "GAUSS", radius: int = 2, noise_std: float = 0.1,
                n_train: int = 1024, seed: int = 0, L: int = 32,
                n_test: int = 512) -> DatasetConfig:
    """Build a deterministic DatasetConfig with a sampled kernel."""
    rng = np.random.default_rng(seed)
    kernel = sample_kernel(family, radius, rng)
    return DatasetConfig(
        family=family,
        kernel=tuple(float(v) for v in kernel),
        radius=radius,
        noise_std=noise_std,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        L=L,
    )


def make_config_dict(family: str = "GAUSS", radius: int = 2,
                      noise_std: float = 0.1, seed: int = 0) -> dict:
    """Build a config dict (the form stored in configs.json)."""
    cfg = make_config(family=family, radius=radius, noise_std=noise_std, seed=seed)
    return {
        "family": cfg.family,
        "kernel": list(cfg.kernel),
        "radius": cfg.radius,
        "noise_std": cfg.noise_std,
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "seed": cfg.seed,
        "L": cfg.L,
    }


def random_configs(n: int, seed: int = 0) -> List[DatasetConfig]:
    """Generate n configs spanning all families and radii."""
    rng = np.random.default_rng(seed)
    configs: List[DatasetConfig] = []
    for i in range(n):
        family = FAMILIES[i % len(FAMILIES)]
        radius = int(rng.choice((1, 2, 3)))
        noise_std = float(rng.choice((0.0, 0.05, 0.1, 0.2)))
        krng = np.random.default_rng(seed + i + 1)
        kernel = sample_kernel(family, radius, krng)
        configs.append(DatasetConfig(
            family=family,
            kernel=tuple(float(v) for v in kernel),
            radius=radius,
            noise_std=noise_std,
            n_train=1024,
            n_test=512,
            seed=int(rng.integers(0, 2 ** 31 - 1)),
            L=32,
        ))
    return configs


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def test_config_encoder() -> None:
    """Check the config feature vector layout and the encoder embedding."""
    print("\n=== Config encoder ===")
    cfg = make_config(family="DIFF", radius=3, noise_std=0.2)
    cfg_dict = make_config_dict(family="DIFF", radius=3, noise_std=0.2)

    # Feature vector from DatasetConfig and from dict must match.
    feat = config_to_features(cfg)
    feat_dict = config_to_features(cfg_dict)
    assert feat.shape == (CONFIG_FEATURE_DIM,), \
        f"feature shape {feat.shape} != ({CONFIG_FEATURE_DIM},)"
    assert torch.allclose(feat, feat_dict, atol=1e-6), \
        "DatasetConfig and dict forms must produce identical features"
    print(f"  feature dim        : {CONFIG_FEATURE_DIM}  "
          f"(5 family one-hot + 7 kernel + 1 radius + 1 noise_std)")
    print(f"  feature (DIFF,r=3) : {feat.tolist()}")

    # Layout checks: one-hot in [0:5], kernel in [5:12], radius in [12], noise in [13].
    onehot = feat[:5]
    assert onehot.sum().item() == 1.0, "family one-hot must sum to 1"
    assert feat[5:12].shape == (7,), "kernel window must be length 7"
    # Center tap (radius index) lands at slot MAX_RADIUS=3 within the window.
    assert abs(feat[5 + 3].item() - cfg.kernel[cfg.radius]) < 1e-5, \
        "center kernel tap must be at window slot 3"
    assert 0.0 <= feat[12].item() <= 1.0, "radius feature must be in [0,1]"
    assert 0.0 <= feat[13].item() <= 1.0, "noise feature must be in [0,1]"

    # Batch encoding.
    configs = random_configs(4)
    feats = configs_to_features(configs)
    assert feats.shape == (4, CONFIG_FEATURE_DIM), f"batch feats {feats.shape}"

    # Encoder module -> 128-dim embedding.
    enc = ConfigEncoder().to(DEVICE)
    emb = enc(feats.to(DEVICE))
    assert emb.shape == (4, 128), f"embedding shape {emb.shape}"
    assert torch.isfinite(emb).all(), "embedding must be finite"
    print(f"  encoder embedding   : {tuple(emb.shape)}  (B, 128)")
    print("  PASS: config encoder shapes + layout correct")


def test_weight_normalizer() -> None:
    """Check the WeightNormalizer fit / standardize / destandardize round-trip."""
    print("\n=== Weight normalization ===")
    D = DEFAULT_D
    rng = torch.Generator().manual_seed(123)
    # A corpus with some constant dims and some varying dims.
    n = 64
    base = torch.randn(n, D, generator=rng)
    # Make the first 10 dims constant (std 0) to exercise the constant path.
    base[:, :10] = 7.0
    weights = base  # [n, D]

    norm = WeightNormalizer.fit(weights)
    assert norm.D == D
    n_const = int(norm.constant_mask.sum().item())
    assert n_const == 10, f"expected 10 constant dims, got {n_const}"
    print(f"  fitted normalizer   : {norm}")
    print(f"  constant dims        : {n_const}/{D}  (std below floor -> sigma=1, z=0)")

    # Round-trip: standardize then destandardize == original.
    z = norm.standardize(weights)
    rec = norm.destandardize(z)
    assert torch.allclose(rec, weights, atol=1e-4), \
        "standardize/destandardize round-trip failed"
    # Constant dims map to exactly 0 in z-space.
    assert torch.all(z[:, :10] == 0.0), "constant dims must be 0 in z-space"
    # Non-constant dims have ~unit std in z-space.
    z_var = z[:, 10:].var(dim=0, unbiased=False)
    assert z_var.mean().item() < 1.5 and z_var.mean().item() > 0.5, \
        f"non-constant z-space std should be ~1, got var mean {z_var.mean():.3f}"
    print(f"  z-space non-const var: {z_var.mean().item():.4f}  (target ~1.0)")
    print("  PASS: round-trip exact, constant dims -> 0, non-const ~unit var")

    # State dict round-trip.
    norm2 = WeightNormalizer.from_state_dict(norm.state_dict())
    assert torch.allclose(norm2.mu, norm.mu) and torch.allclose(norm2.sigma, norm.sigma)
    assert torch.equal(norm2.constant_mask, norm.constant_mask)
    print("  PASS: state_dict round-trip")


def test_noise_schedule() -> None:
    """Check the noise schedule buffers and forward-process properties."""
    print("\n=== Noise schedule ===")
    sched = NoiseSchedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    assert sched.T == 1000
    assert abs(sched.betas[0].item() - 1e-4) < 1e-9
    assert abs(sched.betas[-1].item() - 0.02) < 1e-9
    # alphas_cumprod is monotonically decreasing in [0,1].
    acp = sched.alphas_cumprod
    assert (acp <= 1.0).all() and (acp >= 0.0).all()
    assert torch.all(acp[1:] <= acp[:-1] + 1e-9), "bar_alpha must be non-increasing"
    # sqrt coefficients are consistent.
    assert torch.allclose(sched.sqrt_alphas_cumprod ** 2, acp, atol=1e-5)
    assert torch.allclose(
        sched.sqrt_one_minus_alphas_cumprod ** 2, 1.0 - acp, atol=1e-5)
    print(f"  T={sched.T}, beta in [{sched.beta_start}, {sched.beta_end}] (linear)")
    print(f"  bar_alpha[0]={acp[0]:.6f}, bar_alpha[-1]={acp[-1]:.6e}")
    print("  PASS: schedule monotone, coefficients consistent")


def test_instantiation_and_params() -> DiffusionModel:
    """Instantiate the diffusion model, print + assert the parameter count."""
    print("\n=== Instantiation + parameter count ===")
    model = DiffusionModel(
        D=DEFAULT_D, num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
        hidden=256, n_blocks=3, embed_dim=128, use_scale=False,
    ).to(DEVICE)
    n = model.n_params()
    print(f"  DiffusionModel params : {n:,}  ({n / 1e6:.3f}M)")
    print("  Per-submodule breakdown:")
    for name, cnt in model.param_breakdown().items():
        if name != "total":
            print(f"    {name:<14}: {cnt:>10,}")
    print(f"    {'total':<14}: {model.param_breakdown()['total']:>10,}")
    assert n < MAX_PARAMS, f"param count {n} >= {MAX_PARAMS} (model not small!)"
    print(f"  PASS: param count {n:,} < {MAX_PARAMS:,} (small meta-learner)")
    return model


def test_forward_loss(model: DiffusionModel) -> None:
    """Run a forward pass (training loss) on random data; check shapes/finite."""
    print("\n=== Forward pass (training loss) ===")
    B, D = 8, model.D
    model.train()
    # Random standardized weights (z_0 ~ N(0,1)) — no normalizer attached.
    z_0 = torch.randn(B, D, device=DEVICE)
    t = torch.randint(0, model.T, (B,), device=DEVICE)
    configs = random_configs(B)
    cond_emb = model.denoiser.encode_configs(configs).to(DEVICE)
    assert cond_emb.shape == (B, 128), f"cond_emb {cond_emb.shape}"

    loss = model(z_0, t, cond_emb)
    assert loss.dim() == 0, f"loss must be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss).item(), "loss must be finite"
    # Backward pass works (gradients flow).
    loss.backward()
    gnorm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gnorm += float(p.grad.detach().norm().item() ** 2)
    gnorm = gnorm ** 0.5
    print(f"  batch={B}, D={D}, t in [0,{model.T})")
    print(f"  loss = {loss.item():.6f}  (finite, scalar)")
    print(f"  grad norm = {gnorm:.6f}  (backward OK)")
    assert gnorm > 0.0, "gradients must be non-zero"
    print("  PASS: forward loss finite, scalar, gradients flow")


def test_forward_with_normalizer() -> None:
    """Forward pass with an attached normalizer accepts RAW weights."""
    print("\n=== Forward pass with attached WeightNormalizer ===")
    D = DEFAULT_D
    rng = torch.Generator().manual_seed(7)
    weights = torch.randn(32, D, generator=rng)
    weights[:, :5] = 3.0  # some constant dims
    norm = WeightNormalizer.fit(weights).to(DEVICE)
    model = DiffusionModel(
        D=D, num_timesteps=1000, hidden=256, n_blocks=3, embed_dim=128,
        normalizer=norm,
    ).to(DEVICE)
    model.train()
    B = 6
    raw = torch.randn(B, D, device=DEVICE)
    t = torch.randint(0, model.T, (B,), device=DEVICE)
    configs = random_configs(B)
    cond_emb = model.denoiser.encode_configs(configs).to(DEVICE)
    loss = model(raw, t, cond_emb)
    assert torch.isfinite(loss).item(), "loss with normalizer must be finite"
    print(f"  loss (raw weights in) = {loss.item():.6f}  (finite)")
    print("  PASS: normalizer-attached forward accepts raw weights")


def test_sampling(model: DiffusionModel) -> None:
    """Run ancestral sampling; check output shape and finiteness."""
    print("\n=== Sampling (ancestral DDPM) ===")
    model.eval()
    B = 4
    configs = random_configs(B)
    cond_emb = model.denoiser.encode_configs(configs).to(DEVICE)
    # Full 1000-step sampling.
    samples = model.sample(cond_emb, device=DEVICE)
    assert samples.shape == (B, model.D), \
        f"sample shape {samples.shape} != ({B}, {model.D})"
    assert torch.isfinite(samples).all(), "samples must be finite"
    print(f"  full 1000-step sample: {tuple(samples.shape)}  (B, D)")
    print(f"  sample mean={samples.mean().item():.4f}, "
          f"std={samples.std().item():.4f}, "
          f"range=[{samples.min().item():.3f}, {samples.max().item():.3f}]")
    print("  PASS: sampling output [B, 8352], finite")

    # Batched generation via sample_configs (encode + sample + destandardize).
    # Without a normalizer this returns standardized-space samples.
    samples2 = model.sample_configs(configs, device=DEVICE)
    assert samples2.shape == (B, model.D)
    assert torch.isfinite(samples2).all()
    print(f"  sample_configs()     : {tuple(samples2.shape)}  (batched generation)")
    print("  PASS: batched generation works")


def test_sampling_with_normalizer() -> None:
    """Sampling with an attached normalizer applies destandardize (raw output).

    Verifies the destandardize wiring precisely with controlled normalizers and
    a fixed RNG seed: the normalizer only acts at the *end* of sampling, so the
    standardized-space trajectory is identical across normalizers for the same
    seed. Therefore:
      - identity normalizer (mu=0, sigma=1)  -> output == standardized output
      - shift normalizer   (mu=c, sigma=1)  -> output == standardized + c
      - scale normalizer   (mu=0, sigma=s) -> output == s * standardized

    (We do NOT assert that constant dims destandardize to mu: that only holds
    once the *trained* model learns to produce z~0 on constant dims. An
    untrained denoiser outputs large random values, so the raw output is
    sigma*z + mu with z far from 0. The destandardize transform itself is
    validated by the round-trip test above.)
    """
    print("\n=== Sampling with attached WeightNormalizer ===")
    D = DEFAULT_D
    # Use a short schedule for this wiring check (faster; we only test the
    # destandardize application, not the full 1000-step process).
    model = DiffusionModel(
        D=D, num_timesteps=50, hidden=256, n_blocks=3, embed_dim=128,
    ).to(DEVICE)
    model.eval()
    B = 2
    configs = random_configs(B)
    cond_emb = model.denoiser.encode_configs(configs).to(DEVICE)

    # 1. No normalizer: output is in standardized space.
    torch.manual_seed(42)
    z_std = model.sample(cond_emb, device=DEVICE)
    assert z_std.shape == (B, D) and torch.isfinite(z_std).all()

    # 2. Identity normalizer (mu=0, sigma=1): output must equal standardized.
    ident = WeightNormalizer(D=D, mu=torch.zeros(D), sigma=torch.ones(D)).to(DEVICE)
    model.normalizer = ident
    torch.manual_seed(42)
    raw_ident = model.sample(cond_emb, device=DEVICE)
    assert torch.allclose(raw_ident, z_std, atol=1e-1, rtol=1e-3), \
        "identity normalizer must not change the output"
    model.normalizer = None

    # 3. Shift normalizer (mu=c, sigma=1): output == standardized + c.
    c = 5.0
    shift = WeightNormalizer(D=D, mu=torch.full((D,), c),
                             sigma=torch.ones(D)).to(DEVICE)
    model.normalizer = shift
    torch.manual_seed(42)
    raw_shift = model.sample(cond_emb, device=DEVICE)
    assert torch.allclose(raw_shift, z_std + c, atol=1e-1, rtol=1e-3), \
        "shift normalizer must add mu to the output"
    model.normalizer = None

    # 4. Scale normalizer (mu=0, sigma=s): output == s * standardized.
    s = 2.0
    scale = WeightNormalizer(D=D, mu=torch.zeros(D),
                             sigma=torch.full((D,), s)).to(DEVICE)
    model.normalizer = scale
    torch.manual_seed(42)
    raw_scale = model.sample(cond_emb, device=DEVICE)
    assert torch.allclose(raw_scale, s * z_std, atol=1e-1, rtol=1e-3), \
        "scale normalizer must multiply the output by sigma"
    model.normalizer = None

    print(f"  identity normalizer : output == standardized space (OK)")
    print(f"  shift normalizer    : output == standardized + {c} (OK)")
    print(f"  scale normalizer    : output == {s} * standardized (OK)")
    print("  PASS: destandardize wiring verified (mu shift + sigma scale)")


def test_generated_weights_instantiate() -> None:
    """Generated weights can be instantiated into an MLP via WeightCodec."""
    print("\n=== Generated weights -> MLP instantiation ===")
    D = DEFAULT_D
    model = DiffusionModel(
        D=D, num_timesteps=1000, hidden=256, n_blocks=3, embed_dim=128,
    ).to(DEVICE)
    model.eval()
    configs = random_configs(2)
    samples = model.sample_configs(configs, device=DEVICE)
    codec = WeightCodec(L=32, H=128)
    assert codec.D == D
    for i in range(samples.shape[0]):
        theta = samples[i].cpu().float()
        mlp = codec.instantiate(theta)
        x = torch.randn(4, 32)
        with torch.no_grad():
            y = mlp(x)
        assert y.shape == (4, 32), f"MLP output {y.shape}"
        assert torch.isfinite(y).all(), "instantiated MLP output must be finite"
    print(f"  generated {samples.shape[0]} weight vectors -> MLPs, "
          f"forward output [4, 32] finite")
    print("  PASS: generated weights instantiate into working MLPs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the Phase 3 smoke test and assert the validation gates.

    Returns 0 on success, 1 on failure.
    """
    set_seed(0)
    print("=" * 90)
    print("Phase 3 smoke test: diffusion hypernetwork architecture (DDPM over weights)")
    print("=" * 90)
    print(f"Device: {DEVICE}  (cuda available: {torch.cuda.is_available()})")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    all_ok = True
    checks = [
        ("config encoder", test_config_encoder),
        ("weight normalizer", test_weight_normalizer),
        ("noise schedule", test_noise_schedule),
    ]
    for name, fn in checks:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL [{name}]: {e}")
            all_ok = False

    # Instantiation + param count (returns the model for reuse).
    model = None
    try:
        model = test_instantiation_and_params()
    except AssertionError as e:
        print(f"  FAIL [instantiation/params]: {e}")
        all_ok = False

    if model is not None:
        for name, fn in [
            ("forward loss", test_forward_loss),
            ("sampling", test_sampling),
        ]:
            try:
                fn(model)
            except AssertionError as e:
                print(f"  FAIL [{name}]: {e}")
                all_ok = False

    for name, fn in [
        ("forward with normalizer", test_forward_with_normalizer),
        ("sampling with normalizer", test_sampling_with_normalizer),
        ("generated weights instantiate", test_generated_weights_instantiate),
    ]:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL [{name}]: {e}")
            all_ok = False

    print("\n" + "=" * 90)
    if all_ok:
        n = model.n_params() if model is not None else -1
        print("PHASE 3 SMOKE TEST PASSED.")
        print(f"  Diffusion model parameter count: {n:,} ({n / 1e6:.3f}M)  "
              f"[< {MAX_PARAMS:,}]")
        print("  Config feature layout: 14-dim = 5 family one-hot + 7 kernel "
              "(center-aligned) + 1 radius + 1 noise_std")
        print("  Noise schedule: linear beta in [1e-4, 0.02], T=1000")
        print("  Forward loss finite; sampling -> [B, 8352] finite; "
              "weight norm round-trips; generated weights instantiate into MLPs.")
        return 0
    print("PHASE 3 SMOKE TEST FAILED: one or more validation gates did not pass.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
