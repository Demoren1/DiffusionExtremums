"""Smoke test for Phase 2: MLP target collection (Approach B).

Verifies that the converged MLP weights collected by the Phase 2 pipeline are
good regression targets. Specifically, on a small subset of datasets it checks:

(a) **Convergence**: MLPs reach low train MSE (the optimization actually
    converges, not stuck at a high loss).
(b) **Generalization**: with n_train=1024 >> L=32, the converged MLPs generalize
    well — test MSE is close to the convolution/oracle baseline. This validates
    that the targets are *good solutions* (unlike the small-n_train Phase 1 case
    where the MLP overfits). The conv baseline (learned kernel) and the oracle
    conv (true kernel) bound the achievable test MSE.
(c) **Weight diversity**: different random initializations converge to
    *different* weight vectors (pairwise L2 distance / per-dim variance well
    above zero). This is the gauge freedom of the (W1, W2) factorization and is
    what gives the target collection a rich weight distribution.
(d) **Well-behaved weights**: the weight vectors are finite and have a
    reasonable scale (norm).

It also exercises the full collection pipeline end-to-end on a tiny corpus and
confirms the saved files (weights.pt, losses.pt, configs.json, metadata.json)
load back correctly and round-trip through the ``WeightCodec``.

Run with::

    python -m src.smoke.test_mlp_targets
"""
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from src.data.dataset import DatasetFamily
from src.data.families import FAMILIES, sample_kernel
from src.models.mlp import MLPModel
from src.models.weight_codec import WeightCodec
from src.smoke.models import Conv1dModel
from src.training.collect_targets import CollectConfig, collect_targets
from src.training.train_mlp import TrainConfig, train_mlp_to_convergence
from src.utils.seeding import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers (mirror the Phase 1 smoke test's dataset/conv helpers)
# ---------------------------------------------------------------------------

def make_config(family: str, radius: int, noise_std: float, n_train: int,
                seed: int, L: int = 32, n_test: int = 512):
    """Sample a kernel for ``family`` and build a deterministic DatasetConfig."""
    from src.configs.base import DatasetConfig
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


def oracle_test_mse(x_test_np: np.ndarray, y_test_np: np.ndarray,
                    kernel_np: np.ndarray) -> float:
    """Test MSE of the *true* kernel (oracle conv). With noise_std=0 this is ~0."""
    from src.data.dataset import conv1d_same
    y_pred = conv1d_same(x_test_np, kernel_np)
    return float(np.mean((y_pred - y_test_np) ** 2))


def train_conv_baseline(x_train, y_train, x_test, y_test, K, steps=2000,
                       lr=3e-3) -> Tuple[float, float]:
    """Train a 1D conv baseline (learned kernel) and return (train_mse, test_mse)."""
    conv = Conv1dModel(L=32, kernel_size=K, bias=False).to(DEVICE)
    opt = torch.optim.AdamW(conv.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt = x_train.to(DEVICE)
    yt = y_train.to(DEVICE)
    xe = x_test.to(DEVICE)
    ye = y_test.to(DEVICE)
    for _ in range(steps):
        conv.train()
        opt.zero_grad()
        loss = loss_fn(conv(xt), yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(conv.parameters(), 1.0)
        opt.step()
    conv.eval()
    with torch.no_grad():
        tr = loss_fn(conv(xt), yt).item()
        te = loss_fn(conv(xe), ye).item()
    return tr, te


# ---------------------------------------------------------------------------
# Per-dataset result record
# ---------------------------------------------------------------------------

@dataclass
class TargetResult:
    family: str
    radius: int
    n_train: int
    noise_std: float
    dataset_id: str
    mlp_train: float       # mean over inits
    mlp_val: float
    mlp_test: float        # mean over inits
    mlp_test_std: float    # std over inits
    conv_test: float       # learned-conv baseline
    oracle_test: float     # true-kernel oracle
    weight_norm: float     # mean L2 norm of converged weight vectors
    pairwise_dist: float   # mean pairwise L2 distance between inits
    weight_var: float      # mean per-dim variance across inits
    n_mlp: int
    n_steps_mean: float


# ---------------------------------------------------------------------------
# Core smoke test: train N MLPs per dataset, collect weights, check properties
# ---------------------------------------------------------------------------

def run_target_smoke(
    families: Tuple[str, ...] = ("MA", "DIFF", "GAUSS", "MATCH", "RAND"),
    radii: Tuple[int, ...] = (2,),
    n_train: int = 1024,
    noise_std: float = 0.1,
    seeds: Tuple[int, ...] = (0, 1),
    n_mlp: int = 4,
    steps: int = 3000,
    L: int = 32,
    n_test: int = 512,
) -> List[TargetResult]:
    """Train n_mlp MLPs per dataset, collect weights, and record metrics.

    Returns a list of ``TargetResult`` (one per dataset).
    """
    results: List[TargetResult] = []
    sampler = DatasetFamily(n_test=n_test, L=L)
    codec = WeightCodec(L=L, H=128)

    for family in families:
        for radius in radii:
            K = 2 * radius + 1
            for seed in seeds:
                cfg = make_config(family, radius, noise_std, n_train, seed,
                                  L=L, n_test=n_test)
                inst = sampler.sample_dataset(cfg)

                # Train n_mlp MLPs with different inits.
                thetas = torch.zeros(n_mlp, codec.D, dtype=torch.float32)
                train_ms, val_ms, test_ms, n_steps = [], [], [], []
                for j in range(n_mlp):
                    tcfg = TrainConfig(
                        lr=3e-3, steps=steps, min_steps=300, patience=200,
                        L=L, H=128, seed=10_000 + j, device="auto",
                    )
                    res = train_mlp_to_convergence(
                        inst.x_train, inst.y_train, inst.x_test, inst.y_test,
                        config=tcfg)
                    thetas[j] = res.theta
                    train_ms.append(res.train_mse)
                    val_ms.append(res.val_mse)
                    test_ms.append(res.test_mse)
                    n_steps.append(res.n_steps)

                # Conv baseline (learned kernel) + oracle (true kernel).
                conv_tr, conv_te = train_conv_baseline(
                    inst.x_train, inst.y_train, inst.x_test, inst.y_test, K)
                oracle = oracle_test_mse(
                    inst.x_test.numpy(), inst.y_test.numpy(), inst.kernel.numpy())

                # Weight diversity metrics.
                norms = thetas.norm(dim=1)
                # Mean pairwise L2 distance over all init pairs.
                if n_mlp >= 2:
                    diffs = torch.cdist(thetas, thetas, p=2)
                    n_pairs = n_mlp * (n_mlp - 1)
                    pairwise = diffs.sum().item() / n_pairs
                    # Mean per-dim variance across inits (averaged over dims).
                    var = thetas.var(dim=0, unbiased=True).mean().item()
                else:
                    pairwise = 0.0
                    var = 0.0

                results.append(TargetResult(
                    family=family, radius=radius, n_train=n_train,
                    noise_std=noise_std, dataset_id=inst.dataset_id,
                    mlp_train=float(np.mean(train_ms)),
                    mlp_val=float(np.mean(val_ms)),
                    mlp_test=float(np.mean(test_ms)),
                    mlp_test_std=float(np.std(test_ms)),
                    conv_test=conv_te, oracle_test=oracle,
                    weight_norm=float(norms.mean()),
                    pairwise_dist=pairwise,
                    weight_var=var,
                    n_mlp=n_mlp,
                    n_steps_mean=float(np.mean(n_steps)),
                ))
    return results


def print_table(results: List[TargetResult]) -> None:
    """Print a per-dataset summary table."""
    header = (f"{'family':<6} {'r':>2} {'n_tr':>5} {'n_mlp':>5} "
              f"{'| mlp_train':>10} {'mlp_test':>10} {'conv_test':>10} "
              f"{'oracle':>10} {'| test/conv':>10} {'test/oracle':>11} "
              f"{'| w_norm':>9} {'pair_dist':>10} {'w_var':>10} {'steps':>7}")
    print(header)
    print("-" * len(header))
    for r in results:
        K = 2 * r.radius + 1
        tc = r.mlp_test / r.conv_test if r.conv_test > 0 else float("nan")
        to = r.mlp_test / r.oracle_test if r.oracle_test > 0 else float("nan")
        print(f"{r.family:<6} {r.radius:>2} {r.n_train:>5} {r.n_mlp:>5} "
              f"| {r.mlp_train:>10.6f} {r.mlp_test:>10.6f} {r.conv_test:>10.6f} "
              f"{r.oracle_test:>10.6f} | {tc:>10.3f} {to:>11.3f} "
              f"| {r.weight_norm:>9.3f} {r.pairwise_dist:>10.3f} "
              f"{r.weight_var:>10.3e} {r.n_steps_mean:>7.0f}")
    print("-" * len(header))

    # Aggregate summary.
    mean_mlp = np.mean([r.mlp_test for r in results])
    mean_conv = np.mean([r.conv_test for r in results])
    mean_oracle = np.mean([r.oracle_test for r in results])
    mean_pair = np.mean([r.pairwise_dist for r in results])
    mean_var = np.mean([r.weight_var for r in results])
    print(f"\nSummary over {len(results)} datasets (n_mlp={results[0].n_mlp}):")
    print(f"  mean mlp   test  : {mean_mlp:.6f}")
    print(f"  mean conv  test  : {mean_conv:.6f}  (learned-kernel baseline)")
    print(f"  mean oracle test : {mean_oracle:.6f}  (true-kernel lower bound)")
    print(f"  mean mlp/conv    : {mean_mlp / mean_conv:.3f}")
    print(f"  mean pairwise dist between inits : {mean_pair:.3f}")
    print(f"  mean per-dim weight variance     : {mean_var:.3e}")


# ---------------------------------------------------------------------------
# End-to-end pipeline test: run collect_targets on a tiny corpus, load back
# ---------------------------------------------------------------------------

def test_pipeline_roundtrip() -> str:
    """Run the full collection pipeline on a tiny corpus and verify the files.

    Returns the temp output directory (kept for inspection; caller may remove).
    """
    print("\n=== End-to-end pipeline test (tiny corpus) ===")
    out_dir = tempfile.mkdtemp(prefix="mlp_targets_smoke_")
    cfg = CollectConfig(
        n_datasets=4,
        n_mlp=2,
        n_train=1024,
        n_test=256,
        noise_std=0.1,
        corpus_seed=7,
        out_dir=out_dir,
        train=TrainConfig(lr=3e-3, steps=1500, min_steps=200, patience=150,
                          L=32, H=128, device="auto"),
    )
    collect_targets(cfg, show_progress=True)

    # Verify all expected files exist.
    for fn in ("weights.pt", "losses.pt", "configs.json",
               "dataset_ids.json", "metadata.json"):
        path = os.path.join(out_dir, fn)
        assert os.path.exists(path), f"missing output file: {fn}"

    # Load and check shapes / round-trip.
    weights = torch.load(os.path.join(out_dir, "weights.pt"))
    losses = torch.load(os.path.join(out_dir, "losses.pt"))
    with open(os.path.join(out_dir, "configs.json")) as f:
        configs = json.load(f)
    with open(os.path.join(out_dir, "metadata.json")) as f:
        meta = json.load(f)

    codec = WeightCodec(L=32, H=128)
    assert weights.shape == (4, 2, codec.D), f"weights shape {weights.shape}"
    assert losses.shape == (4, 2, 3), f"losses shape {losses.shape}"
    assert len(configs) == 4
    assert meta["weight_vectorization"]["D"] == codec.D
    assert meta["mlp_architecture"]["n_params"] == codec.D

    # Round-trip: instantiate a weight vector and check it produces finite output.
    theta = weights[0, 0]
    assert torch.isfinite(theta).all(), "weight vector has non-finite entries"
    model = codec.instantiate(theta)
    x = torch.randn(8, 32)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all(), "instantiated MLP produced non-finite output"
    assert y.shape == (8, 32)

    # Verify the flatten order is exactly the canonical order (pack == unpack).
    params = codec.unpack(theta)
    theta2 = codec.pack(params)
    assert torch.allclose(theta, theta2, atol=1e-6), "pack/unpack round-trip failed"

    print(f"  weights shape : {tuple(weights.shape)}  (n_datasets, n_mlp, D={codec.D})")
    print(f"  losses  shape : {tuple(losses.shape)}  (train_mse, val_mse, test_mse)")
    print(f"  configs       : {len(configs)} dataset configs (conditioning inputs)")
    print(f"  metadata      : format_version={meta['format_version']}, "
          f"order={meta['weight_vectorization']['order']}")
    print(f"  round-trip    : pack/unpack OK, instantiate -> finite output OK")
    print(f"  out_dir       : {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the Phase 2 smoke test and assert the validation gates.

    Returns 0 on success, 1 on failure.
    """
    set_seed(0)
    print("=" * 90)
    print("Phase 2 smoke test: MLP target collection (Approach B)")
    print("=" * 90)
    print(f"Device: {DEVICE}  (cuda available: {torch.cuda.is_available()})")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 1. Core smoke test: convergence, generalization, weight diversity.
    print("\n=== Convergence / generalization / weight diversity ===")
    results = run_target_smoke(
        families=FAMILIES, radii=(2,), n_train=1024, noise_std=0.1,
        seeds=(0, 1), n_mlp=4, steps=3000)
    print_table(results)

    # 2. End-to-end pipeline test (tiny corpus, save + load back).
    out_dir = test_pipeline_roundtrip()

    # 3. Assertions (validation gate).
    print("\n=== Assertions (validation gate) ===")
    all_ok = True

    # (a) Convergence: low train MSE (well below the noise floor of 0.1^2=0.01
    #     is not required, but train MSE should be small relative to test).
    max_train = max(r.mlp_train for r in results)
    ok_conv = max_train < 0.01  # train MSE < noise variance
    print(f"  (a) convergence: max train MSE = {max_train:.6f} < 0.01 "
          f"-> {'PASS' if ok_conv else 'FAIL'}")
    all_ok &= ok_conv

    # (b) Generalization: MLP test MSE close to conv baseline. With n_train=1024
    #     the MLP should generalize at least as well as the learned conv (the
    #     system is over-determined). We require mean mlp_test <= 1.5 * mean conv.
    mean_mlp = float(np.mean([r.mlp_test for r in results]))
    mean_conv = float(np.mean([r.conv_test for r in results]))
    mean_oracle = float(np.mean([r.oracle_test for r in results]))
    ok_gen = mean_mlp <= 1.5 * mean_conv
    print(f"  (b) generalization: mean mlp_test = {mean_mlp:.6f} <= "
          f"1.5 * mean conv_test = {1.5 * mean_conv:.6f} "
          f"-> {'PASS' if ok_gen else 'FAIL'}")
    print(f"      (reference: mean oracle_test = {mean_oracle:.6f})")
    all_ok &= ok_gen

    # (c) Weight diversity: different inits give different weight vectors.
    #     Require mean pairwise distance > 0 and per-dim variance > 0.
    mean_pair = float(np.mean([r.pairwise_dist for r in results]))
    mean_var = float(np.mean([r.weight_var for r in results]))
    ok_div = mean_pair > 1.0 and mean_var > 1e-8
    print(f"  (c) weight diversity: mean pairwise dist = {mean_pair:.3f} > 1.0, "
          f"mean per-dim var = {mean_var:.3e} > 1e-8 "
          f"-> {'PASS' if ok_div else 'FAIL'}")
    all_ok &= ok_div

    # (d) Well-behaved weights: finite, reasonable scale.
    all_finite = all(torch.isfinite(torch.tensor(r.weight_norm)).item()
                     for r in results)
    norms = [r.weight_norm for r in results]
    ok_scale = all(0.1 < n < 1000.0 for n in norms)
    print(f"  (d) well-behaved: all finite = {all_finite}, "
          f"weight norms in [0.1, 1000] = {ok_scale} "
          f"(range [{min(norms):.3f}, {max(norms):.3f}]) "
          f"-> {'PASS' if (all_finite and ok_scale) else 'FAIL'}")
    all_ok &= all_finite and ok_scale

    # Clean up the temp pipeline output (optional; keep for inspection).
    try:
        shutil.rmtree(out_dir)
    except OSError:
        pass

    print("\n" + "=" * 90)
    if all_ok:
        print("PHASE 2 SMOKE TEST PASSED: MLPs converge, generalize well (test MSE "
              "close to conv baseline), and different inits give diverse weights.")
        print("The collected weight vectors are valid regression targets.")
        return 0
    print("PHASE 2 SMOKE TEST FAILED: one or more validation gates did not pass.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
