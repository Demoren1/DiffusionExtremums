"""Smoke test: conv1d vs from-scratch MLP on the generated 1D datasets."""
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from src.configs.base import DatasetConfig
from src.data.dataset import DatasetFamily, conv1d_same
from src.data.families import FAMILIES, sample_kernel
from src.data.registry import dataset_id_from_config
from src.smoke.models import Conv1dModel, MLPModel
from src.utils.seeding import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    steps: int = 2000,
    lr: float = 3e-3,
    weight_decay: float = 0.0,
    device: torch.device = DEVICE,
) -> Tuple[float, float]:
    """Train with AdamW + full-batch MSE; return (train_mse, test_mse)."""
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(steps):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        train_mse = loss_fn(model(x_train), y_train).item()
        test_mse = loss_fn(model(x_test), y_test).item()
    return train_mse, test_mse


def make_config(family: str, radius: int, noise_std: float, n_train: int,
                seed: int, L: int = 32, n_test: int = 512) -> DatasetConfig:
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
    y_pred = conv1d_same(x_test_np, kernel_np)
    return float(np.mean((y_pred - y_test_np) ** 2))


@dataclass
class Result:
    family: str
    radius: int
    n_train: int
    noise_std: float
    seed: int
    dataset_id: str
    conv_train: float
    conv_test: float
    mlp_train: float
    mlp_test: float
    oracle_test: float
    conv_params: int
    mlp_params: int

    @property
    def conv_wins(self) -> bool:
        return self.conv_test < self.mlp_test


def run_smoke_test(
    families: Tuple[str, ...] = FAMILIES,
    radii: Tuple[int, ...] = (2,),
    n_trains: Tuple[int, ...] = (16, 32, 64),
    noise_std: float = 0.1,
    seeds: Tuple[int, ...] = (0, 1, 2),
    steps: int = 1500,
    L: int = 32,
    n_test: int = 512,
) -> List[Result]:
    results: List[Result] = []
    family_sampler = DatasetFamily(n_test=n_test, L=L)

    for family in families:
        for radius in radii:
            K = 2 * radius + 1
            for n_train in tqdm(n_trains):
                for seed in seeds:
                    cfg = make_config(family, radius, noise_std, n_train,
                                      seed=seed, L=L, n_test=n_test)
                    inst = family_sampler.sample_dataset(cfg)

                    set_seed(1234)
                    conv = Conv1dModel(L=L, kernel_size=K, bias=False)
                    set_seed(1234)
                    mlp = MLPModel(L=L, H=128)

                    conv_train, conv_test = train_model(
                        conv, inst.x_train, inst.y_train,
                        inst.x_test, inst.y_test, steps=steps)
                    mlp_train, mlp_test = train_model(
                        mlp, inst.x_train, inst.y_train,
                        inst.x_test, inst.y_test, steps=steps)

                    oracle = oracle_test_mse(
                        inst.x_test.numpy(), inst.y_test.numpy(),
                        inst.kernel.numpy())

                    results.append(Result(
                        family=family, radius=radius, n_train=n_train,
                        noise_std=noise_std, seed=seed,
                        dataset_id=inst.dataset_id,
                        conv_train=conv_train, conv_test=conv_test,
                        mlp_train=mlp_train, mlp_test=mlp_test,
                        oracle_test=oracle,
                        conv_params=conv.n_params(),
                        mlp_params=mlp.n_params(),
                    ))
    return results


def print_table(results: List[Result]) -> None:
    header = (f"{'family':<6} {'r':>2} {'K':>2} {'n_tr':>4} {'seed':>4} "
              f"{'| conv_test':>10} {'mlp_test':>10} {'oracle':>10} "
              f"{'| conv/mlp':>9} {'conv_wins':>9}")
    print(header)
    print("-" * len(header))
    for r in results:
        K = 2 * r.radius + 1
        ratio = r.conv_test / r.mlp_test if r.mlp_test > 0 else float("nan")
        print(f"{r.family:<6} {r.radius:>2} {K:>2} {r.n_train:>4} {r.seed:>4} "
              f"| {r.conv_test:>10.6f} {r.mlp_test:>10.6f} {r.oracle_test:>10.6f} "
              f"| {ratio:>9.3f} {'YES' if r.conv_wins else 'NO':>9}")
    print("-" * len(header))

    conv_wins = sum(r.conv_wins for r in results)
    mean_conv = np.mean([r.conv_test for r in results])
    mean_mlp = np.mean([r.mlp_test for r in results])
    mean_oracle = np.mean([r.oracle_test for r in results])
    print(f"\nSummary over {len(results)} datasets:")
    print(f"  conv wins        : {conv_wins}/{len(results)} "
          f"({100.0 * conv_wins / len(results):.1f}%)")
    print(f"  mean conv  test  : {mean_conv:.6f}")
    print(f"  mean mlp   test  : {mean_mlp:.6f}")
    print(f"  mean oracle test : {mean_oracle:.6f}")
    print(f"  mean conv/mlp    : {mean_conv / mean_mlp:.3f}")

    print("\nPer-family mean test MSE (conv vs mlp):")
    print(f"  {'family':<6} {'conv':>10} {'mlp':>10} {'conv/mlp':>9} {'win%':>6}")
    for fam in sorted({r.family for r in results}):
        fr = [r for r in results if r.family == fam]
        mc = np.mean([r.conv_test for r in fr])
        mm = np.mean([r.mlp_test for r in fr])
        w = 100.0 * sum(r.conv_wins for r in fr) / len(fr)
        print(f"  {fam:<6} {mc:>10.6f} {mm:>10.6f} {mc / mm:>9.3f} {w:>5.1f}%")


def oracle_check() -> None:
    """Verify that with noise_std=0 the true kernel gives ~0 test MSE."""
    print("\n=== Oracle check (noise_std=0) ===")
    sampler = DatasetFamily()
    for family in FAMILIES:
        cfg = make_config(family, radius=2, noise_std=0.0, n_train=64,
                          seed=42)
        inst = sampler.sample_dataset(cfg)
        oracle = oracle_test_mse(inst.x_test.numpy(), inst.y_test.numpy(),
                                 inst.kernel.numpy())
        id_again = dataset_id_from_config(cfg)
        assert id_again == inst.dataset_id, "dataset_id not deterministic"
        status = "OK" if oracle < 1e-9 else "FAIL"
        print(f"  {family:<6} oracle_test_mse={oracle:.3e}  id={inst.dataset_id}  [{status}]")
        assert oracle < 1e-9, (
            f"Oracle check failed for {family}: oracle MSE={oracle} (expected ~0 "
            "with noise_std=0). Data generation is inconsistent with the kernel.")
    print("  All oracle checks passed (true kernel reproduces y exactly).")


def main() -> int:
    """Run the smoke test and assert conv1d < MLP test loss."""
    set_seed(0)
    print("=" * 78)
    print("Phase 1 smoke test: conv1d vs from-scratch MLP")
    print("=" * 78)
    print(f"Device: {DEVICE}  "
          f"(cuda available: {torch.cuda.is_available()})")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    oracle_check()

    print("\n=== Conv1d vs MLP comparison ===")
    results = run_smoke_test()
    print_table(results)

    conv_wins = sum(r.conv_wins for r in results)
    win_rate = conv_wins / len(results)
    mean_conv = float(np.mean([r.conv_test for r in results]))
    mean_mlp = float(np.mean([r.mlp_test for r in results]))

    print("\n=== Assertions ===")
    ok_mean = mean_conv < mean_mlp
    ok_rate = win_rate >= 0.8
    print(f"  mean conv_test < mean mlp_test : {mean_conv:.6f} < {mean_mlp:.6f} "
          f"-> {'PASS' if ok_mean else 'FAIL'}")
    print(f"  conv win rate >= 80%            : {win_rate*100:.1f}% "
          f"-> {'PASS' if ok_rate else 'FAIL'}")

    if not (ok_mean and ok_rate):
        print("\nSMOKE TEST FAILED: conv1d did not beat the MLP as expected.")
        print("This indicates a problem with the data generation (too easy/global) "
              "or the comparison setup (MLP too constrained / n_train too large).")
        return 1

    print("\nSMOKE TEST PASSED: conv1d solves the generated data better than the "
          "from-scratch MLP (inductive bias > capacity alone).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
