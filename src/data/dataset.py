"""Dataset generation: configs, instances, and the family sampler."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.configs.base import DatasetConfig
from src.data.families import FAMILIES, sample_kernel
from src.data.registry import dataset_id_from_config


def conv1d_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    B, L = x.shape
    K = kernel.shape[0]
    if K % 2 == 0:
        raise ValueError(f"kernel size K must be odd, got {K}")
    r = K // 2
    xp = np.pad(x, ((0, 0), (r, r)), mode="constant")
    y = np.zeros((B, L), dtype=np.float32)
    for j in range(K):
        y += kernel[j] * xp[:, j : j + L]
    return y


@dataclass
class DatasetInstance:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    kernel: torch.Tensor
    dataset_id: str
    config: DatasetConfig

    def to(self, device: torch.device) -> "DatasetInstance":
        self.x_train = self.x_train.to(device)
        self.y_train = self.y_train.to(device)
        self.x_test = self.x_test.to(device)
        self.y_test = self.y_test.to(device)
        self.kernel = self.kernel.to(device)
        return self


class DatasetFamily:
    RADII = (1, 2, 3)
    NOISE_STDS = (0.0, 0.05, 0.1, 0.2)
    N_TRAINS = (64, 128, 256)

    def __init__(self, families: Optional[Tuple[str, ...]] = None,
                 n_test: int = 512, L: int = 32):
        self.families = tuple(families) if families is not None else FAMILIES
        for f in self.families:
            if f not in FAMILIES:
                raise ValueError(f"Unknown family {f!r}; expected one of {FAMILIES}")
        self.n_test = int(n_test)
        self.L = int(L)

    def sample_random_config(self, rng: np.random.Generator,
                             n_train: Optional[int] = None,
                             noise_std: Optional[float] = None) -> DatasetConfig:
        family = str(rng.choice(self.families))
        radius = int(rng.choice(self.RADII))
        kernel = sample_kernel(family, radius, rng)
        if n_train is None:
            n_train = int(rng.choice(self.N_TRAINS))
        if noise_std is None:
            noise_std = float(rng.choice(self.NOISE_STDS))
        seed = int(rng.integers(0, 2 ** 31 - 1))
        return DatasetConfig(
            family=family,
            kernel=tuple(float(v) for v in kernel),
            radius=radius,
            noise_std=noise_std,
            n_train=int(n_train),
            n_test=self.n_test,
            seed=seed,
            L=self.L,
        )

    def sample_dataset(self, config: DatasetConfig) -> DatasetInstance:
        K = config.K
        if len(config.kernel) != K:
            raise ValueError(
                f"kernel length {len(config.kernel)} != 2*radius+1 = {K}")
        kernel = np.asarray(config.kernel, dtype=np.float32)

        rng = np.random.default_rng(config.seed)

        def _draw(n: int) -> Tuple[np.ndarray, np.ndarray]:
            x = rng.standard_normal((n, config.L)).astype(np.float32)
            y = conv1d_same(x, kernel)
            if config.noise_std > 0.0:
                y = y + (config.noise_std * rng.standard_normal(
                    y.shape)).astype(np.float32)
            return x, y

        x_train, y_train = _draw(config.n_train)
        x_test, y_test = _draw(config.n_test)

        return DatasetInstance(
            x_train=torch.from_numpy(x_train),
            y_train=torch.from_numpy(y_train),
            x_test=torch.from_numpy(x_test),
            y_test=torch.from_numpy(y_test),
            kernel=torch.from_numpy(kernel),
            dataset_id=dataset_id_from_config(config),
            config=config,
        )


def generate_dataset(config: DatasetConfig) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, str]:
    inst = DatasetFamily().sample_dataset(config)
    return (inst.x_train, inst.y_train, inst.x_test, inst.y_test,
            inst.kernel, inst.dataset_id)


def generate_corpus(n_datasets: int, seed: int = 0,
                     families: Optional[Tuple[str, ...]] = None,
                     n_train: Optional[int] = None,
                     noise_std: Optional[float] = None,
                     n_test: int = 512, L: int = 32) -> List[DatasetInstance]:
    rng = np.random.default_rng(seed)
    family = DatasetFamily(families=families, n_test=n_test, L=L)
    instances: List[DatasetInstance] = []
    for _ in range(n_datasets):
        cfg = family.sample_random_config(rng, n_train=n_train, noise_std=noise_std)
        instances.append(family.sample_dataset(cfg))
    return instances
