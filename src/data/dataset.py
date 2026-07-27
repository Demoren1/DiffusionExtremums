"""Dataset generation: configs, instances, and the family sampler.

Implements the clean API described in ``plans/plan.md`` Section 1:

    generate_dataset(config) -> (x_train, y_train, x_test, y_test, kernel, dataset_id)
    generate_corpus(n_datasets, ...) -> list[DatasetInstance]

Conventions (plan Section 1.2 / 1.5 / 1.6 / 1.7):
- Input length L = 32 (fixed for v1); each sample x in R^L.
- Kernel radius r in {1,2,3}  ->  kernel size K = 2r+1 in {3,5,7}.
- Convolution mode "same" (output length = L) with zero-padding at boundaries:
      y_p = sum_{j=-r}^{r} k_j * x_{p+j}.
- Inputs x ~ i.i.d. N(0,1) per coordinate.
- Train/test split: test inputs are independent i.i.d. draws from the same
  distribution (measures generalization to new inputs, not memorization).
- All tensors float32. Shapes: x [B, L], y [B, L], kernel [K].
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.configs.base import DatasetConfig
from src.data.families import FAMILIES, sample_kernel
from src.data.registry import dataset_id_from_config


# ---------------------------------------------------------------------------
# Convolution (same padding, zero-padding at boundaries)
# ---------------------------------------------------------------------------

def conv1d_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """1D 'same'-padding convolution with zero-padding at boundaries.

    Args:
        x: float32 array of shape [B, L].
        kernel: float32 array of shape [K] (K odd).

    Returns:
        float32 array of shape [B, L] where
        ``y[b, p] = sum_{j=-r}^{r} k[j+r] * x[b, p+j]`` with zero-padding.
    """
    B, L = x.shape
    K = kernel.shape[0]
    if K % 2 == 0:
        raise ValueError(f"kernel size K must be odd, got {K}")
    r = K // 2
    # Zero-pad x along the length axis: [B, L + 2r].
    xp = np.pad(x, ((0, 0), (r, r)), mode="constant")
    y = np.zeros((B, L), dtype=np.float32)
    for j in range(K):
        # window [p+j-r .. p+j-r] -> for output p, contribution k[j]*x[p + (j-r)]
        y += kernel[j] * xp[:, j : j + L]
    return y


# ---------------------------------------------------------------------------
# Dataset instance
# ---------------------------------------------------------------------------

@dataclass
class DatasetInstance:
    """A fully materialized dataset: tensors + kernel + id + config.

    Attributes:
        x_train: [n_train, L] float32.
        y_train: [n_train, L] float32 (conv(x,k) + noise).
        x_test:  [n_test, L] float32.
        y_test:  [n_test, L] float32 (conv(x,k) + noise, independent draws).
        kernel:  [K] float32 ground-truth FIR filter.
        dataset_id: 16-hex string.
        config:  the ``DatasetConfig`` that generated this instance.
    """

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    kernel: torch.Tensor
    dataset_id: str
    config: DatasetConfig

    def to(self, device: torch.device) -> "DatasetInstance":
        """Move all tensors to ``device`` and return self."""
        self.x_train = self.x_train.to(device)
        self.y_train = self.y_train.to(device)
        self.x_test = self.x_test.to(device)
        self.y_test = self.y_test.to(device)
        self.kernel = self.kernel.to(device)
        return self


# ---------------------------------------------------------------------------
# Dataset family sampler
# ---------------------------------------------------------------------------

class DatasetFamily:
    """Sampler over dataset configs and instances for one (or all) families.

    Provides ``sample_random_config`` (for corpus generation) and
    ``sample_dataset`` (materialize tensors from a config), matching the
    DataModule API in plan Section 1.7.
    """

    # Discrete parameter grids (plan Section 1.4).
    RADII = (1, 2, 3)
    NOISE_STDS = (0.0, 0.05, 0.1, 0.2)
    N_TRAINS = (64, 128, 256)

    def __init__(self, families: Optional[Tuple[str, ...]] = None,
                 n_test: int = 512, L: int = 32):
        """Initialize the sampler.

        Args:
            families: Restrict to a subset of families; None means all 5.
            n_test: Number of test samples per dataset (default 512).
            L: Input length (fixed at 32 for v1).
        """
        self.families = tuple(families) if families is not None else FAMILIES
        for f in self.families:
            if f not in FAMILIES:
                raise ValueError(f"Unknown family {f!r}; expected one of {FAMILIES}")
        self.n_test = int(n_test)
        self.L = int(L)

    def sample_random_config(self, rng: np.random.Generator,
                             n_train: Optional[int] = None,
                             noise_std: Optional[float] = None) -> DatasetConfig:
        """Sample a random ``DatasetConfig`` (kernel + parameters).

        Args:
            rng: NumPy ``Generator`` for reproducibility.
            n_train: Override the sampled training-sample count; None samples
                uniformly from {64, 128, 256}.
            noise_std: Override the sampled noise std; None samples uniformly
                from {0.0, 0.05, 0.1, 0.2}.

        Returns:
            A ``DatasetConfig`` with a sampled kernel.
        """
        family = str(rng.choice(self.families))
        radius = int(rng.choice(self.RADII))
        # Kernel uses its own RNG stream derived from the base seed.
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
        """Materialize tensors from a ``DatasetConfig``.

        Inputs x ~ i.i.d. N(0,1); targets y = conv1d_same(x, k) + N(0, noise_std^2).
        Train and test inputs are independent draws (generalization to new inputs).
        """
        K = config.K
        if len(config.kernel) != K:
            raise ValueError(
                f"kernel length {len(config.kernel)} != 2*radius+1 = {K}")
        kernel = np.asarray(config.kernel, dtype=np.float32)

        # Deterministic RNG from the config seed.
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


# ---------------------------------------------------------------------------
# Top-level convenience API
# ---------------------------------------------------------------------------

def generate_dataset(config: DatasetConfig) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, str]:
    """Generate a single dataset from a config.

    Returns:
        ``(x_train, y_train, x_test, y_test, kernel, dataset_id)`` where
        x_train [n_train, L], y_train [n_train, L], x_test [n_test, L],
        y_test [n_test, L], kernel [K], dataset_id is a 16-hex string.
    """
    inst = DatasetFamily().sample_dataset(config)
    return (inst.x_train, inst.y_train, inst.x_test, inst.y_test,
            inst.kernel, inst.dataset_id)


def generate_corpus(n_datasets: int, seed: int = 0,
                     families: Optional[Tuple[str, ...]] = None,
                     n_train: Optional[int] = None,
                     noise_std: Optional[float] = None,
                     n_test: int = 512, L: int = 32) -> List[DatasetInstance]:
    """Generate a corpus of ``n_datasets`` dataset instances.

    Args:
        n_datasets: Number of datasets to generate.
        seed: Base RNG seed for sampling configs (each config gets its own
            derived seed for input/noise generation).
        families: Restrict to a subset of families; None means all 5.
        n_train: Override training-sample count for all datasets.
        noise_std: Override noise std for all datasets.
        n_test: Number of test samples per dataset.
        L: Input length (fixed at 32 for v1).

    Returns:
        List of ``DatasetInstance``.
    """
    rng = np.random.default_rng(seed)
    family = DatasetFamily(families=families, n_test=n_test, L=L)
    instances: List[DatasetInstance] = []
    for _ in range(n_datasets):
        cfg = family.sample_random_config(rng, n_train=n_train, noise_std=noise_std)
        instances.append(family.sample_dataset(cfg))
    return instances
