"""Base configuration dataclasses.

See ``plans/plan.md`` Section 1.4 for the ``DatasetConfig`` specification.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DatasetConfig:
    """Full specification of a single 1D regression dataset instance.

    A dataset instance = one sampled kernel + noise config + length + sample count.
    The same config always maps to the same ``dataset_id`` (deterministic hash),
    so conditioning is reproducible.

    Attributes:
        family: Dataset family in {MA, DIFF, GAUSS, MATCH, RAND}.
        kernel: The ground-truth FIR filter as a tuple of floats, length K = 2*radius+1.
        radius: Kernel radius r in {1, 2, 3}  ->  kernel size K = 2r+1 in {3, 5, 7}.
        noise_std: Std of additive Gaussian noise on y. 0.0 means no noise.
        n_train: Number of training samples.
        n_test: Number of test samples (default 512).
        seed: Base RNG seed for input and noise generation.
        L: Input length (fixed at 32 for v1).
    """

    family: str
    kernel: Tuple[float, ...]
    radius: int
    noise_std: float
    n_train: int
    n_test: int = 512
    seed: int = 0
    L: int = 32

    @property
    def K(self) -> int:
        """Kernel size K = 2*radius + 1."""
        return 2 * self.radius + 1
