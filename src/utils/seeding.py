"""Reproducibility helpers: seed torch, numpy, and random.

See ``plans/plan.md`` Section 6.4.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
