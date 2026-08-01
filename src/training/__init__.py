"""Training package: MLP training-to-convergence, target collection, and the
effective-map regressor training loop.

Phase 2 (Approach B): train many MLPs (different random initializations) to
convergence on generated 1D datasets and collect their converged weights.
"""
from src.training.train_mlp import TrainConfig, TrainResult, train_mlp_to_convergence

__all__ = ["TrainConfig", "TrainResult", "train_mlp_to_convergence"]
