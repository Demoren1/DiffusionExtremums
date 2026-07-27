"""Training package: MLP training-to-convergence and target collection.

Phase 2 (Approach B): train many MLPs (different random initializations) to
convergence on generated 1D datasets and collect their converged weights as
diffusion targets.
"""
from src.training.train_mlp import TrainConfig, TrainResult, train_mlp_to_convergence

__all__ = ["TrainConfig", "TrainResult", "train_mlp_to_convergence"]
