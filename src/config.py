"""Configuration dataclasses for the diffusion-extremums project.

All experiment knobs live here so that train/inference scripts can be driven
purely from a config (dataclass) instance.  Keeping everything in one module
makes it easy to serialize / log the full config to TensorBoard.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


# --------------------------------------------------------------------------- #
# Target function
# --------------------------------------------------------------------------- #
@dataclass
class TargetConfig:
    """Parameters of the periodic target function y = sin(k0 x + b0) + sin(k1 x + b1)."""
    k: Tuple[float, float] = (0.7, 1.5)
    b: Tuple[float, float] = (1.0, -1.0)
    # Domain on which the function is sampled / the MLP is trained.
    x_min: float = -40.0
    x_max: float = 40.0
    n_samples: int = 1000


# --------------------------------------------------------------------------- #
# Target MLP (the network whose *weights* the diffusion model generates)
# --------------------------------------------------------------------------- #
@dataclass
class TargetMLPConfig:
    """Architecture of the small MLP that approximates the target function."""
    in_dim: int = 1
    hidden_dim: int = 32          # width of the hidden layer (configurable, > 10)
    out_dim: int = 1
    activation: str = "tanh"      # "tanh" | "relu"

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters (weights + biases)."""
        return (self.in_dim * self.hidden_dim + self.hidden_dim) + \
               (self.hidden_dim * self.out_dim + self.out_dim)


# --------------------------------------------------------------------------- #
# Dataset of MLP weights
# --------------------------------------------------------------------------- #
@dataclass
class WeightsDatasetConfig:
    """How the *population* of trained MLPs (the diffusion training data) is built."""
    n_networks: int = 2000       # how many MLPs to train
    base_seed: int = 0           # first seed; seeds are range(base_seed, base_seed + n_networks)
    mlp_epochs: int = 2000       # epochs to train each individual MLP
    mlp_lr: float = 1e-2
    mlp_batch_size: int = 256
    # Normalization applied to the flattened weight vectors before diffusion.
    normalize: bool = True       # standardize per-dimension (zero mean, unit std)
    # Canonicalize each MLP's weights before flattening to remove permutation
    # and sign-flip symmetries of the hidden neurons (reduces multimodality).
    canonicalize: bool = True
    # Where the collected weight matrix + stats are stored.
    out_path: str = "datasets/weights.pt"


# --------------------------------------------------------------------------- #
# 2D toy dataset (sanity check for the DDPM implementation)
# --------------------------------------------------------------------------- #
@dataclass
class Toy2DConfig:
    """Two-moons configuration used to validate the diffusion core."""
    n_samples: int = 50000
    noise: float = 0.05
    seed: int = 0


# --------------------------------------------------------------------------- #
# DDPM core
# --------------------------------------------------------------------------- #
@dataclass
class DDPMConfig:
    """Denoising-diffusion hyper-parameters."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"     # "linear" | "cosine"
    # What the denoiser predicts: "epsilon" (noise) or "x0".
    objective: str = "epsilon"
    # Variance of the posterior used in sampling: "fixed_small" | "fixed_large" | "learned".
    variance_type: str = "fixed_small"


# --------------------------------------------------------------------------- #
# Denoiser network
# --------------------------------------------------------------------------- #
@dataclass
class DenoiserConfig:
    """Architecture of the score / denoiser network."""
    data_dim: int = 2            # set automatically from the dataset
    hidden_dims: tuple = (256, 256, 256)
    time_emb_dim: int = 128
    # Optional conditioning dimension (0 = unconditional).
    cond_dim: int = 0
    dropout: float = 0.0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Training-loop knobs."""
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 500
    # Logging / checkpointing
    log_dir: str = "runs/diffusion"
    log_every: int = 50          # steps between scalar logs
    sample_every: int = 500      # steps between image/figure logs
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 1000        # steps between checkpoint saves
    # Reproducibility
    seed: int = 42
    # Device: "auto" | "cuda" | "cpu"
    device: str = "auto"
    num_workers: int = 0
    # Train / validation split: fraction of the dataset held out for val loss.
    val_fraction: float = 0.1    # 0 => no validation split
    val_every: int = 500         # steps between val-loss evaluations
    # EMA of denoiser weights (improves sample quality at inference).
    ema_decay: float = 0.9999    # 0 => EMA disabled
    ema_update_after: int = 0    # start EMA after this many steps
    # Use EMA weights for sampling / checkpointing (recommended).
    use_ema_for_sampling: bool = True


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    """Sampling / evaluation knobs."""
    n_samples: int = 64
    # Number of DDPM denoising steps at sampling time (can differ from training T).
    sampling_timesteps: Optional[int] = None   # None => use num_timesteps
    # Evaluation grid for the reconstructed function.
    eval_x_min: float = -40.0
    eval_x_max: float = 40.0
    eval_n: int = 1000
    seed: int = 123
    device: str = "auto"
    out_dir: str = "outputs"


# --------------------------------------------------------------------------- #
# Top-level experiment config
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    """Aggregate config; one object describes a full run."""
    name: str = "default"
    # Which dataset to train the diffusion model on: "weights" | "toy2d".
    data_source: str = "toy2d"
    target: TargetConfig = field(default_factory=TargetConfig)
    target_mlp: TargetMLPConfig = field(default_factory=TargetMLPConfig)
    weights_dataset: WeightsDatasetConfig = field(default_factory=WeightsDatasetConfig)
    toy2d: Toy2DConfig = field(default_factory=Toy2DConfig)
    ddpm: DDPMConfig = field(default_factory=DDPMConfig)
    denoiser: DenoiserConfig = field(default_factory=DenoiserConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Reconstruct a full config from a (possibly nested) plain dict.

        Nested dataclass fields are rebuilt explicitly so that attribute access
        (e.g. ``cfg.denoiser.data_dim``) works after loading from a checkpoint.
        """
        def _sub(sub_cls, sub_d):
            return sub_cls(**{k: v for k, v in sub_d.items() if k in sub_cls.__dataclass_fields__})

        return cls(
            name=d.get("name", "default"),
            data_source=d.get("data_source", "toy2d"),
            target=_sub(TargetConfig, d.get("target", {})),
            target_mlp=_sub(TargetMLPConfig, d.get("target_mlp", {})),
            weights_dataset=_sub(WeightsDatasetConfig, d.get("weights_dataset", {})),
            toy2d=_sub(Toy2DConfig, d.get("toy2d", {})),
            ddpm=_sub(DDPMConfig, d.get("ddpm", {})),
            denoiser=_sub(DenoiserConfig, d.get("denoiser", {})),
            train=_sub(TrainConfig, d.get("train", {})),
            inference=_sub(InferenceConfig, d.get("inference", {})),
        )
