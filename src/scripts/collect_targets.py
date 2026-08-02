"""CLI: collect converged MLP weights as regression targets (Phase 2, Approach B).

Generates a corpus of datasets, trains ``n_mlp`` MLPs (different random inits)
per dataset to convergence, and saves the flattened converged weight vectors
plus dataset configs and metadata to ``--out-dir``.

Multi-GPU: pass ``--gpus`` (comma-separated GPU ids, or ``auto`` for all visible
GPUs). With >1 GPU the corpus is sharded across GPUs (one process per GPU via
``torch.multiprocessing.spawn``) and the shards are merged into the standard
output format. With 0/1 GPU the original sequential single-device path is used.

Examples:
    # Small smoke run (10 datasets x 4 MLPs, single GPU)
    python -m src.scripts.collect_targets --n-datasets 10 --n-mlp 4 --out-dir data/processed/targets_smoke

    # Full corpus (2000 datasets x 8 MLPs, n_train=1024) on all visible GPUs
    python -m src.scripts.collect_targets --n-datasets 2000 --n-mlp 8 --gpus auto --out-dir data/processed/targets

    # Use a specific subset of GPUs (e.g. 4 of 8 A100s)
    python -m src.scripts.collect_targets --n-datasets 2000 --n-mlp 8 --gpus 0,1,2,3 --out-dir data/processed/targets

Run with the conda env activated:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
"""
import argparse
import os
import sys

import torch

from src.training.collect_targets import CollectConfig, collect_targets, resolve_gpus
from src.training.train_mlp import TrainConfig


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Collect converged MLP weights as regression targets "
                    "(Phase 2, Approach B).")
    # Corpus
    p.add_argument("--n-datasets", type=int, default=2000,
                   help="Number of datasets to generate and process (default 2000).")
    p.add_argument("--n-mlp", type=int, default=8,
                   help="Number of MLPs (random inits) per dataset (default 8).")
    p.add_argument("--mlp-hidden", type=int, default=128,
                   help="MLP hidden dimension H (default 128).")
    p.add_argument("--n-train", type=int, default=1024,
                   help="Training samples per dataset (default 1024).")
    p.add_argument("--n-test", type=int, default=512,
                   help="Test samples per dataset (default 512).")
    p.add_argument("--noise-std", type=float, default=0.1,
                   help="Noise std for all datasets (default 0.1).")
    p.add_argument("--corpus-seed", type=int, default=0,
                   help="Base RNG seed for sampling dataset configs (default 0).")
    p.add_argument("--mlp-seed-base", type=int, default=1000,
                   help="Base seed for MLP inits (default 1000).")
    p.add_argument("--families", type=str, default=None,
                   help="Comma-separated family subset, e.g. 'MA,GAUSS' "
                        "(default: all 5).")
    # Training
    p.add_argument("--lr", type=float, default=3e-3, help="AdamW peak LR.")
    p.add_argument("--steps", type=int, default=5000,
                   help="Max training steps per MLP (default 5000).")
    p.add_argument("--min-steps", type=int, default=500,
                   help="Min steps before early stopping (default 500).")
    p.add_argument("--patience", type=int, default=300,
                   help="Early-stopping patience in steps (default 300).")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Validation fraction of the training set (default 0.1).")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight decay (default 0.0).")
    p.add_argument("--device", type=str, default="auto",
                   help="torch device: 'auto', 'cuda', 'cpu' (default auto). "
                        "Used only in the single-GPU sequential path; in the "
                        "multi-GPU path each worker is pinned to its GPU.")
    # Multi-GPU
    p.add_argument("--gpus", type=str, default="auto",
                   help="Comma-separated GPU ids to use for multi-GPU parallel "
                        "collection (e.g. '0,1,2,3'), or 'auto' to use all visible "
                        "GPUs (default auto). With >1 GPU the corpus is sharded "
                        "across GPUs (one process per GPU) and shards are merged. "
                        "With 0/1 GPU the sequential single-device path is used.")
    # Output
    p.add_argument("--out-dir", type=str, default="data/processed/targets",
                   help="Output directory (default data/processed/targets).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Run the target-collection pipeline."""
    args = parse_args(argv)

    families = None
    if args.families is not None:
        families = tuple(s.strip() for s in args.families.split(",") if s.strip())

    # Resolve the GPU list. With >1 GPU the parallel path is used (each worker
    # pins itself to its GPU and overrides the device to "cuda"); with 0/1 GPU
    # the sequential path uses --device (default "auto").
    gpu_ids = resolve_gpus(args.gpus)
    if len(gpu_ids) > 1:
        device = "cuda"
        print(f"[collect_targets] multi-GPU: {len(gpu_ids)} GPUs {gpu_ids}")
    else:
        device = args.device
        if len(gpu_ids) == 1:
            # Pin the single GPU via CUDA_VISIBLE_DEVICES for consistency.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
            device = "cuda" if torch.cuda.is_available() else args.device
            print(f"[collect_targets] single-GPU: gpu {gpu_ids[0]}")
        else:
            print(f"[collect_targets] no GPUs selected; using device '{device}'")

    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        steps=args.steps,
        min_steps=args.min_steps,
        patience=args.patience,
        val_frac=args.val_frac,
        L=32,
        H=args.mlp_hidden,
        device=device,
    )
    collect_cfg = CollectConfig(
        n_datasets=args.n_datasets,
        n_mlp=args.n_mlp,
        n_train=args.n_train,
        n_test=args.n_test,
        noise_std=args.noise_std,
        families=families,
        corpus_seed=args.corpus_seed,
        mlp_seed_base=args.mlp_seed_base,
        L=32,
        H=args.mlp_hidden,
        out_dir=args.out_dir,
        gpus=gpu_ids if len(gpu_ids) > 1 else None,
        train=train_cfg,
    )
    collect_targets(collect_cfg, show_progress=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
