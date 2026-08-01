"""CLI: train the deterministic effective-map regressor (sanity-check baseline).

Trains a small MLP to map the 14-dim config feature vector directly to the
1056-dim effective linear map, using a deterministic config-level train/val
split (``--seed`` / ``--val-configs``).

Examples:
    # Full run
    python -m src.scripts.train_effective_map_regressor \
        --checkpoint-dir results/regressor_sanity

    # Smoke run
    python -m src.scripts.train_effective_map_regressor \
        --steps 200 --batch-size 16 \
        --checkpoint-dir results/regressor_sanity_smoke

Run with the conda env activated:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
"""
import argparse
import sys

from src.training.train_effective_map_regressor import (
    TrainRegressorConfig,
    train_regressor,
)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Train the deterministic effective-map regressor "
                    "(sanity-check baseline).")
    # Data
    p.add_argument("--targets-dir", type=str,
                   default="data/processed/targets_eff",
                   help="Directory with eff_maps.pt + configs.json (default "
                        "data/processed/targets_eff).")
    p.add_argument("--val-configs", type=int, default=50,
                   help="Number of val configs held out for validation "
                        "(deterministic split from --seed). Default 50.")
    p.add_argument("--target-mode", type=str, default="learned",
                   choices=["learned", "oracle"],
                   help="Regression target construction. 'learned' (default) "
                        "uses the MLP-averaged effective maps from eff_maps.pt. "
                        "'oracle' builds the exact kernel-derived effective map "
                        "per config (deterministic, noise-free baseline).")
    # Training
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Peak AdamW learning rate (default 1e-3).")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay (default 1e-4).")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Final LR for the cosine schedule (default 1e-6).")
    p.add_argument("--epochs", type=int, default=20000,
                   help="Full passes over train configs (default 20000). Used "
                        "when --steps is not set.")
    p.add_argument("--steps", dest="max_steps", type=int, default=None,
                   help="Total optimizer steps (takes precedence over --epochs).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Training batch size in configs (default 64).")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (default 1.0; 0 disables).")
    p.add_argument("--patience", type=int, default=50,
                   help="Early-stopping patience in evals (default 50; 0 = off).")
    # Logging / checkpointing
    p.add_argument("--log-every", type=int, default=100,
                   help="Log train loss every N steps (default 100).")
    p.add_argument("--eval-every", type=int, default=500,
                   help="Evaluate val loss every N steps (default 500).")
    p.add_argument("--save-every", type=int, default=2000,
                   help="Save a periodic checkpoint every N steps (default 2000).")
    p.add_argument("--checkpoint-dir", type=str,
                   default="results/regressor_sanity",
                   help="Directory for checkpoints + logs (default "
                        "results/regressor_sanity).")
    p.add_argument("--log-dir", type=str, default=None,
                   help="TensorBoard log dir (default "
                        "<checkpoint-dir>/tensorboard).")
    p.add_argument("--resume", type=str, default=None,
                   help="Optional checkpoint path to resume from.")
    # Device / seed
    p.add_argument("--device", type=str, default="cuda",
                   help="torch device: 'auto', 'cuda', 'cpu' (default cuda).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default 0).")
    # Model
    p.add_argument("--hidden-dims", type=str, default="256,512,512",
                   help="Comma-separated hidden widths (default 256,512,512).")
    p.add_argument("--activation", type=str, default="gelu",
                   choices=["gelu", "silu", "relu"],
                   help="Activation (default gelu).")
    p.add_argument("--no-residual", dest="use_residual", action="store_false",
                   help="Disable the residual connection.")
    p.add_argument("--no-layer-norm", dest="use_layer_norm",
                   action="store_false",
                   help="Disable LayerNorm.")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout probability (default 0).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Run the effective-map regressor training."""
    args = parse_args(argv)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]

    if args.max_steps is not None:
        epochs = None
    else:
        epochs = args.epochs

    config = TrainRegressorConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_min=args.lr_min,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        epochs=epochs,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume=args.resume,
        targets_dir=args.targets_dir,
        val_configs=args.val_configs,
        target_mode=args.target_mode,
        hidden_dims=hidden_dims,
        activation=args.activation,
        use_residual=args.use_residual,
        use_layer_norm=args.use_layer_norm,
        dropout=args.dropout,
    )
    train_regressor(config, show_progress=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
