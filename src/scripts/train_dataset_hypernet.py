"""CLI: train DatasetHypernet (examples → weights)."""
import argparse, sys
from src.training.train_dataset_hypernet import DatasetHypernetConfig, train_dataset_hypernet

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-dir", type=str, default="results/dataset_hypernet")
    p.add_argument("--k-enc", type=int, default=32)
    p.add_argument("--n-loss", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-emb", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--corpus-dir", type=str, default="data/processed/targets_relu_h16")
    p.add_argument("--mlp-hidden", type=int, default=16)
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    cfg = DatasetHypernetConfig(
        lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
        max_steps=args.max_steps, grad_clip=args.grad_clip,
        device=args.device, seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        K_enc=args.k_enc, N_loss=args.n_loss,
        d_model=args.d_model, d_emb=args.d_emb,
        n_layers=args.n_layers, n_heads=args.n_heads,
        corpus_dir=args.corpus_dir, mlp_hidden=args.mlp_hidden,
    )
    train_dataset_hypernet(cfg)
    return 0

if __name__ == "__main__":
    sys.exit(main())
