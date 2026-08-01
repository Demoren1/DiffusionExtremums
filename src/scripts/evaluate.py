"""CLI: run the 3-method evaluation pipeline.

Generates datasets for a sample of train + val configs, runs the 3 methods
(from-scratch MLP, learned conv, oracle conv), computes Toeplitz-ness and
kernel-recovery metrics, saves results to ``--output-dir``, and generates
figures to ``--figures-dir``.

Examples:
    # Full evaluation
    python -m src.scripts.evaluate \
        --n-eval-train 20 --n-eval-val 20

    # Quick smoke run
    python -m src.scripts.evaluate \
        --n-eval-train 2 --n-eval-val 2 --n-train-mlp 256

Run with the conda env activated:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import torch

from src.evaluation.evaluate import (
    aggregate_results,
    evaluate_datasets,
    print_comparison_table,
    save_results,
)
from src.evaluation.visualize import generate_all_figures
from src.training.train_effective_map_regressor import deterministic_split
from src.utils.seeding import set_seed


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="3-method evaluation: metrics and visualization.")
    p.add_argument("--targets-dir", type=str,
                   default="data/processed/targets_eff",
                   help="Directory with the target effective maps + configs.")
    p.add_argument("--n-eval-train", type=int, default=20,
                   help="Number of train configs to evaluate (default 20).")
    p.add_argument("--n-eval-val", type=int, default=20,
                   help="Number of val (held-out) configs to evaluate (default 20).")
    p.add_argument("--val-configs", type=int, default=50,
                   help="Number of configs held out for the deterministic "
                        "train/val split (default 50).")
    p.add_argument("--n-train-mlp", type=int, default=1024,
                   help="n_train for the from-scratch MLP baseline (default 1024).")
    p.add_argument("--device", type=str, default="cuda",
                   help="torch device (default cuda).")
    p.add_argument("--output-dir", type=str, default="results/evaluation",
                   help="Directory for evaluation results (default results/evaluation).")
    p.add_argument("--figures-dir", type=str, default="figures",
                   help="Directory for figures (default figures).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default 0).")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Run the 3-method evaluation pipeline."""
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load configs + targets
    # ------------------------------------------------------------------
    print("=" * 70)
    print(" 3-method evaluation")
    print("=" * 70)
    print(f"  targets_dir:  {args.targets_dir}")
    print(f"  n_eval_train: {args.n_eval_train}")
    print(f"  n_eval_val:   {args.n_eval_val}")
    print(f"  n_train_mlp:  {args.n_train_mlp}")
    print(f"  device:       {device}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  figures_dir:  {args.figures_dir}")
    print("-" * 70)

    print("\n[1/3] Loading target effective maps + configs...")
    with open(os.path.join(args.targets_dir, "configs.json"), "r") as f:
        configs: List[Dict] = json.load(f)
    target_eff_maps = torch.load(
        os.path.join(args.targets_dir, "eff_maps.pt"), map_location="cpu").float()
    print(f"  targets: {tuple(target_eff_maps.shape)}, configs: {len(configs)}")

    # Deterministic config-level split (same scheme as the regressor pipeline).
    train_config_indices, val_config_indices = deterministic_split(
        len(configs), args.val_configs, args.seed)
    print(f"  train configs: {len(train_config_indices)}, "
          f"val configs: {len(val_config_indices)}")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print(f"\n[2/3] Running evaluation ({args.n_eval_train} train + "
          f"{args.n_eval_val} val configs)...")
    results = evaluate_datasets(
        configs=configs,
        train_config_indices=train_config_indices,
        val_config_indices=val_config_indices,
        n_eval_train=args.n_eval_train,
        n_eval_val=args.n_eval_val,
        n_train_mlp=args.n_train_mlp,
        device=device,
        target_eff_maps=target_eff_maps,
        seed=args.seed,
        verbose=True,
    )

    summary = aggregate_results(results)
    print_comparison_table(results, summary)
    save_results(results, summary, args.output_dir)

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    if not args.no_figures:
        print(f"\n[3/3] Generating figures -> {args.figures_dir}/...")
        paths = generate_all_figures(
            results=results,
            configs=configs,
            target_eff_maps=target_eff_maps,
            figures_dir=args.figures_dir,
        )
        print(f"  generated {len(paths)} figures: {paths}")
    else:
        print("\n[3/3] Skipping figures (--no-figures).")

    print("\n" + "=" * 70)
    print(" 3-method evaluation complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
