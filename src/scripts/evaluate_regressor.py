"""CLI: evaluate the trained effective-map regressor (sanity check).

Loads a trained regressor checkpoint, evaluates normalized/raw MSE,
Toeplitz-ness, kernel recovery, and functional test MSE on a sample of train +
val configs, saves metrics to ``--output-dir``, and generates figures to
``--figures-dir``.

Examples:
    python -m src.scripts.evaluate_regressor \
        --checkpoint results/regressor_sanity/regressor_best.pt \
        --n-eval-train 20 --n-eval-val 20

Run with the conda env activated:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import torch

from src.evaluation.regressor_eval import (
    aggregate_regressor_results,
    evaluate_regressor,
    generate_regressor_figures,
    print_regressor_summary,
    save_regressor_results,
)
from src.models.effective_map import DEFAULT_H, DEFAULT_L
from src.models.effective_map_regressor import (
    EffectiveMapRegressor,
    RegressorConfig,
)
from src.models.weight_norm import WeightNormalizer
from src.utils.seeding import set_seed


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate the deterministic effective-map regressor "
                    "(sanity-check baseline).")
    p.add_argument("--checkpoint", type=str,
                   default="results/regressor_sanity/regressor_best.pt",
                   help="Path to the trained regressor checkpoint.")
    p.add_argument("--targets-dir", type=str,
                   default="data/processed/targets_eff",
                   help="Directory with eff_maps.pt + configs.json.")
    p.add_argument("--n-eval-train", type=int, default=20,
                   help="Number of train configs to evaluate (default 20).")
    p.add_argument("--n-eval-val", type=int, default=20,
                   help="Number of val (held-out) configs to evaluate (default 20).")
    p.add_argument("--device", type=str, default="cuda",
                   help="torch device (default cuda).")
    p.add_argument("--output-dir", type=str,
                   default="results/regressor_sanity/evaluation",
                   help="Directory for evaluation results.")
    p.add_argument("--figures-dir", type=str,
                   default="figures/regressor_sanity",
                   help="Directory for figures.")
    p.add_argument("--train-log", type=str, default=None,
                   help="Path to train_log.csv for the loss-curve figure. "
                        "Default: <checkpoint_dir>/train_log.csv.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default 0).")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Run the regressor sanity-check evaluation."""
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 70)
    print(" Regressor sanity-check evaluation")
    print("=" * 70)
    print(f"  checkpoint:  {args.checkpoint}")
    print(f"  targets_dir: {args.targets_dir}")
    print(f"  n_eval_train: {args.n_eval_train}")
    print(f"  n_eval_val:   {args.n_eval_val}")
    print(f"  device:       {device}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  figures_dir:  {args.figures_dir}")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Load checkpoint + targets
    # ------------------------------------------------------------------
    print("\n[1/3] Loading regressor checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    model = EffectiveMapRegressor(
        RegressorConfig(
            feature_dim=cfg_dict.get("feature_dim", 14),
            hidden_dims=list(cfg_dict.get("hidden_dims", [256, 512, 512])),
            output_dim=cfg_dict.get("output_dim", 1056),
            activation=cfg_dict.get("activation", "gelu"),
            use_residual=cfg_dict.get("use_residual", True),
            use_layer_norm=cfg_dict.get("use_layer_norm", True),
            dropout=cfg_dict.get("dropout", 0.0),
        )
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    normalizer = WeightNormalizer.from_state_dict(ckpt["normalizer_state"])
    normalizer = normalizer.to(device)
    train_config_indices = ckpt["train_config_indices"]
    val_config_indices = ckpt["val_config_indices"]
    print(f"  params: {model.n_params():,}")
    print(f"  normalizer: {normalizer}")
    print(f"  train configs: {len(train_config_indices)}, "
          f"val configs: {len(val_config_indices)}")
    print(f"  split_source: {ckpt.get('split_source', 'unknown')}")

    print("\n[2/3] Loading targets + configs...")
    with open(os.path.join(args.targets_dir, "configs.json"), "r") as f:
        configs: List[Dict] = json.load(f)
    eff_maps = torch.load(
        os.path.join(args.targets_dir, "eff_maps.pt"),
        map_location="cpu").float()
    # Per-config average over MLPs (matches training).
    targets_all = eff_maps.mean(dim=1).contiguous()
    print(f"  eff_maps: {tuple(eff_maps.shape)} -> per-config "
          f"{tuple(targets_all.shape)}")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print(f"\n[3/3] Running evaluation ({args.n_eval_train} train + "
          f"{args.n_eval_val} val configs)...")
    results = evaluate_regressor(
        model=model,
        normalizer=normalizer,
        configs=configs,
        targets_all=targets_all,
        train_config_indices=train_config_indices,
        val_config_indices=val_config_indices,
        n_eval_train=args.n_eval_train,
        n_eval_val=args.n_eval_val,
        device=device,
        L=DEFAULT_L,
        H=DEFAULT_H,
        seed=args.seed,
        verbose=True,
    )
    summary = aggregate_regressor_results(results)
    print_regressor_summary(results, summary)
    save_regressor_results(results, summary, args.output_dir)

    if not args.no_figures:
        print(f"\nGenerating figures -> {args.figures_dir}/...")
        train_log = args.train_log
        if train_log is None:
            train_log = os.path.join(os.path.dirname(args.checkpoint),
                                     "train_log.csv")
        paths = generate_regressor_figures(
            results=results,
            configs=configs,
            model=model,
            normalizer=normalizer,
            targets_all=targets_all,
            device=device,
            figures_dir=args.figures_dir,
            train_log_csv=train_log,
            L=DEFAULT_L,
        )
        print(f"  generated {len(paths)} figures: {paths}")

    print("\n" + "=" * 70)
    print(" Regressor sanity-check evaluation complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
