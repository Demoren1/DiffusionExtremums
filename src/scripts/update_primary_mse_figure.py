"""Generate the primary ``figures/mse_comparison.png`` (deterministic baseline).

Builds a single bar chart of held-out functional test MSE comparing the
deterministic oracle-regressor baseline against the oracle convolution, the
learned convolution, the from-scratch MLP, and the target MLP.

This only overwrites ``figures/mse_comparison.png``.

Run with the conda env activated:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
    python -m src.scripts.update_primary_mse_figure
"""
import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Generate the primary figures/mse_comparison.png "
                    "(deterministic oracle-regressor baseline).")
    p.add_argument("--eval-summary", type=str,
                   default="results/evaluation/summary.json",
                   help="Path to the 3-method evaluation summary.json "
                        "(used for oracle_conv / learned_conv / "
                        "from-scratch MLP values).")
    p.add_argument("--oracle-summary", type=str,
                   default="results/regressor_oracle/evaluation/summary.json",
                   help="Path to the oracle-regressor evaluation summary.json.")
    p.add_argument("--out", type=str, default="figures/mse_comparison.png",
                   help="Output figure path.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Generate the primary held-out MSE comparison figure."""
    args = parse_args(argv)
    eval_path = args.eval_summary
    oracle_path = args.oracle_summary
    out_path = args.out

    if not os.path.exists(eval_path):
        print(f"[update_primary_mse_figure] {eval_path} not found; skipping")
        return 1
    if not os.path.exists(oracle_path):
        print(f"[update_primary_mse_figure] {oracle_path} not found; skipping")
        return 1

    with open(eval_path) as f:
        evl = json.load(f)
    with open(oracle_path) as f:
        orc = json.load(f)

    # Collect held-out (val) functional MSE means/stds for each method.
    methods = [
        ("oracle conv", evl["by_method"]["oracle_conv"]["val"]),
        ("learned conv", evl["by_method"]["learned_conv"]["val"]),
        ("from-scratch MLP", evl["by_method"]["from_scratch_mlp"]["val"]),
        ("target MLP", orc["functional_mse"]["target_mlp"]["val"]),
        ("oracle regressor", orc["functional_mse"]["oracle_regressor"]["val"]),
    ]
    colors = ["#2ca02c", "#2ca02c", "#9467bd", "#1f77b4", "#ff7f0e"]

    labels = [m[0] for m in methods]
    means = [m[1]["mean"] for m in methods]
    stds = [m[1].get("std", 0.0) or 0.0 for m in methods]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, 0.6, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Held-out functional test MSE")
    ax.set_title("Held-out functional test MSE: deterministic "
                 "oracle-regressor baseline\n"
                 "(oracle conv, learned conv, from-scratch MLP, "
                 "target MLP, oracle regressor)")
    ax.grid(axis="y", alpha=0.3)
    # Annotate bars with their mean value.
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + max(means) * 0.02,
                f"{m:.4g}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")

    # Annotate the ratio.
    ratio = orc.get("ratio_to_oracle_conv", {}).get("val")
    if ratio is not None:
        ax.text(0.02, 0.97,
                f"oracle regressor / oracle conv = {ratio:.2f}x (held-out)",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[update_primary_mse_figure] saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())