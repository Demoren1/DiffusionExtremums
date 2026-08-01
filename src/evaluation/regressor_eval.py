"""Evaluation for the deterministic effective-map regressor (sanity check).

Computes the full metric suite required by the task spec:

1. Normalized target-space MSE (train and val).
2. Raw effective-map MSE (train and val).
3. Toeplitz-ness (oracle / target / predicted), reusing the exact
   definition from ``src.evaluation.evaluate.toeplitzness``.
4. Kernel recovery: cosine similarity + L2 distance between the kernel
   recovered from the predicted M and the ground-truth kernel, reusing
   ``src.evaluation.evaluate.recover_kernel_from_matrix`` / ``kernel_recovery``.
5. Functional test MSE: convert the predicted effective map to an MLP via
   ``instantiate_mlp_from_eff_map`` (SVD factorization) and evaluate on
   generated data for >= 20 train + 20 held-out configs. Compared with the
   oracle convolution and the target effective-map MLP.
6. Saves a concise JSON/CSV metrics artifact and figures.

Reuses the established evaluation utilities (``toeplitzness``,
``recover_kernel_from_matrix``, ``kernel_recovery``, ``generate_dataset``,
``instantiate_mlp_from_eff_map``) rather than duplicating them.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import conv1d_same, generate_dataset
from src.evaluation._shared import (
    collect_eval_lists,
    config_dict_to_datasetconfig,
    mse,
    sample_eval_config_indices,
    stats_of,
    write_summary_csv,
)
from src.evaluation.evaluate import (
    kernel_recovery,
    recover_kernel_from_matrix,
    toeplitzness,
)
from src.evaluation.visualize import (
    plot_functional_mse_boxplot,
    plot_loss_curve as _plot_loss_curve,
    plot_toeplitz_comparison as _plot_toeplitz_comparison,
)
from src.models.effective_map import (
    DEFAULT_H,
    DEFAULT_L,
    effective_map_to_matrix,
    instantiate_mlp_from_eff_map,
    kernel_to_effective_map,
)
from src.models.effective_map_regressor import EffectiveMapRegressor
from src.models.weight_norm import WeightNormalizer


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

@dataclass
class RegressorConfigEval:
    """Evaluation result for one config.

    Attributes:
        config_idx: Index into configs.json.
        split: "train" or "val".
        family: Dataset family.
        radius: Kernel radius.
        noise_std: Noise std.
        norm_mse: Normalized target-space MSE.
        raw_mse: Raw effective-map MSE.
        toeplitz: Dict source -> score (oracle/target/predicted).
        kernel_recovery: Dict with cosine_sim, l2_dist (predicted vs GT).
        functional: Dict with test MSEs for oracle_conv, target_mlp, predicted_mlp.
    """

    config_idx: int
    split: str
    family: str
    radius: int
    noise_std: float
    norm_mse: float
    raw_mse: float
    toeplitz: Dict[str, float] = field(default_factory=dict)
    kernel_recovery: Dict[str, float] = field(default_factory=dict)
    functional: Dict[str, float] = field(default_factory=dict)


def evaluate_one_config(
    config: Dict,
    config_idx: int,
    split: str,
    model: EffectiveMapRegressor,
    normalizer: WeightNormalizer,
    target_eff: torch.Tensor,
    device: torch.device,
    L: int = DEFAULT_L,
    H: int = DEFAULT_H,
    seed: int = 0,
) -> RegressorConfigEval:
    """Evaluate the regressor on one config: MSE, Toeplitz, kernel, functional.

    Args:
        config: Config dict (from configs.json).
        config_idx: Index into configs.json.
        split: "train" or "val".
        model: The trained regressor (in eval mode).
        normalizer: The fitted WeightNormalizer (for destandardization).
        target_eff: ``[D_eff]`` target effective map (per-config average).
        device: torch device.
        L, H: MLP dimensions.
        seed: RNG seed for dataset generation.

    Returns:
        A ``RegressorConfigEval``.
    """
    model.eval()
    ds_config = config_dict_to_datasetconfig(config)
    family = str(config["family"])
    radius = int(config["radius"])
    noise_std = float(config["noise_std"])

    # Predict (normalized) and destandardize to raw effective-map space.
    from src.models.config_encoder import config_to_features
    feats = config_to_features(config, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(feats)  # [1, D_eff]
    pred_raw = normalizer.destandardize(pred_norm).squeeze(0).cpu()  # [D_eff]
    target_raw = target_eff.cpu().float()  # [D_eff]
    target_norm = normalizer.standardize(target_raw).to(device)

    # 1+2. MSE in normalized and raw space.
    norm_mse = mse(pred_norm.squeeze(0).cpu(), target_norm.cpu())
    raw_mse = mse(pred_raw, target_raw)

    result = RegressorConfigEval(
        config_idx=config_idx, split=split, family=family,
        radius=radius, noise_std=noise_std,
        norm_mse=norm_mse, raw_mse=raw_mse,
    )

    # 3. Toeplitz-ness: oracle / target / predicted.
    oracle_M = effective_map_to_matrix(
        kernel_to_effective_map(torch.tensor(ds_config.kernel, dtype=torch.float32),
                                L=L), L=L)
    target_M = effective_map_to_matrix(target_raw, L=L)
    pred_M = effective_map_to_matrix(pred_raw, L=L)
    result.toeplitz = {
        "oracle": toeplitzness(oracle_M),
        "target": toeplitzness(target_M),
        "predicted": toeplitzness(pred_M),
    }

    # 4. Kernel recovery (predicted M -> recovered kernel vs GT kernel).
    result.kernel_recovery = kernel_recovery(
        pred_M, torch.tensor(ds_config.kernel, dtype=torch.float32),
        max_radius=3)

    # 5. Functional test MSE.
    x_train, y_train, x_test, y_test, kernel, _ = generate_dataset(ds_config)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    kernel = kernel.to(device)

    # Oracle convolution.
    k_np = kernel.reshape(-1).cpu().numpy().astype(np.float32)
    x_test_np = x_test.cpu().numpy().astype(np.float32)
    pred_oracle = torch.from_numpy(conv1d_same(x_test_np, k_np)).to(device)
    oracle_mse = mse(pred_oracle, y_test)

    # Target effective-map MLP.
    target_mlp = instantiate_mlp_from_eff_map(target_raw, L=L, H=H).to(device)
    with torch.no_grad():
        target_mse = mse(target_mlp(x_test), y_test)

    # Predicted effective-map MLP.
    pred_mlp = instantiate_mlp_from_eff_map(pred_raw, L=L, H=H).to(device)
    with torch.no_grad():
        pred_mse = mse(pred_mlp(x_test), y_test)

    result.functional = {
        "oracle_conv": oracle_mse,
        "target_mlp": target_mse,
        "predicted_mlp": pred_mse,
    }

    return result


# ---------------------------------------------------------------------------
# Full evaluation over many configs
# ---------------------------------------------------------------------------

def evaluate_regressor(
    model: EffectiveMapRegressor,
    normalizer: WeightNormalizer,
    configs: List[Dict],
    targets_all: torch.Tensor,
    train_config_indices: List[int],
    val_config_indices: List[int],
    n_eval_train: int = 20,
    n_eval_val: int = 20,
    device: torch.device = torch.device("cpu"),
    L: int = DEFAULT_L,
    H: int = DEFAULT_H,
    seed: int = 0,
    verbose: bool = True,
) -> List[RegressorConfigEval]:
    """Evaluate the regressor on a sample of train + val configs.

    Args:
        model: Trained regressor (eval mode).
        normalizer: Fitted WeightNormalizer.
        configs: Full list of config dicts.
        targets_all: ``[n_configs, D_eff]`` per-config target effective maps.
        train_config_indices, val_config_indices: Config indices for splits.
        n_eval_train, n_eval_val: Number of train/val configs to evaluate.
        device: torch device.
        L, H: MLP dimensions.
        seed: RNG seed.
        verbose: Print progress.

    Returns:
        List of ``RegressorConfigEval``.
    """
    model = model.to(device)
    model.eval()
    # Sample configs to evaluate (deterministic, same scheme across eval modules).
    all_indices = sample_eval_config_indices(
        train_config_indices, val_config_indices,
        n_eval_train, n_eval_val, seed)

    results: List[RegressorConfigEval] = []
    for k, (idx, split) in enumerate(all_indices):
        if verbose:
            print(f"  [{k+1}/{len(all_indices)}] config {idx} ({split}, "
                  f"family={configs[idx]['family']}, "
                  f"radius={configs[idx]['radius']})")
        res = evaluate_one_config(
            configs[idx], idx, split, model, normalizer,
            targets_all[idx], device, L=L, H=H, seed=seed + k)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_regressor_results(
    results: List[RegressorConfigEval],
) -> Dict[str, Any]:
    """Aggregate per-config results into summary statistics."""
    splits = ["train", "val"]
    lists = collect_eval_lists(
        results,
        toeplitz_sources=["oracle", "target", "predicted"],
        functional_methods=["oracle_conv", "target_mlp", "predicted_mlp"],
    )
    return {
        "norm_mse": {s: stats_of(lists["norm_mse"][s]) for s in splits},
        "raw_mse": {s: stats_of(lists["raw_mse"][s]) for s in splits},
        "toeplitzness": {src: {s: stats_of(v) for s, v in sd.items()}
                         for src, sd in lists["toeplitz"].items()},
        "kernel_recovery": {s: {k: stats_of(v) for k, v in lists["kernel_rec"][s].items()}
                            for s in splits},
        "functional_mse": {m: {s: stats_of(v) for s, v in sd.items()}
                           for m, sd in lists["functional"].items()},
    }


def print_regressor_summary(results: List[RegressorConfigEval],
                            summary: Dict) -> None:
    """Print a human-readable summary table."""
    splits = ["train", "val"]
    print("\n" + "=" * 80)
    print(" REGRESSOR SANITY-CHECK RESULTS")
    print("=" * 80)

    print("\n--- Normalized target-space MSE ---")
    print(f"{'Split':<10} {'mean':>14} {'std':>14} {'n':>4}")
    for s in splits:
        st = summary["norm_mse"][s]
        print(f"{s:<10} {st['mean']:.6f}      {st['std']:.6f}      {st['n']}")

    print("\n--- Raw effective-map MSE ---")
    print(f"{'Split':<10} {'mean':>14} {'std':>14} {'n':>4}")
    for s in splits:
        st = summary["raw_mse"][s]
        print(f"{s:<10} {st['mean']:.6f}      {st['std']:.6f}      {st['n']}")

    print("\n--- Toeplitz-ness (mean diagonal std; lower = more Toeplitz) ---")
    print(f"{'Source':<12} {'train':>14} {'val':>14}")
    for src in ["oracle", "target", "predicted"]:
        row = f"{src:<12}"
        for s in splits:
            st = summary["toeplitzness"][src][s]
            row += f"  {st['mean']:.6f}±{st['std']:.6f}".rjust(18) + "  "
        print(row)

    print("\n--- Kernel recovery (predicted M -> recovered kernel vs GT) ---")
    print(f"{'Metric':<14} {'train':>14} {'val':>14}")
    for k in ["cosine_sim", "l2_dist"]:
        row = f"{k:<14}"
        for s in splits:
            st = summary["kernel_recovery"][s][k]
            row += f"  {st['mean']:.6f}±{st['std']:.6f}".rjust(18) + "  "
        print(row)

    print("\n--- Functional test MSE ---")
    print(f"{'Method':<16} {'train':>14} {'val':>14}")
    for m in ["oracle_conv", "target_mlp", "predicted_mlp"]:
        row = f"{m:<16}"
        for s in splits:
            st = summary["functional_mse"][m][s]
            row += f"  {st['mean']:.6f}±{st['std']:.6f}".rjust(18) + "  "
        print(row)
    print("=" * 80)


def save_regressor_results(
    results: List[RegressorConfigEval],
    summary: Dict,
    output_dir: str,
) -> None:
    """Save evaluation results (metrics) as JSON and CSV to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)

    full = [asdict(r) for r in results]
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(full, f, indent=2)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_summary_csv(summary, os.path.join(output_dir, "summary.csv"))

    print(f"\nResults saved to {output_dir}/ (results.json, summary.json, "
          f"summary.csv)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_loss_curve(train_log_csv: str, figures_dir: str) -> str:
    """Loss curve from train_log.csv."""
    return _plot_loss_curve(train_log_csv, figures_dir,
                            title="Regressor training loss curve")


def plot_eff_map_heatmaps(
    results: List[RegressorConfigEval],
    configs: List[Dict],
    model: EffectiveMapRegressor,
    normalizer: WeightNormalizer,
    targets_all: torch.Tensor,
    device: torch.device,
    figures_dir: str,
    L: int = DEFAULT_L,
    n_per_split: int = 3,
) -> str:
    """Heatmaps of oracle / target / predicted M for representative held-out configs."""
    from src.models.config_encoder import config_to_features

    # Pick representative val configs (one per family if possible).
    seen: Dict[str, RegressorConfigEval] = {}
    for r in results:
        if r.split == "val" and r.family not in seen:
            seen[r.family] = r
    reps = [seen[f] for f in list(seen)[:n_per_split]]
    if not reps:
        reps = [r for r in results if r.split == "val"][:n_per_split]
    if not reps:
        print("[visualize] no val configs for heatmaps; skipping")
        return ""
    n = len(reps)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(reps):
        cfg = configs[r.config_idx]
        ds_cfg = config_dict_to_datasetconfig(cfg)
        oracle_M = effective_map_to_matrix(
            kernel_to_effective_map(torch.tensor(ds_cfg.kernel, dtype=torch.float32),
                                    L=L), L=L)
        target_M = effective_map_to_matrix(targets_all[r.config_idx], L=L)
        feats = config_to_features(cfg, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(feats)
        pred_raw = normalizer.destandardize(pred_norm).squeeze(0).cpu()
        pred_M = effective_map_to_matrix(pred_raw, L=L)
        for col, (M, title) in enumerate([
            (oracle_M, "Oracle M"),
            (target_M, "Target M"),
            (pred_M, "Predicted M"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(M.detach().cpu().numpy(), cmap="RdBu_r",
                           aspect="auto", interpolation="nearest")
            ax.set_title(f"{r.family} (r={r.radius}) — {title}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Effective map M: Oracle vs Target vs Predicted (held-out val configs)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(figures_dir, "eff_map_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


def plot_toeplitz_comparison(results: List[RegressorConfigEval],
                             figures_dir: str) -> str:
    """Bar chart of Toeplitz-ness by source (oracle/target/predicted)."""
    sources = ["oracle", "target", "predicted"]
    colors = {"oracle": "#2ca02c", "target": "#1f77b4",
              "predicted": "#ff7f0e"}
    return _plot_toeplitz_comparison(results, figures_dir, sources, colors)


def plot_functional_mse(results: List[RegressorConfigEval],
                       figures_dir: str) -> str:
    """Box plot of functional test MSE: oracle / target / predicted, train vs val."""
    methods = ["oracle_conv", "target_mlp", "predicted_mlp"]
    return plot_functional_mse_boxplot(
        results, figures_dir,
        methods=methods, labels=methods,
        colors=["#2ca02c", "#1f77b4", "#ff7f0e"],
        suptitle="Functional test MSE: oracle conv vs target MLP vs predicted MLP",
        tick_rotation=15)


def generate_regressor_figures(
    results: List[RegressorConfigEval],
    configs: List[Dict],
    model: EffectiveMapRegressor,
    normalizer: WeightNormalizer,
    targets_all: torch.Tensor,
    device: torch.device,
    figures_dir: str,
    train_log_csv: Optional[str] = None,
    L: int = DEFAULT_L,
) -> List[str]:
    """Generate all regressor sanity-check figures."""
    os.makedirs(figures_dir, exist_ok=True)
    paths: List[str] = []
    if train_log_csv:
        p = plot_loss_curve(train_log_csv, figures_dir)
        if p:
            paths.append(p)
    p = plot_eff_map_heatmaps(results, configs, model, normalizer, targets_all,
                              device, figures_dir, L=L)
    if p:
        paths.append(p)
    p = plot_toeplitz_comparison(results, figures_dir)
    if p:
        paths.append(p)
    p = plot_functional_mse(results, figures_dir)
    if p:
        paths.append(p)
    return paths
