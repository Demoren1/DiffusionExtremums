"""Oracle-target regressor evaluation (oracle-quality baseline).

Extends the sanity-check evaluation (:mod:`src.evaluation.regressor_eval`) for
the oracle-target deterministic regressor baseline. Reports the full metric
suite required by the task spec on at least 20 train + 20 held-out configs:

1. Target-space normalized/raw MSE.
2. Toeplitz-ness: oracle map, predicted map, target MLP map.
3. Kernel recovery cosine/L2.
4. Functional MSE on generated x/y test data for:
   (a) oracle convolution,
   (b) target MLP effective map (the learned MLP-averaged map, if relevant),
   (c) oracle-target deterministic regressor (the trained regressor).

Train and held-out values are reported separately. The success condition is
that the held-out regressor functional MSE should be close to the oracle noise
floor (~0.01); the ratio is quantified.
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
from src.evaluation.evaluate import kernel_recovery, toeplitzness
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
)
from src.models.effective_map_regressor import EffectiveMapRegressor
from src.models.weight_norm import WeightNormalizer


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

@dataclass
class OracleRegressorConfigEval:
    """Evaluation result for one config under the oracle-target baseline.

    Attributes:
        config_idx: Index into configs.json.
        split: "train" or "val".
        family: Dataset family.
        radius: Kernel radius.
        noise_std: Noise std.
        norm_mse: Normalized target-space MSE (regressor vs oracle target).
        raw_mse: Raw effective-map MSE (regressor vs oracle target).
        toeplitz: Dict source -> score (oracle/target_mlp/predicted).
        kernel_recovery: Dict with cosine_sim, l2_dist (predicted vs GT).
        functional: Dict with test MSEs for oracle_conv, target_mlp,
            oracle_regressor.
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
    functional: Dict[str, Optional[float]] = field(default_factory=dict)


def evaluate_one_config_oracle(
    config: Dict,
    config_idx: int,
    split: str,
    model: EffectiveMapRegressor,
    normalizer: WeightNormalizer,
    oracle_target: torch.Tensor,
    learned_target: torch.Tensor,
    device: torch.device,
    L: int = DEFAULT_L,
    H: int = DEFAULT_H,
    seed: int = 0,
) -> OracleRegressorConfigEval:
    """Evaluate the oracle-target regressor on one config.

    Args:
        config: Config dict (from configs.json).
        config_idx: Index into configs.json.
        split: "train" or "val".
        model: The trained regressor (eval mode).
        normalizer: The fitted WeightNormalizer (for destandardization).
        oracle_target: ``[D_eff]`` exact oracle effective map for this config.
        learned_target: ``[D_eff]`` MLP-averaged learned target (for the
            target-MLP functional baseline and Toeplitz comparison).
        device: torch device.
        L, H: MLP dimensions.
        seed: RNG seed for dataset generation.

    Returns:
        An ``OracleRegressorConfigEval``.
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
    oracle_raw = oracle_target.cpu().float()  # [D_eff]
    oracle_norm = normalizer.standardize(oracle_raw).to(device)
    learned_raw = learned_target.cpu().float()

    # 1+2. MSE in normalized and raw space (regressor vs oracle target).
    norm_mse = mse(pred_norm.squeeze(0).cpu(), oracle_norm.cpu())
    raw_mse = mse(pred_raw, oracle_raw)

    result = OracleRegressorConfigEval(
        config_idx=config_idx, split=split, family=family,
        radius=radius, noise_std=noise_std,
        norm_mse=norm_mse, raw_mse=raw_mse,
    )

    # 3. Toeplitz-ness: oracle / target_mlp / predicted.
    oracle_M = effective_map_to_matrix(oracle_raw, L=L)
    target_M = effective_map_to_matrix(learned_raw, L=L)
    pred_M = effective_map_to_matrix(pred_raw, L=L)
    result.toeplitz = {
        "oracle": toeplitzness(oracle_M),
        "target_mlp": toeplitzness(target_M),
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

    # (a) Oracle convolution.
    k_np = kernel.reshape(-1).cpu().numpy().astype(np.float32)
    x_test_np = x_test.cpu().numpy().astype(np.float32)
    pred_oracle = torch.from_numpy(conv1d_same(x_test_np, k_np)).to(device)
    oracle_mse = mse(pred_oracle, y_test)

    # (b) Target MLP effective map (learned MLP-averaged map).
    target_mlp = instantiate_mlp_from_eff_map(learned_raw, L=L, H=H).to(device)
    with torch.no_grad():
        target_mse = mse(target_mlp(x_test), y_test)

    # (c) Oracle-target deterministic regressor.
    pred_mlp = instantiate_mlp_from_eff_map(pred_raw, L=L, H=H).to(device)
    with torch.no_grad():
        pred_mse = mse(pred_mlp(x_test), y_test)

    result.functional = {
        "oracle_conv": oracle_mse,
        "target_mlp": target_mse,
        "oracle_regressor": pred_mse,
    }

    return result


# ---------------------------------------------------------------------------
# Full evaluation over many configs
# ---------------------------------------------------------------------------

def evaluate_oracle_regressor(
    model: EffectiveMapRegressor,
    normalizer: WeightNormalizer,
    configs: List[Dict],
    oracle_targets_all: torch.Tensor,
    learned_targets_all: torch.Tensor,
    train_config_indices: List[int],
    val_config_indices: List[int],
    n_eval_train: int = 20,
    n_eval_val: int = 20,
    device: torch.device = torch.device("cpu"),
    L: int = DEFAULT_L,
    H: int = DEFAULT_H,
    seed: int = 0,
    verbose: bool = True,
) -> List[OracleRegressorConfigEval]:
    """Evaluate the oracle-target regressor on a sample of train + val configs.

    Args:
        model: Trained regressor (eval mode).
        normalizer: Fitted WeightNormalizer.
        configs: Full list of config dicts.
        oracle_targets_all: ``[n_configs, D_eff]`` exact oracle effective maps.
        learned_targets_all: ``[n_configs, D_eff]`` MLP-averaged learned maps.
        train_config_indices, val_config_indices: Config indices for splits.
        n_eval_train, n_eval_val: Number of train/val configs to evaluate.
        device: torch device.
        L, H: MLP dimensions.
        seed: RNG seed.
        verbose: Print progress.

    Returns:
        List of ``OracleRegressorConfigEval``.
    """
    model = model.to(device)
    model.eval()
    # Sample configs to evaluate (deterministic, same scheme across eval modules).
    all_indices = sample_eval_config_indices(
        train_config_indices, val_config_indices,
        n_eval_train, n_eval_val, seed)

    results: List[OracleRegressorConfigEval] = []
    for k, (idx, split) in enumerate(all_indices):
        if verbose:
            print(f"  [{k+1}/{len(all_indices)}] config {idx} ({split}, "
                  f"family={configs[idx]['family']}, "
                  f"radius={configs[idx]['radius']})")
        res = evaluate_one_config_oracle(
            configs[idx], idx, split, model, normalizer,
            oracle_targets_all[idx], learned_targets_all[idx],
            device, L=L, H=H, seed=seed + k)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_oracle_results(
    results: List[OracleRegressorConfigEval],
) -> Dict[str, Any]:
    """Aggregate per-config results into summary statistics."""
    splits = ["train", "val"]
    lists = collect_eval_lists(
        results,
        toeplitz_sources=["oracle", "target_mlp", "predicted"],
        functional_methods=["oracle_conv", "target_mlp", "oracle_regressor"],
    )
    summary = {
        "norm_mse": {s: stats_of(lists["norm_mse"][s]) for s in splits},
        "raw_mse": {s: stats_of(lists["raw_mse"][s]) for s in splits},
        "toeplitzness": {src: {s: stats_of(v) for s, v in sd.items()}
                         for src, sd in lists["toeplitz"].items()},
        "kernel_recovery": {s: {k: stats_of(v) for k, v in lists["kernel_rec"][s].items()}
                            for s in splits},
        "functional_mse": {m: {s: stats_of(v) for s, v in sd.items()}
                           for m, sd in lists["functional"].items()},
    }
    # Ratio: oracle_regressor / oracle_conv (held-out).
    for s in splits:
        reg = summary["functional_mse"]["oracle_regressor"][s]["mean"]
        orc = summary["functional_mse"]["oracle_conv"][s]["mean"]
        ratio = (reg / orc) if (reg is not None and orc is not None
                                and orc > 1e-12) else None
        summary.setdefault("ratio_to_oracle_conv", {})[s] = ratio
    return summary


def print_oracle_summary(results: List[OracleRegressorConfigEval],
                        summary: Dict) -> None:
    """Print a human-readable summary table."""
    splits = ["train", "val"]
    print("\n" + "=" * 80)
    print(" ORACLE-TARGET REGRESSOR BASELINE RESULTS")
    print("=" * 80)

    print("\n--- Normalized target-space MSE (regressor vs oracle target) ---")
    print(f"{'Split':<10} {'mean':>14} {'std':>14} {'n':>4}")
    for s in splits:
        st = summary["norm_mse"][s]
        print(f"{s:<10} {st['mean']:.6f}      {st['std']:.6f}      {st['n']}")

    print("\n--- Raw effective-map MSE (regressor vs oracle target) ---")
    print(f"{'Split':<10} {'mean':>14} {'std':>14} {'n':>4}")
    for s in splits:
        st = summary["raw_mse"][s]
        print(f"{s:<10} {st['mean']:.6f}      {st['std']:.6f}      {st['n']}")

    print("\n--- Toeplitz-ness (mean diagonal std; lower = more Toeplitz) ---")
    print(f"{'Source':<14} {'train':>18} {'val':>18}")
    for src in ["oracle", "target_mlp", "predicted"]:
        row = f"{src:<14}"
        for s in splits:
            st = summary["toeplitzness"][src][s]
            row += f"  {st['mean']:.6f}±{st['std']:.6f}".rjust(18) + "  "
        print(row)

    print("\n--- Kernel recovery (predicted M -> recovered kernel vs GT) ---")
    print(f"{'Metric':<14} {'train':>18} {'val':>18}")
    for k in ["cosine_sim", "l2_dist"]:
        row = f"{k:<14}"
        for s in splits:
            st = summary["kernel_recovery"][s][k]
            row += f"  {st['mean']:.6f}±{st['std']:.6f}".rjust(18) + "  "
        print(row)

    print("\n--- Functional test MSE ---")
    print(f"{'Method':<20} {'train':>18} {'val':>18}")
    for m in ["oracle_conv", "target_mlp", "oracle_regressor"]:
        row = f"{m:<20}"
        for s in splits:
            st = summary["functional_mse"][m][s]
            if st["mean"] is not None:
                row += f"  {st['mean']:.6f}±{st['std']:.6f}".rjust(18) + "  "
            else:
                row += "  N/A".rjust(18) + "  "
        print(row)

    print("\n--- Ratio: oracle_regressor / oracle_conv ---")
    for s in splits:
        r = summary.get("ratio_to_oracle_conv", {}).get(s)
        if r is not None:
            print(f"  {s}: {r:.4f}x")
        else:
            print(f"  {s}: N/A")
    print("=" * 80)


def save_oracle_results(
    results: List[OracleRegressorConfigEval],
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

    extra_rows = [
        ["ratio_to_oracle_conv", "oracle_regressor", s, r, "", ""]
        for s, r in summary.get("ratio_to_oracle_conv", {}).items()
    ]
    write_summary_csv(summary, os.path.join(output_dir, "summary.csv"),
                      extra_rows=extra_rows)

    print(f"\nResults saved to {output_dir}/ (results.json, summary.json, "
          f"summary.csv)")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_loss_curve_oracle(train_log_csv: str, figures_dir: str) -> str:
    """Loss curve from train_log.csv."""
    return _plot_loss_curve(train_log_csv, figures_dir,
                            title="Oracle-target regressor training loss curve")


def plot_eff_map_heatmaps_oracle(
    results: List[OracleRegressorConfigEval],
    configs: List[Dict],
    model: EffectiveMapRegressor,
    normalizer: WeightNormalizer,
    oracle_targets_all: torch.Tensor,
    device: torch.device,
    figures_dir: str,
    L: int = DEFAULT_L,
    n_per_split: int = 3,
) -> str:
    """Heatmaps of oracle / predicted M for representative held-out configs."""
    from src.models.config_encoder import config_to_features

    seen: Dict[str, OracleRegressorConfigEval] = {}
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
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(reps):
        cfg = configs[r.config_idx]
        oracle_M = effective_map_to_matrix(oracle_targets_all[r.config_idx], L=L)
        feats = config_to_features(cfg, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(feats)
        pred_raw = normalizer.destandardize(pred_norm).squeeze(0).cpu()
        pred_M = effective_map_to_matrix(pred_raw, L=L)
        for col, (M, title) in enumerate([
            (oracle_M, "Oracle M"),
            (pred_M, "Predicted M"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(M.detach().cpu().numpy(), cmap="RdBu_r",
                           aspect="auto", interpolation="nearest")
            ax.set_title(f"{r.family} (r={r.radius}) — {title}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Effective map M: Oracle vs Predicted (held-out val configs) "
                 "— Toeplitz structure visible", fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(figures_dir, "eff_map_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


def plot_toeplitz_comparison_oracle(results: List[OracleRegressorConfigEval],
                                   figures_dir: str) -> str:
    """Bar chart of Toeplitz-ness by source (oracle/target_mlp/predicted)."""
    sources = ["oracle", "target_mlp", "predicted"]
    colors = {"oracle": "#2ca02c", "target_mlp": "#1f77b4",
              "predicted": "#ff7f0e"}
    return _plot_toeplitz_comparison(results, figures_dir, sources, colors)


def plot_kernel_recovery_oracle(results: List[OracleRegressorConfigEval],
                               figures_dir: str) -> str:
    """Kernel recovery: cosine similarity and L2 distance, train vs val."""
    splits = ["train", "val"]
    cos = {s: [] for s in splits}
    l2 = {s: [] for s in splits}
    for r in results:
        if "cosine_sim" in r.kernel_recovery:
            cos[r.split].append(r.kernel_recovery["cosine_sim"])
            l2[r.split].append(r.kernel_recovery["l2_dist"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col, (metric, data, ylabel) in enumerate([
        ("cosine_sim", cos, "cosine similarity"),
        ("l2_dist", l2, "L2 distance"),
    ]):
        ax = axes[col]
        positions = [1, 1.5]
        box_data = [data[s] if data[s] else [0.0] for s in splits]
        bp = ax.boxplot(box_data, positions=positions, widths=0.35,
                        labels=splits, showmeans=True, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#1f77b4", "#ff7f0e"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(f"Kernel recovery — {metric}")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Kernel recovery: predicted M -> recovered kernel vs ground truth",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(figures_dir, "kernel_recovery.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


def plot_functional_mse_oracle(results: List[OracleRegressorConfigEval],
                               figures_dir: str) -> str:
    """Functional MSE comparison for the deterministic oracle-regressor baseline.

    Plots the deterministic methods (oracle conv, target MLP, oracle regressor)
    on a linear scale appropriate for values ~0.01.
    """
    methods = ["oracle_conv", "target_mlp", "oracle_regressor"]
    labels = ["oracle conv", "target MLP", "oracle regressor"]
    return plot_functional_mse_boxplot(
        results, figures_dir,
        methods=methods, labels=labels,
        colors=["#2ca02c", "#1f77b4", "#ff7f0e"],
        suptitle="Functional test MSE: deterministic oracle-regressor "
                 "baseline (oracle conv vs target MLP vs oracle regressor)",
        tick_rotation=20,
        figsize=(13, 5.5))


def generate_oracle_figures(
    results: List[OracleRegressorConfigEval],
    configs: List[Dict],
    model: EffectiveMapRegressor,
    normalizer: WeightNormalizer,
    oracle_targets_all: torch.Tensor,
    device: torch.device,
    figures_dir: str,
    train_log_csv: Optional[str] = None,
    L: int = DEFAULT_L,
) -> List[str]:
    """Generate all oracle-regressor figures."""
    os.makedirs(figures_dir, exist_ok=True)
    paths: List[str] = []
    if train_log_csv:
        p = plot_loss_curve_oracle(train_log_csv, figures_dir)
        if p:
            paths.append(p)
    p = plot_eff_map_heatmaps_oracle(results, configs, model, normalizer,
                                     oracle_targets_all, device, figures_dir,
                                     L=L)
    if p:
        paths.append(p)
    p = plot_toeplitz_comparison_oracle(results, figures_dir)
    if p:
        paths.append(p)
    p = plot_kernel_recovery_oracle(results, figures_dir)
    if p:
        paths.append(p)
    p = plot_functional_mse_oracle(results, figures_dir)
    if p:
        paths.append(p)
    return paths