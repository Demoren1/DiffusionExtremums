"""Visualization: figures for the evaluation pipeline.

Generates and saves the following figures to ``figures/``:

1. ``eff_map_comparison.png`` — side-by-side heatmaps of the effective matrix M
   for oracle (ground-truth Toeplitz) and target (collected MLP) for one
   representative dataset per family.
2. ``toeplitz_analysis.png`` — bar chart of the Toeplitz-ness score for oracle
   and target M, across all 5 families.
3. ``kernel_recovery.png`` — line plots of the ground-truth kernel vs the
   recovered kernel (averaged diagonals of M) from target M, for a few datasets.
4. ``mse_comparison.png`` — box/bar chart of test MSE for each method
   (from-scratch MLP, learned conv, oracle conv) across all evaluated datasets,
   separated by train/val configs.
5. ``weight_distribution.png`` — PCA of target effective maps, colored by
   family.

All figures use ``matplotlib`` and are saved as PNG.

The module also hosts the *shared* plot helpers used by the regressor eval
modules (:mod:`src.evaluation.regressor_eval` and
:mod:`src.evaluation.oracle_regressor_eval`) — ``plot_loss_curve``,
``plot_toeplitz_comparison``, and ``plot_functional_mse_boxplot`` — so the
near-duplicate plotting code lives in one place.
"""
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.data.families import FAMILIES
from src.evaluation.evaluate import (
    DatasetEval,
    recover_kernel_from_matrix,
    toeplitzness,
)
from src.models.effective_map import (
    DEFAULT_L,
    effective_map_to_matrix,
    kernel_to_effective_map,
)


# ---------------------------------------------------------------------------
# Figure 1: Effective map comparison (heatmaps)
# ---------------------------------------------------------------------------

def plot_eff_map_comparison(
    results: List[DatasetEval],
    configs: List[Dict],
    target_eff_maps: Optional[torch.Tensor],
    figures_dir: str,
    L: int = DEFAULT_L,
) -> str:
    """Figure 1: side-by-side heatmaps of oracle / target M.

    Picks one representative dataset per family (the first evaluated config of
    each family). For each, shows 2 heatmaps: oracle M (ground-truth Toeplitz),
    target M (collected MLP).
    """
    from src.configs.base import DatasetConfig

    # Find one representative per family.
    seen: Dict[str, DatasetEval] = {}
    for r in results:
        if r.family not in seen:
            seen[r.family] = r
    families_present = [f for f in FAMILIES if f in seen]
    n_fam = len(families_present)
    if n_fam == 0:
        print("[visualize] no families found for eff_map_comparison; skipping")
        return ""

    fig, axes = plt.subplots(n_fam, 2, figsize=(9, 3.5 * n_fam))
    if n_fam == 1:
        axes = axes[np.newaxis, :]

    for row, fam in enumerate(families_present):
        r = seen[fam]
        cfg = configs[r.config_idx]
        # Oracle M.
        from src.data.dataset import generate_dataset
        ds_cfg = DatasetConfig(
            family=cfg["family"],
            kernel=tuple(float(v) for v in cfg["kernel"]),
            radius=int(cfg["radius"]),
            noise_std=float(cfg["noise_std"]),
            n_train=int(cfg["n_train"]),
            n_test=int(cfg.get("n_test", 512)),
            seed=int(cfg.get("seed", 0)),
            L=int(cfg.get("L", 32)),
        )
        _, _, _, _, kernel, _ = generate_dataset(ds_cfg)
        oracle_M = effective_map_to_matrix(kernel_to_effective_map(kernel, L=L))

        # Target M.
        if target_eff_maps is not None:
            tgt_M = effective_map_to_matrix(target_eff_maps[r.config_idx, 0], L=L)
        else:
            tgt_M = torch.zeros(L, L)

        for col, (M, title) in enumerate([
            (oracle_M, "Oracle M"),
            (tgt_M, "Target M"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(M.detach().cpu().numpy(), cmap="RdBu_r",
                           aspect="auto", interpolation="nearest")
            ax.set_title(f"{fam} — {title}", fontsize=10)
            ax.set_xlabel("j")
            ax.set_ylabel("i")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Effective Map M: Oracle vs Target (collected MLP)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(figures_dir, "eff_map_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 2: Toeplitz-ness analysis (bar chart)
# ---------------------------------------------------------------------------

def plot_toeplitz_analysis(
    results: List[DatasetEval],
    figures_dir: str,
) -> str:
    """Figure 2: bar chart of Toeplitz-ness score by source and family."""
    sources = ["oracle", "target"]
    # Collect per-family means.
    fam_data: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        if r.family not in fam_data:
            fam_data[r.family] = {s: [] for s in sources}
        for s in sources:
            if s in r.toeplitzness:
                fam_data[r.family][s].append(r.toeplitzness[s])

    families = [f for f in FAMILIES if f in fam_data]
    n_fam = len(families)
    if n_fam == 0:
        print("[visualize] no families for toeplitz_analysis; skipping")
        return ""

    x = np.arange(n_fam)
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(8, n_fam * 1.5), 5))
    colors = {"oracle": "#2ca02c", "target": "#1f77b4"}
    for i, s in enumerate(sources):
        means = [np.mean(fam_data[f][s]) if fam_data[f][s] else 0.0 for f in families]
        stds = [np.std(fam_data[f][s]) if fam_data[f][s] else 0.0 for f in families]
        ax.bar(x + i * width, means, width, yerr=stds, label=s, capsize=3,
               color=colors[s], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(families)
    ax.set_ylabel("Toeplitz-ness score (mean diagonal std)")
    ax.set_title("Toeplitz-ness: lower = more convolution-like\n"
                 "(oracle ~0 = true convolution)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "toeplitz_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 3: Kernel recovery (line plots)
# ---------------------------------------------------------------------------

def plot_kernel_recovery(
    results: List[DatasetEval],
    configs: List[Dict],
    target_eff_maps: Optional[torch.Tensor],
    figures_dir: str,
    L: int = DEFAULT_L,
) -> str:
    """Figure 3: ground-truth vs recovered (target) kernel."""
    from src.configs.base import DatasetConfig
    from src.data.dataset import generate_dataset

    # Pick a few representative datasets (one per family, up to 5).
    seen: Dict[str, DatasetEval] = {}
    for r in results:
        if r.family not in seen:
            seen[r.family] = r
    families_present = [f for f in FAMILIES if f in seen]
    n = len(families_present)
    if n == 0:
        print("[visualize] no families for kernel_recovery; skipping")
        return ""

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for col, fam in enumerate(families_present):
        r = seen[fam]
        cfg = configs[r.config_idx]
        ds_cfg = DatasetConfig(
            family=cfg["family"],
            kernel=tuple(float(v) for v in cfg["kernel"]),
            radius=int(cfg["radius"]),
            noise_std=float(cfg["noise_std"]),
            n_train=int(cfg["n_train"]),
            n_test=int(cfg.get("n_test", 512)),
            seed=int(cfg.get("seed", 0)),
            L=int(cfg.get("L", 32)),
        )
        _, _, _, _, kernel, _ = generate_dataset(ds_cfg)
        gt_k = kernel.cpu().numpy()
        r_gt = len(gt_k) // 2
        max_radius = 3
        K = 2 * max_radius + 1
        gt_padded = np.zeros(K)
        off = max_radius - r_gt
        gt_padded[off:off + len(gt_k)] = gt_k

        if target_eff_maps is not None:
            tgt_M = effective_map_to_matrix(target_eff_maps[r.config_idx, 0], L=L)
            rec_tgt = recover_kernel_from_matrix(tgt_M, max_radius=max_radius).numpy()
        else:
            rec_tgt = np.zeros(K)

        ax = axes[0, col]
        x = np.arange(K) - max_radius
        ax.plot(x, gt_padded, "go-", label="Ground truth", linewidth=2, markersize=5)
        ax.plot(x, rec_tgt, "s--", color="#1f77b4", label="Recovered (target)", linewidth=1.5)
        ax.set_title(f"{fam} (r={cfg['radius']})", fontsize=10)
        ax.set_xlabel("kernel position")
        ax.set_ylabel("kernel value")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Kernel recovery: ground truth vs averaged diagonals of M", fontsize=12)
    fig.tight_layout()
    path = os.path.join(figures_dir, "kernel_recovery.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 4: Test MSE comparison (box plot)
# ---------------------------------------------------------------------------

def plot_mse_comparison(
    results: List[DatasetEval],
    figures_dir: str,
) -> str:
    """Figure 4: box plot of test MSE for each method, train vs val configs."""
    methods = ["from_scratch_mlp", "learned_conv", "oracle_conv"]
    splits = ["train", "val"]
    data: Dict[str, Dict[str, List[float]]] = {m: {s: [] for s in splits} for m in methods}
    for r in results:
        for m in methods:
            if m in r.methods:
                data[m][r.split].append(r.methods[m].test_mse)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for col, s in enumerate(splits):
        ax = axes[col]
        box_data = [data[m][s] if data[m][s] else [0.0] for m in methods]
        bp = ax.boxplot(box_data, labels=methods, showmeans=True, patch_artist=True)
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(f"Test MSE — {s} configs")
        ax.set_ylabel("Test MSE" if col == 0 else "")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Test MSE comparison across methods", fontsize=12)
    fig.tight_layout()
    path = os.path.join(figures_dir, "mse_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 5: Weight distribution (PCA)
# ---------------------------------------------------------------------------

def plot_weight_distribution(
    target_eff_maps: torch.Tensor,
    configs: List[Dict],
    figures_dir: str,
    n_target_per_family: int = 100,
    L: int = DEFAULT_L,
) -> str:
    """Figure 5: PCA of target effective maps, colored by family.

    Projects the 1056-dim target effective maps to 2D via PCA and scatters them,
    colored by family.
    """
    from src.models.config_encoder import configs_to_features

    n_configs, n_mlp, D_eff = target_eff_maps.shape
    # Subsample targets: n_target_per_family per family.
    family_to_indices: Dict[str, List[int]] = {}
    for i, c in enumerate(configs):
        family_to_indices.setdefault(str(c["family"]), []).append(i)
    target_sel: List[int] = []
    target_families: List[str] = []
    for fam, idxs in family_to_indices.items():
        take = idxs[:max(1, n_target_per_family // n_mlp)]
        for i in take:
            target_sel.append(i)
            target_families.append(fam)
    if not target_sel:
        print("[visualize] no targets for weight_distribution; skipping")
        return ""

    tgt = target_eff_maps[target_sel].reshape(len(target_sel) * n_mlp, D_eff).cpu().numpy()
    tgt_fam_expanded = []
    for fam in target_families:
        tgt_fam_expanded.extend([fam] * n_mlp)

    # PCA on the targets.
    mu = tgt.mean(axis=0)
    centered = tgt - mu
    # SVD-based PCA (top-2 components).
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    tgt_proj = centered @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(9, 7))
    fam_colors = {f: plt.cm.tab10(i) for i, f in enumerate(FAMILIES)}
    # Plot targets.
    for fam in FAMILIES:
        mask = np.array([f == fam for f in tgt_fam_expanded])
        if mask.any():
            ax.scatter(tgt_proj[mask, 0], tgt_proj[mask, 1], c=[fam_colors[fam]],
                       marker="o", s=15, alpha=0.4, label=f"target {fam}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Weight distribution: target effective maps (PCA)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "weight_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(
    results: List[DatasetEval],
    configs: List[Dict],
    target_eff_maps: Optional[torch.Tensor],
    figures_dir: str,
    L: int = DEFAULT_L,
) -> List[str]:
    """Generate all figures and return their paths."""
    os.makedirs(figures_dir, exist_ok=True)
    paths: List[str] = []

    # Figure 1: eff map comparison.
    if target_eff_maps is not None:
        p = plot_eff_map_comparison(
            results, configs, target_eff_maps, figures_dir, L=L)
        if p:
            paths.append(p)

    # Figure 2: Toeplitz-ness.
    p = plot_toeplitz_analysis(results, figures_dir)
    if p:
        paths.append(p)

    # Figure 3: kernel recovery.
    if target_eff_maps is not None:
        p = plot_kernel_recovery(
            results, configs, target_eff_maps, figures_dir, L=L)
        if p:
            paths.append(p)

    # Figure 4: MSE comparison.
    p = plot_mse_comparison(results, figures_dir)
    if p:
        paths.append(p)

    # Figure 5: weight distribution.
    if target_eff_maps is not None:
        p = plot_weight_distribution(
            target_eff_maps, configs, figures_dir, L=L)
        if p:
            paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Shared plot helpers for the regressor eval modules
# ---------------------------------------------------------------------------
# The learned-target and oracle-target regressor evaluations each produce the
# same figures (training loss, Toeplitz-ness, functional MSE). The helpers below
# parameterize the only things that differ (title, sources, method labels,
# colors), so the two modules keep thin wrappers instead of duplicated code.

def plot_loss_curve(train_log_csv: str, figures_dir: str,
                    title: str = "Regressor training loss curve") -> str:
    """Plot the training loss curve from a ``train_log.csv``.

    Shared by the regressor eval modules; ``title`` lets each caller customize
    the figure title. Saves ``training_loss.png``.

    Args:
        train_log_csv: Path to ``train_log.csv`` (columns step/train_loss/val_loss).
        figures_dir: Directory for the output figure.
        title: Matplotlib title for the plot.

    Returns:
        Path to the saved figure, or ``""`` if the CSV is missing.
    """
    if not os.path.exists(train_log_csv):
        print(f"[visualize] train_log.csv not found at {train_log_csv}; skipping")
        return ""
    train_pts, val_pts = [], []
    with open(train_log_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            t = row.get("train_loss", "")
            v = row.get("val_loss", "")
            if t and t != "":
                train_pts.append((step, float(t)))
            if v and v != "":
                val_pts.append((step, float(v)))
    fig, ax = plt.subplots(figsize=(10, 5))
    if train_pts:
        steps, losses = zip(*train_pts)
        ax.plot(steps, losses, label="train loss", color="#1f77b4", alpha=0.8)
    if val_pts:
        steps, losses = zip(*val_pts)
        ax.plot(steps, losses, label="val loss", color="#ff7f0e", alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE (normalized effective-map space)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(figures_dir, "training_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


def plot_toeplitz_comparison(results, figures_dir: str, sources, colors) -> str:
    """Bar chart of Toeplitz-ness by source, train vs val configs.

    Shared by the regressor eval modules; each caller supplies its own
    ``sources`` (e.g. ``["oracle", "target", "predicted"]``) and ``colors``
    dict (source name -> matplotlib color). Saves ``toeplitz_analysis.png``.

    Args:
        results: Sequence of per-config eval results (each with a ``toeplitz``
            dict and a ``split`` field).
        figures_dir: Directory for the output figure.
        sources: Ordered list of Toeplitz-ness sources to plot.
        colors: Dict mapping each source to its bar color.

    Returns:
        Path to the saved figure.
    """
    splits = ["train", "val"]
    data = {s: {sp: [] for sp in splits} for s in sources}
    for r in results:
        for s in sources:
            if s in r.toeplitz:
                data[s][r.split].append(r.toeplitz[s])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for col, sp in enumerate(splits):
        ax = axes[col]
        means = [np.mean(data[s][sp]) if data[s][sp] else 0.0 for s in sources]
        stds = [np.std(data[s][sp]) if data[s][sp] else 0.0 for s in sources]
        x = np.arange(len(sources))
        ax.bar(x, means, 0.6, yerr=stds, capsize=4,
               color=[colors[s] for s in sources], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=15)
        ax.set_title(f"Toeplitz-ness — {sp} configs")
        ax.set_ylabel("mean diagonal std (lower = more Toeplitz)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Toeplitz-ness: oracle ~0 = true convolution", fontsize=11)
    fig.tight_layout()
    path = os.path.join(figures_dir, "toeplitz_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path


def plot_functional_mse_boxplot(results, figures_dir: str, methods, labels,
                                colors, suptitle: str, tick_rotation: int = 15,
                                figsize=(12, 5)) -> str:
    """Box plot of functional test MSE by method, train vs val configs.

    Shared by the regressor eval modules; each caller supplies its own
    ``methods``, display ``labels``, ``colors``, and ``suptitle``. Methods with
    no data are drawn as an empty (NaN) box labelled "(N/A)". Saves
    ``mse_comparison.png``.

    Args:
        results: Sequence of per-config eval results (each with a ``functional``
            dict and a ``split`` field).
        figures_dir: Directory for the output figure.
        methods: Ordered list of functional-MSE method keys.
        labels: Display labels, parallel to ``methods``.
        colors: Bar colors, parallel to ``methods``.
        suptitle: Figure super-title.
        tick_rotation: X-tick label rotation (degrees).
        figsize: Figure size ``(width, height)``.

    Returns:
        Path to the saved figure.
    """
    splits = ["train", "val"]
    data = {m: {s: [] for s in splits} for m in methods}
    for r in results:
        for m in methods:
            v = r.functional.get(m)
            if v is not None:
                data[m][r.split].append(v)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for col, sp in enumerate(splits):
        ax = axes[col]
        box_data = []
        present = []
        for m in methods:
            if data[m][sp]:
                box_data.append(data[m][sp])
                present.append(labels[methods.index(m)])
            else:
                box_data.append([float("nan")])
                present.append(labels[methods.index(m)] + " (N/A)")
        bp = ax.boxplot(box_data, labels=present, showmeans=True,
                        patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(f"Functional test MSE — {sp} configs")
        ax.set_ylabel("Test MSE" if col == 0 else "")
        ax.tick_params(axis="x", rotation=tick_rotation)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    path = os.path.join(figures_dir, "mse_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {path}")
    return path
