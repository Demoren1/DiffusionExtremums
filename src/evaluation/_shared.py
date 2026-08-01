"""Shared helpers for the evaluation modules.

Small, dependency-light utilities used by :mod:`src.evaluation.evaluate`,
:mod:`src.evaluation.regressor_eval`, and
:mod:`src.evaluation.oracle_regressor_eval` so that the three modules do not
each re-implement the same MSE / config-conversion / split-sampling /
aggregation / summary-CSV boilerplate.

These helpers are internal to the package and are not part of the public
evaluation API.
"""
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.configs.base import DatasetConfig


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error between two tensors (scalar float)."""
    return float(torch.nn.functional.mse_loss(pred, target).item())


def config_dict_to_datasetconfig(cfg: Dict) -> DatasetConfig:
    """Convert a config dict (from ``configs.json``) to a ``DatasetConfig``."""
    return DatasetConfig(
        family=str(cfg["family"]),
        kernel=tuple(float(v) for v in cfg["kernel"]),
        radius=int(cfg["radius"]),
        noise_std=float(cfg["noise_std"]),
        n_train=int(cfg["n_train"]),
        n_test=int(cfg.get("n_test", 512)),
        seed=int(cfg.get("seed", 0)),
        L=int(cfg.get("L", 32)),
    )


def sample_eval_config_indices(
    train_config_indices: List[int],
    val_config_indices: List[int],
    n_eval_train: int,
    n_eval_val: int,
    seed: int,
) -> List[Tuple[int, str]]:
    """Sample the ``(config_idx, split)`` pairs to evaluate, deterministically.

    Reproduces the exact sampling order used across the eval modules: one NumPy
    ``Generator`` seeded from ``seed`` first samples the train configs, then the
    val configs (both without replacement).

    Returns:
        List of ``(idx, split)`` tuples, train configs first then val configs.
    """
    rng = np.random.default_rng(seed)
    train_sample = list(rng.choice(
        train_config_indices,
        size=min(n_eval_train, len(train_config_indices)),
        replace=False))
    val_sample = list(rng.choice(
        val_config_indices,
        size=min(n_eval_val, len(val_config_indices)),
        replace=False))
    return ([(int(i), "train") for i in train_sample]
            + [(int(i), "val") for i in val_sample])


def collect_eval_lists(results, toeplitz_sources: List[str],
                       functional_methods: List[str]) -> Dict[str, Any]:
    """Collect per-config metric values into per-split lists.

    Shared by the regressor aggregation functions: walks the per-config eval
    records and buckets ``norm_mse`` / ``raw_mse`` / ``toeplitz`` /
    ``kernel_recovery`` / ``functional`` values by split (and by source/method
    where applicable).

    Args:
        results: Sequence of per-config eval records, each exposing ``split``,
            ``norm_mse``, ``raw_mse``, ``toeplitz``, ``kernel_recovery``, and
            ``functional`` attributes.
        toeplitz_sources: Ordered list of Toeplitz-ness sources.
        functional_methods: Ordered list of functional-MSE method keys.

    Returns:
        Dict with keys ``norm_mse``/``raw_mse`` (split -> values),
        ``toeplitz`` (source -> split -> values), ``kernel_rec``
        (split -> metric -> values), and ``functional``
        (method -> split -> values).
    """
    splits = ["train", "val"]
    norm_mse = {s: [] for s in splits}
    raw_mse = {s: [] for s in splits}
    toeplitz = {src: {s: [] for s in splits} for src in toeplitz_sources}
    kernel_rec = {s: {"cosine_sim": [], "l2_dist": []} for s in splits}
    functional = {m: {s: [] for s in splits} for m in functional_methods}
    for r in results:
        s = r.split
        norm_mse[s].append(r.norm_mse)
        raw_mse[s].append(r.raw_mse)
        for src, v in r.toeplitz.items():
            toeplitz[src][s].append(v)
        for k, v in r.kernel_recovery.items():
            kernel_rec[s][k].append(v)
        for m, v in r.functional.items():
            functional[m][s].append(v)
    return {
        "norm_mse": norm_mse,
        "raw_mse": raw_mse,
        "toeplitz": toeplitz,
        "kernel_rec": kernel_rec,
        "functional": functional,
    }


def stats_of(values: List[Optional[float]]) -> Dict[str, Any]:
    """Summary statistics ``{mean, std, n}`` over a list of values.

    ``None`` entries are skipped (functional-MSE values may be absent). An
    empty (or all-None) list yields ``{"mean": None, "std": None, "n": 0}``.

    Args:
        values: List of scalar values or ``None``.

    Returns:
        Dict with ``mean``/``std`` as floats (or ``None`` when empty) and ``n``
        as the count of non-None entries.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(vals)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}


def write_summary_csv(summary: Dict, path: str,
                      extra_rows: Optional[List[List]] = None) -> None:
    """Write the standard ``summary.csv`` layout shared by the eval modules.

    Writes the ``norm_mse`` / ``raw_mse`` / ``toeplitzness`` /
    ``kernel_recovery`` / ``functional_mse`` sections (train/val splits
    interleaved, matching the original per-module writers), followed by any
    ``extra_rows`` (e.g. the oracle ``ratio_to_oracle_conv`` rows).

    Args:
        summary: Aggregated summary dict (see ``aggregate_regressor_results``).
        path: Output ``.csv`` path.
        extra_rows: Optional additional rows appended after the standard ones.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "name", "split", "mean", "std", "n"])
        for s in ("train", "val"):
            st = summary["norm_mse"][s]
            w.writerow(["norm_mse", s, s, st["mean"], st["std"], st["n"]])
            st = summary["raw_mse"][s]
            w.writerow(["raw_mse", s, s, st["mean"], st["std"], st["n"]])
        for src, sd in summary["toeplitzness"].items():
            for s, st in sd.items():
                w.writerow(["toeplitzness", src, s, st["mean"], st["std"], st["n"]])
        for s, sd in summary["kernel_recovery"].items():
            for k, st in sd.items():
                w.writerow(["kernel_recovery", k, s, st["mean"], st["std"], st["n"]])
        for m, sd in summary["functional_mse"].items():
            for s, st in sd.items():
                w.writerow(["functional_mse", m, s, st["mean"], st["std"], st["n"]])
        for row in extra_rows or []:
            w.writerow(row)
