"""Shared helpers for the evaluation modules."""
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.configs.base import DatasetConfig


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.nn.functional.mse_loss(pred, target).item())


def config_dict_to_datasetconfig(cfg: Dict) -> DatasetConfig:
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
    """Sample (config_idx, split) pairs to evaluate, train configs first."""
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
    """Bucket per-config metric values into per-split lists."""
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
    """Summary {mean, std, n} over values, skipping None entries."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.array(vals)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}


def write_summary_csv(summary: Dict, path: str,
                      extra_rows: Optional[List[List]] = None) -> None:
    """Write the standard summary.csv layout shared by the eval modules."""
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
