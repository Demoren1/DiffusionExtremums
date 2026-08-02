"""Shared corpus loading utilities for the DatasetHypernet pipeline."""
import json, os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.weight_codec import WeightCodec


@dataclass
class CorpusBundle:
    weights: torch.Tensor
    losses: torch.Tensor
    configs: List[dict]
    dataset_ids: List[str]
    D: int
    n_configs: int
    n_mlp: int
    train_cfg_indices: List[int]
    val_cfg_indices: List[int]


def deterministic_split(n_configs: int, val_configs: int, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_configs)
    n_val = min(int(val_configs), max(0, n_configs - 1))
    val_idx = sorted(int(i) for i in perm[:n_val])
    train_idx = sorted(int(i) for i in perm[n_val:])
    return train_idx, val_idx


def load_relu_corpus(corpus_dir: str, val_configs: int = 50, seed: int = 0,
                     use_configs_subset: Optional[List[int]] = None) -> CorpusBundle:
    cdir = corpus_dir
    weights_path = os.path.join(cdir, "weights.pt")
    losses_path = os.path.join(cdir, "losses.pt")
    cfg_path = os.path.join(cdir, "configs.json")
    ids_path = os.path.join(cdir, "dataset_ids.json")
    for p in (weights_path, cfg_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing required file: {p}")
    weights = torch.load(weights_path, map_location="cpu").float()
    if weights.dim() != 3:
        raise ValueError(f"weights.pt must be [n_configs, n_mlp, D], got {tuple(weights.shape)}")
    n_configs, n_mlp, D = weights.shape
    with open(cfg_path, "r") as f:
        configs = json.load(f)
    dataset_ids: List[str] = []
    if os.path.exists(ids_path):
        with open(ids_path, "r") as f:
            dataset_ids = json.load(f)
    losses = None
    if os.path.exists(losses_path):
        losses = torch.load(losses_path, map_location="cpu").float()
    if use_configs_subset is not None:
        subset = sorted(int(i) for i in use_configs_subset)
        weights = weights[subset]
        if losses is not None:
            losses = losses[subset]
        configs = [configs[i] for i in subset]
        dataset_ids = [dataset_ids[i] for i in subset] if dataset_ids else []
        n_configs = len(subset)
    train_idx, val_idx = deterministic_split(n_configs, val_configs, seed)
    print(f"[load_relu_corpus] corpus {tuple(weights.shape)} ({len(train_idx)} train / {len(val_idx)} val configs)")
    return CorpusBundle(weights=weights, losses=losses, configs=configs, dataset_ids=dataset_ids,
                        D=D, n_configs=n_configs, n_mlp=n_mlp,
                        train_cfg_indices=train_idx, val_cfg_indices=val_idx)


def config_from_record(rec: dict):
    from src.configs.base import DatasetConfig
    return DatasetConfig(
        family=rec["family"], kernel=tuple(float(v) for v in rec["kernel"]),
        radius=int(rec["radius"]), noise_std=float(rec.get("noise_std", 0.0)),
        n_train=int(rec.get("n_train", 1024)), n_test=int(rec.get("n_test", 512)),
        seed=int(rec.get("seed", 0)), L=int(rec.get("L", 32)),
    )
