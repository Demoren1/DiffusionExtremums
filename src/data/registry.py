"""Deterministic dataset-ID hashing from a config."""
import hashlib
import json
from typing import Dict

from src.configs.base import DatasetConfig


def _canonical_config_dict(config: DatasetConfig) -> Dict:
    """Kernel floats rounded to 6 decimals so the hash is stable across runs."""
    return {
        "family": str(config.family),
        "kernel": [round(float(v), 6) for v in config.kernel],
        "radius": int(config.radius),
        "noise_std": round(float(config.noise_std), 6),
        "n_train": int(config.n_train),
        "n_test": int(config.n_test),
        "seed": int(config.seed),
        "L": int(config.L),
    }


def dataset_id_from_config(config: DatasetConfig) -> str:
    payload = json.dumps(_canonical_config_dict(config), sort_keys=True,
                         separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:16]
