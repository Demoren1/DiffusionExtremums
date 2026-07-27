"""Dataset ID hashing and config registry.

The dataset ID is a deterministic SHA1-based 16-hex string derived from the
dataset's configuration, so the same config always maps to the same ID and
conditioning is reproducible. See ``plans/plan.md`` Section 1.4.

The kernel is included in the hash (rounded to a fixed precision) so that two
configs with different ground-truth kernels get different IDs, while the same
config is stable across runs.
"""
import hashlib
import json
from typing import Dict

from src.configs.base import DatasetConfig


def _canonical_config_dict(config: DatasetConfig) -> Dict:
    """Build a canonical, JSON-serializable dict from a ``DatasetConfig``.

    The kernel floats are rounded to 6 decimals so floating-point noise does
    not destabilize the hash across runs / platforms.
    """
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
    """Compute the deterministic 16-hex dataset ID for a config.

    ``dataset_id = SHA1(canonical_json(DatasetConfig))[:16]``.
    """
    payload = json.dumps(_canonical_config_dict(config), sort_keys=True,
                         separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:16]
