"""1D data generation module.

See ``plans/plan.md`` Section 1 for the full specification.
"""
from src.data.dataset import (
    DatasetInstance,
    DatasetFamily,
    generate_dataset,
    generate_corpus,
)
from src.data.registry import dataset_id_from_config

__all__ = [
    "DatasetInstance",
    "DatasetFamily",
    "generate_dataset",
    "generate_corpus",
    "dataset_id_from_config",
]
