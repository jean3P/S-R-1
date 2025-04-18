# src/datasets/__init__.py

from src.datasets.base_dataset import BaseDataset
from src.datasets.json_dataset import JSONDataset
from src.datasets.swe_bench import SWEBenchDataset
from src.datasets.registry import (
    register_dataset,
    get_dataset,
    get_dataset_class,
    list_available_datasets,
    clear_dataset_cache,
    get_dataset_configs
)

# Version of the datasets package
__version__ = "0.1.0"

__all__ = [
    'BaseDataset',
    'JSONDataset',
    'SWEBenchDataset',
    'register_dataset',
    'get_dataset',
    'get_dataset_class',
    'list_available_datasets',
    'clear_dataset_cache',
    'get_dataset_configs'
]