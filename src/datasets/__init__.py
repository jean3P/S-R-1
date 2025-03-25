# src/datasets/__init__.py
"""
Datasets for the AI system.

This package contains different types of datasets that can be used
to store and manipulate data. The main dataset types are:

- JSONDataset: Dataset implementation for JSON files
- CSVDataset: Dataset implementation for CSV files
- CodingProblemsDataset: Specialized dataset for coding problems

New dataset types can be added by implementing the BaseDataset interface and
registering them with the dataset registry.
"""

from src.datasets.base_dataset import BaseDataset
from src.datasets.json_dataset import JSONDataset
from src.datasets.csv_dataset import CSVDataset
from src.datasets.coding_problems import CodingProblemsDataset
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
    'CSVDataset',
    'CodingProblemsDataset',
    'SWEBenchDataset',
    'register_dataset',
    'get_dataset',
    'get_dataset_class',
    'list_available_datasets',
    'clear_dataset_cache',
    'get_dataset_configs'
]