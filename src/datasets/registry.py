# src/datasets/registry.py

from typing import Dict, Any, Type, Optional

from src.datasets.swe_bench import SWEBenchDataset
from src.datasets.base_dataset import BaseDataset
from src.datasets.json_dataset import JSONDataset
from src.datasets.csv_dataset import CSVDataset
from src.datasets.coding_problems import CodingProblemsDataset
from src.utils.logging import get_logger
from src.config.settings import load_config

# Initialize logger
logger = get_logger("dataset_registry")

# Registry of available datasets
DATASET_REGISTRY = {
    "json": JSONDataset,
    "csv": CSVDataset,
    "coding_problems": CodingProblemsDataset,
    "swe_bench": SWEBenchDataset
    # Add more dataset types as they are implemented
}

# Cache for instantiated datasets
_dataset_instances = {}


def register_dataset(dataset_type: str, dataset_class: Type[BaseDataset]) -> None:
    """
    Register a new dataset type.

    Args:
        dataset_type: Type identifier for the dataset
        dataset_class: Dataset class to register
    """
    if dataset_type in DATASET_REGISTRY:
        logger.warning(f"Overwriting existing dataset type: {dataset_type}")

    DATASET_REGISTRY[dataset_type] = dataset_class
    logger.info(f"Registered dataset type: {dataset_type}")


def get_dataset_class(dataset_type: str) -> Optional[Type[BaseDataset]]:
    """
    Get a dataset class by type.

    Args:
        dataset_type: Type of the dataset

    Returns:
        Dataset class or None if not found
    """
    if dataset_type not in DATASET_REGISTRY:
        logger.error(f"Dataset type not found: {dataset_type}")
        return None

    return DATASET_REGISTRY[dataset_type]


def get_dataset(dataset_id: str) -> BaseDataset:
    """
    Get a dataset instance by ID.

    Args:
        dataset_id: ID of the dataset

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset type is not registered
    """
    # Check if dataset is already instantiated
    if dataset_id in _dataset_instances:
        logger.debug(f"Using cached dataset instance: {dataset_id}")
        return _dataset_instances[dataset_id]

    # Load dataset configuration
    try:
        config_path = f"configs/datasets/{dataset_id}.yaml"
        dataset_config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Dataset configuration not found: {config_path}")
        raise ValueError(f"Dataset configuration not found: {dataset_id}")

    dataset_type = dataset_config.get("type")

    if dataset_type not in DATASET_REGISTRY:
        logger.error(f"Dataset type not found: {dataset_type}")
        raise ValueError(f"Dataset type not registered: {dataset_type}")

    # Instantiate dataset
    dataset_class = DATASET_REGISTRY[dataset_type]

    logger.info(f"Creating dataset of type '{dataset_type}' with ID '{dataset_id}'")
    dataset = dataset_class(dataset_config.get("config", {}))

    # Cache dataset instance
    _dataset_instances[dataset_id] = dataset

    return dataset


def list_available_datasets() -> Dict[str, str]:
    """
    List all available dataset types with their descriptions.

    Returns:
        Dictionary mapping dataset types to their descriptions
    """
    # Get descriptions from docstrings
    descriptions = {}
    for dataset_type, dataset_class in DATASET_REGISTRY.items():
        doc = dataset_class.__doc__ or ""
        # Use the first line of the docstring as the description
        description = doc.split("\n")[0].strip()
        descriptions[dataset_type] = description

    return descriptions


def clear_dataset_cache() -> None:
    """
    Clear the dataset instance cache.
    This is useful when dataset configurations change.
    """
    global _dataset_instances

    # Get a list of dataset IDs to avoid modification during iteration
    dataset_ids = list(_dataset_instances.keys())

    for dataset_id in dataset_ids:
        logger.info(f"Removing dataset from cache: {dataset_id}")

    # Clear the cache
    _dataset_instances = {}
    logger.info("Dataset cache cleared")


def get_dataset_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for all available datasets.

    Returns:
        Dictionary mapping dataset IDs to their configurations
    """
    import os
    import glob

    configs_dir = "configs/datasets"
    if not os.path.exists(configs_dir):
        logger.warning(f"Datasets configuration directory not found: {configs_dir}")
        return {}

    configs = {}
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))

    for config_file in config_files:
        try:
            dataset_id = os.path.splitext(os.path.basename(config_file))[0]
            config = load_config(config_file)
            configs[dataset_id] = config
        except Exception as e:
            logger.error(f"Error loading dataset configuration from {config_file}: {str(e)}")

    return configs
