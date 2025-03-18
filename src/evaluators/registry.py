# src/evaluators/registry.py

from typing import Dict, Any, Type, Optional
from src.evaluators.base_evaluator import BaseEvaluator
from src.evaluators.python_executor import PythonExecutor
from src.evaluators.unit_tester import UnitTester
from src.utils.logging import get_logger
from src.config.settings import load_config

# Initialize logger
logger = get_logger("evaluator_registry")

# Registry of available evaluators
EVALUATOR_REGISTRY = {
    "python_executor": PythonExecutor,
    "unit_tester": UnitTester
    # Add more evaluator types as they are implemented
}

# Cache for instantiated evaluators
_evaluator_instances = {}


def register_evaluator(evaluator_type: str, evaluator_class: Type[BaseEvaluator]) -> None:
    """
    Register a new evaluator type.

    Args:
        evaluator_type: Type identifier for the evaluator
        evaluator_class: Evaluator class to register
    """
    if evaluator_type in EVALUATOR_REGISTRY:
        logger.warning(f"Overwriting existing evaluator type: {evaluator_type}")

    EVALUATOR_REGISTRY[evaluator_type] = evaluator_class
    logger.info(f"Registered evaluator type: {evaluator_type}")


def get_evaluator_class(evaluator_type: str) -> Optional[Type[BaseEvaluator]]:
    """
    Get an evaluator class by type.

    Args:
        evaluator_type: Type of the evaluator

    Returns:
        Evaluator class or None if not found
    """
    if evaluator_type not in EVALUATOR_REGISTRY:
        logger.error(f"Evaluator type not found: {evaluator_type}")
        return None

    return EVALUATOR_REGISTRY[evaluator_type]


def get_evaluator(evaluator_id: str) -> BaseEvaluator:
    """
    Get an evaluator instance by ID.

    Args:
        evaluator_id: ID of the evaluator

    Returns:
        Evaluator instance

    Raises:
        ValueError: If evaluator type is not registered
    """
    # Check if evaluator is already instantiated
    if evaluator_id in _evaluator_instances:
        logger.debug(f"Using cached evaluator instance: {evaluator_id}")
        return _evaluator_instances[evaluator_id]

    # Load evaluator configuration
    try:
        config_path = f"configs/evaluators/{evaluator_id}.yaml"
        evaluator_config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Evaluator configuration not found: {config_path}")
        raise ValueError(f"Evaluator configuration not found: {evaluator_id}")

    evaluator_type = evaluator_config.get("type")

    if evaluator_type not in EVALUATOR_REGISTRY:
        logger.error(f"Evaluator type not found: {evaluator_type}")
        raise ValueError(f"Evaluator type not registered: {evaluator_type}")

    # Instantiate evaluator
    evaluator_class = EVALUATOR_REGISTRY[evaluator_type]

    logger.info(f"Creating evaluator of type '{evaluator_type}' with ID '{evaluator_id}'")
    evaluator = evaluator_class(evaluator_config.get("config", {}))

    # Cache evaluator instance
    _evaluator_instances[evaluator_id] = evaluator

    return evaluator


def list_available_evaluators() -> Dict[str, str]:
    """
    List all available evaluator types with their descriptions.

    Returns:
        Dictionary mapping evaluator types to their descriptions
    """
    # Get descriptions from docstrings
    descriptions = {}
    for evaluator_type, evaluator_class in EVALUATOR_REGISTRY.items():
        doc = evaluator_class.__doc__ or ""
        # Use the first line of the docstring as the description
        description = doc.split("\n")[0].strip()
        descriptions[evaluator_type] = description

    return descriptions


def clear_evaluator_cache() -> None:
    """
    Clear the evaluator instance cache.
    This is useful for freeing up resources or when configurations change.
    """
    global _evaluator_instances

    # Get a list of evaluator IDs to avoid modification during iteration
    evaluator_ids = list(_evaluator_instances.keys())

    for evaluator_id in evaluator_ids:
        logger.info(f"Removing evaluator from cache: {evaluator_id}")

    # Clear the cache
    _evaluator_instances = {}
    logger.info("Evaluator cache cleared")


def get_evaluator_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for all available evaluators.

    Returns:
        Dictionary mapping evaluator IDs to their configurations
    """
    import os
    import glob

    configs_dir = "configs/evaluators"
    if not os.path.exists(configs_dir):
        logger.warning(f"Evaluators configuration directory not found: {configs_dir}")
        return {}

    configs = {}
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))

    for config_file in config_files:
        try:
            evaluator_id = os.path.splitext(os.path.basename(config_file))[0]
            config = load_config(config_file)
            configs[evaluator_id] = config
        except Exception as e:
            logger.error(f"Error loading evaluator configuration from {config_file}: {str(e)}")

    return configs