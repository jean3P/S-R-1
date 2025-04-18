# src/models/registry.py

from typing import Dict, Any, Type, Optional
from src.models.base_model import BaseModel
from src.models.huggingface_model import HuggingFaceModel
from src.utils.logging import get_logger
from src.config.settings import load_config

# Initialize logger
logger = get_logger("model_registry")

# Registry of available models
MODEL_REGISTRY = {
    "huggingface": HuggingFaceModel,
}

# Cache for instantiated models
_model_instances = {}


def register_model(model_type: str, model_class: Type[BaseModel]) -> None:
    """
    Register a new model type.

    Args:
        model_type: Type identifier for the model
        model_class: Model class to register
    """
    if model_type in MODEL_REGISTRY:
        logger.warning(f"Overwriting existing model type: {model_type}")

    MODEL_REGISTRY[model_type] = model_class
    logger.info(f"Registered model type: {model_type}")


def get_model_class(model_type: str) -> Optional[Type[BaseModel]]:
    """
    Get a model class by type.

    Args:
        model_type: Type of the model

    Returns:
        Model class or None if not found
    """
    if model_type not in MODEL_REGISTRY:
        logger.error(f"Model type not found: {model_type}")
        return None

    return MODEL_REGISTRY[model_type]


def get_model(model_id: str) -> BaseModel:
    """
    Get a model instance by ID.

    Args:
        model_id: ID of the model

    Returns:
        Model instance

    Raises:
        ValueError: If model type is not registered
    """
    # Check if model is already instantiated
    if model_id in _model_instances:
        logger.debug(f"Using cached model instance: {model_id}")
        return _model_instances[model_id]

    # Load model configuration
    try:
        config_path = f"configs/models/{model_id}.yaml"
        model_config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Model configuration not found: {config_path}")
        raise ValueError(f"Model configuration not found: {model_id}")

    model_type = model_config.get("type")

    if model_type not in MODEL_REGISTRY:
        logger.error(f"Model type not found: {model_type}")
        raise ValueError(f"Model type not registered: {model_type}")

    # Instantiate model
    model_class = MODEL_REGISTRY[model_type]

    logger.info(f"Creating model of type '{model_type}' with ID '{model_id}'")
    model = model_class(model_config.get("config", {}))

    # Cache model instance
    _model_instances[model_id] = model

    return model


def list_available_models() -> Dict[str, str]:
    """
    List all available model types with their descriptions.

    Returns:
        Dictionary mapping model types to their descriptions
    """
    # Get descriptions from docstrings
    descriptions = {}
    for model_type, model_class in MODEL_REGISTRY.items():
        doc = model_class.__doc__ or ""
        # Use the first line of the docstring as the description
        description = doc.split("\n")[0].strip()
        descriptions[model_type] = description

    return descriptions


def clear_model_cache() -> None:
    """
    Clear the model instance cache.
    This is useful for freeing up memory or when configurations change.
    """
    global _model_instances

    # Get a list of model IDs to avoid modification during iteration
    model_ids = list(_model_instances.keys())

    for model_id in model_ids:
        logger.info(f"Removing model from cache: {model_id}")
        # Call any cleanup methods if available
        model = _model_instances[model_id]
        if hasattr(model, "clear_cuda_cache"):
            model.clear_cuda_cache()

    # Clear the cache
    _model_instances = {}
    logger.info("Model cache cleared")


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for all available models.

    Returns:
        Dictionary mapping model IDs to their configurations
    """
    import os
    import glob

    configs_dir = "configs/models"
    if not os.path.exists(configs_dir):
        logger.warning(f"Models configuration directory not found: {configs_dir}")
        return {}

    configs = {}
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))

    for config_file in config_files:
        try:
            model_id = os.path.splitext(os.path.basename(config_file))[0]
            config = load_config(config_file)
            configs[model_id] = config
        except Exception as e:
            logger.error(f"Error loading model configuration from {config_file}: {str(e)}")

    return configs
