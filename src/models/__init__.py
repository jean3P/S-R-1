# src/models/__init__.py

from src.models.base_model import BaseModel
from src.models.huggingface_model import HuggingFaceModel
from src.models.registry import (
    register_model,
    get_model,
    get_model_class,
    list_available_models,
    clear_model_cache,
    get_model_configs
)

# Version of the models package
__version__ = "0.1.0"

__all__ = [
    'BaseModel',
    'HuggingFaceModel',
    'register_model',
    'get_model',
    'get_model_class',
    'list_available_models',
    'clear_model_cache',
    'get_model_configs'
]