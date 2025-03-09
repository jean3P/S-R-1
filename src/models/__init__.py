# src/models/__init__.py
"""
Language models for the AI system.

This package contains different types of language models that can be used
for text generation. The main model types are:

- HuggingFaceModel: Local models from HuggingFace Transformers
- OpenAIModel: API-based models from OpenAI
- AnthropicModel: API-based models from Anthropic

New model types can be added by implementing the BaseModel interface and
registering them with the model registry.
"""

from src.models.base_model import BaseModel
from src.models.huggingface_model import HuggingFaceModel
from src.models.openai_model import OpenAIModel
from src.models.anthropic_model import AnthropicModel
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
    'OpenAIModel',
    'AnthropicModel',
    'register_model',
    'get_model',
    'get_model_class',
    'list_available_models',
    'clear_model_cache',
    'get_model_configs'
]