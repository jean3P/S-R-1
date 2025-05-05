"""
Module for model loading and initialization.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# Import model classes lazily to avoid unnecessary imports
def create_model(model_name: str, config) -> Any:
    """
    Create and return a model instance.

    Args:
        model_name: Name of the model to use
        config: Configuration object

    Returns:
        Model instance
    """
    logger.info(f"Creating model: {model_name}")

    # Define supported models
    supported_models = {
        "deepseek-r1-distill",
        "qwen2-5-coder",
        "qwq-preview"
    }

    # Validate model name
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model: {model_name}. Supported models are: {supported_models}")

    # Lazy import the appropriate model class
    if model_name == "deepseek-r1-distill":
        try:
            from .deepseek_r1_model import DeepseekR1Model
            return DeepseekR1Model(config)
        except ImportError as e:
            logger.error(f"Failed to import DeepseekR1Model: {e}")
            raise

    elif model_name == "qwen2-5-coder":
        try:
            from .qwen25_coder_model import Qwen25CoderModel
            return Qwen25CoderModel(config)
        except ImportError as e:
            logger.error(f"Failed to import Qwen25CoderModel: {e}")
            raise

    elif model_name == "qwq-preview":
        try:
            from .qwq_preview_model import QwqPreviewModel
            return QwqPreviewModel(config)
        except ImportError as e:
            logger.error(f"Failed to import QwqPreviewModel: {e}")
            raise

    # Fallback to BaseModel if something goes wrong with the specific model
    from .base_model import BaseModel
    logger.warning(f"Using BaseModel as fallback for {model_name}")
    return BaseModel(model_name, config)


# Convenience function to list available models
def list_available_models():
    return ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]
