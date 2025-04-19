# src/models/__init__.py

from typing import Dict, Any


def create_model(model_name: str, config: Dict[str, Any]):
    """
    Create an instance of a model based on the model name.

    Args:
        model_name: Name of the model to create.
        config: Configuration dictionary.

    Returns:
        An instance of the requested model.
    """
    # First check for required dependencies
    missing_deps = check_dependencies(model_name, config)
    if missing_deps:
        print(f"Warning: Missing dependencies for {model_name}: {', '.join(missing_deps)}")
        print(f"Install them with: pip install {' '.join(missing_deps)}")

    # Map specific model names to the correct class
    model_map = {
        # DeepSeek models
        "deepseek-r1-distill": "DeepseekR1Model",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepseekR1Model",

        # Qwen Coder models
        "qwen2-5-coder": "Qwen25CoderModel",
        "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen25CoderModel",

        # QwQ models
        "qwq-preview": "QwqPreviewModel",
        "Qwen/QwQ-32B-Preview": "QwqPreviewModel"
    }

    # Try to match the model name to a known model
    model_class_name = None

    # Check if we have an exact match
    if model_name in model_map:
        model_class_name = model_map[model_name]
    else:
        # Check for partial matches in the model name
        lowercase_name = model_name.lower()
        if "deepseek" in lowercase_name or "r1-distill" in lowercase_name:
            model_class_name = "DeepseekR1Model"
        elif "qwen2.5" in lowercase_name or "coder" in lowercase_name:
            model_class_name = "Qwen25CoderModel"
        elif "qwq" in lowercase_name or "preview" in lowercase_name:
            model_class_name = "QwqPreviewModel"

    # Import and instantiate the model class
    if model_class_name:
        try:
            # Import the corresponding module
            if model_class_name == "DeepseekR1Model":
                from .deepseek_r1_model import DeepseekR1Model
                return DeepseekR1Model(config)
            elif model_class_name == "Qwen25CoderModel":
                from .qwen25_coder_model import Qwen25CoderModel
                return Qwen25CoderModel(config)
            elif model_class_name == "QwqPreviewModel":
                from .qwq_preview_model import QwqPreviewModel
                return QwqPreviewModel(config)
        except ImportError as e:
            print(f"Warning: Could not import {model_class_name}. Using BaseModel instead. Error: {e}")

    # Fallback to base model with the given name
    print(f"Using BaseModel for {model_name}")
    from .base_model import BaseModel
    return BaseModel(model_name, config)


def check_dependencies(model_name: str, config: Dict[str, Any]):
    """Check if required dependencies for the model are installed."""
    missing = []

    # Check for bitsandbytes if quantization is enabled
    model_config = config.get_model_config(model_name)
    if "quantization" in model_config:
        try:
            import bitsandbytes
        except ImportError:
            missing.append("bitsandbytes==0.41.0")

    # Check for flash attention if enabled
    if model_config.get("use_flash_attention", False):
        try:
            import flash_attn
        except ImportError:
            missing.append("flash-attn")

    # Add other dependency checks as needed

    return missing
