# src/utils/tokenizer_config.py

"""
Tokenizer configuration module for registering model-specific tokenizers.

This module dynamically registers tokenizers for models specified in configuration,
ensuring proper token counting and processing throughout the application.
"""

import os
from typing import Dict, List, Optional, Callable, Any
import traceback
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.utils.tokenization import register_tokenizer
from src.utils.logging import get_logger
from src.config.settings import load_config, get_config_value

# Initialize logger
logger = get_logger("tokenizer_config")

# Model configuration mapping
MODEL_CONFIG = {
    # Map model IDs to their HuggingFace paths and aliases
    "qwen_coder": {
        "path": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "aliases": ["qwen", "qwen2.5", "qwen2.5-coder"],
    },
    "qwq_preview": {
        "path": "Qwen/QwQ-32B-Preview",
        "aliases": ["qwq", "qwq32b"],
    },
    "deepseek_qwen": {
        "path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "aliases": ["deepseek", "deepseek-r1"],
    },
    "llama3": {
        "path": "meta-llama/Llama-3-70b-instruct",
        "aliases": ["llama", "llama3"]
    },
    "mistral": {
        "path": "mistralai/Mistral-7B-Instruct-v0.2",
        "aliases": ["mistral7b"]
    },
    # Add more models as needed
}


def get_model_info() -> Dict[str, Dict[str, Any]]:
    """
    Get model information from configuration or environment.

    Prioritizes:
    1. Environment variable MODEL_ID
    2. Config files
    3. Default mapping

    Returns:
        Dictionary of model configurations
    """
    # Check for model override in environment
    model_id = os.environ.get("MODEL_ID")

    if model_id:
        logger.info(f"Using model ID from environment: {model_id}")
        # Look up the model in our mapping
        if model_id in MODEL_CONFIG:
            return {model_id: MODEL_CONFIG[model_id]}
        else:
            # Assume the environment variable contains the direct model path
            logger.info(f"Model ID not in known configs, using as direct path")
            return {model_id: {"path": model_id, "aliases": []}}

    # Try to load from config files
    try:
        # Check if we have any model configs loaded
        models_config = {}

        # Look through model config files
        config_dir = "configs/models"
        if os.path.exists(config_dir):
            import glob
            for config_file in glob.glob(f"{config_dir}/*.yaml"):
                try:
                    config = load_config(config_file)
                    model_id = config.get("id")
                    if model_id:
                        model_name = get_config_value(config, "config.model_name")
                        if model_name:
                            models_config[model_id] = {
                                "path": model_name,
                                "aliases": [model_id.lower()]
                            }
                except Exception as e:
                    logger.warning(f"Error loading model config from {config_file}: {e}")

        if models_config:
            logger.info(f"Loaded {len(models_config)} models from config files")
            return models_config
    except Exception as e:
        logger.warning(f"Error loading model configs: {e}")

    # Fall back to the default mapping
    logger.info("Using default model configuration mapping")
    return MODEL_CONFIG


def register_model_tokenizer(model_id: str, model_config: Dict[str, Any]) -> bool:
    """
    Register tokenizer for a specific model.

    Args:
        model_id: Identifier for the model
        model_config: Configuration for the model including path and aliases

    Returns:
        bool: True if tokenizer registration was successful, False otherwise.
    """
    model_path = model_config["path"]
    aliases = model_config.get("aliases", [])

    try:
        logger.info(f"Loading tokenizer for {model_id} from {model_path}")
        # Try to load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if not tokenizer:
            logger.error(f"Failed to load tokenizer for {model_id}: returned None")
            return False

        logger.debug(f"Tokenizer loaded successfully. Vocabulary size: {len(tokenizer)}")

        def tokenize_fn(text: str) -> List[int]:
            """Tokenize text using the loaded tokenizer."""
            return tokenizer.encode(text)

        # Register for the model ID
        logger.info(f"Registering tokenizer for '{model_id}'")
        register_tokenizer(model_id, tokenize_fn)

        # Register for the full model path
        logger.info(f"Registering tokenizer for '{model_path}'")
        register_tokenizer(model_path, tokenize_fn)

        # Register for aliases
        for alias in aliases:
            logger.debug(f"Registering tokenizer for alias '{alias}'")
            register_tokenizer(alias, tokenize_fn)

        logger.info(f"✓ Successfully registered tokenizer for {model_id}")
        return True

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure transformers is installed: pip install transformers")
        return False

    except Exception as e:
        logger.warning(f"Failed to register tokenizer for {model_id}: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return False


def setup_tokenizers() -> bool:
    """
    Set up all required tokenizers based on configuration.

    Returns:
        bool: True if all required tokenizers were registered successfully.
    """
    success = True
    model_configs = get_model_info()

    if not model_configs:
        logger.warning("No model configurations found")
        return False

    logger.info(f"Setting up tokenizers for {len(model_configs)} models")

    for model_id, config in model_configs.items():
        if not register_model_tokenizer(model_id, config):
            success = False
            logger.warning(f"Failed to register tokenizer for {model_id}")

    return success


# Register tokenizers when this module is imported
setup_result = setup_tokenizers()

if __name__ == "__main__":
    # When run directly, report status and perform a basic test
    if setup_result:
        print("✓ Tokenizer setup completed successfully")
    else:
        print("⚠ Tokenizer setup completed with warnings or errors (see log)")

    # Run a simple test if module is executed directly
    try:
        from src.utils.tokenization import count_tokens

        # Get first model from the configuration
        test_models = list(get_model_info().keys())
        if test_models:
            test_model = test_models[0]
            test_text = "Hello, this is a test of the tokenizer configuration"
            token_count = count_tokens(test_text, test_model)
            print(f"Test tokenization with {test_model} - '{test_text}' => {token_count} tokens")
    except Exception as e:
        print(f"Test tokenization failed: {e}")

