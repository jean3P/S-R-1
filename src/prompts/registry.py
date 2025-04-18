# src/prompts/registry.py

from typing import Dict, Any, Type, Optional
from src.prompts.base_prompt import BasePrompt
from src.utils.logging import get_logger
from src.config.settings import load_config

# Initialize logger
logger = get_logger("prompt_registry")

# Registry of available prompts
PROMPT_REGISTRY = {
    # Add more prompt types as they are implemented
}

# Cache for instantiated prompts
_prompt_instances = {}


def register_prompt(prompt_type: str, prompt_class: Type[BasePrompt]) -> None:
    """
    Register a new prompt type.

    Args:
        prompt_type: Type identifier for the prompt
        prompt_class: Prompt class to register
    """
    if prompt_type in PROMPT_REGISTRY:
        logger.warning(f"Overwriting existing prompt type: {prompt_type}")

    PROMPT_REGISTRY[prompt_type] = prompt_class
    logger.info(f"Registered prompt type: {prompt_type}")


def get_prompt_class(prompt_type: str) -> Optional[Type[BasePrompt]]:
    """
    Get a prompt class by type.

    Args:
        prompt_type: Type of the prompt

    Returns:
        Prompt class or None if not found
    """
    if prompt_type not in PROMPT_REGISTRY:
        logger.error(f"Prompt type not found: {prompt_type}")
        return None

    return PROMPT_REGISTRY[prompt_type]


def get_prompt(prompt_id: str) -> BasePrompt:
    """
    Get a prompt instance by ID.

    Args:
        prompt_id: ID of the prompt

    Returns:
        Prompt instance

    Raises:
        ValueError: If prompt type is not registered
    """
    # Check if prompt is already instantiated
    if prompt_id in _prompt_instances:
        logger.debug(f"Using cached prompt instance: {prompt_id}")
        return _prompt_instances[prompt_id]

    # Load prompt configuration
    try:
        config_path = f"configs/prompts/{prompt_id}.yaml"
        prompt_config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Prompt configuration not found: {config_path}")
        raise ValueError(f"Prompt configuration not found: {prompt_id}")

    prompt_type = prompt_config.get("type")

    if prompt_type not in PROMPT_REGISTRY:
        logger.error(f"Prompt type not found: {prompt_type}")
        raise ValueError(f"Prompt type not registered: {prompt_type}")

    # Instantiate prompt
    prompt_class = PROMPT_REGISTRY[prompt_type]

    logger.info(f"Creating prompt of type '{prompt_type}' with ID '{prompt_id}'")
    prompt = prompt_class(prompt_config.get("config", {}))

    # Cache prompt instance
    _prompt_instances[prompt_id] = prompt

    return prompt


def list_available_prompts() -> Dict[str, str]:
    """
    List all available prompt types with their descriptions.

    Returns:
        Dictionary mapping prompt types to their descriptions
    """
    # Get descriptions from docstrings
    descriptions = {}
    for prompt_type, prompt_class in PROMPT_REGISTRY.items():
        doc = prompt_class.__doc__ or ""
        # Use the first line of the docstring as the description
        description = doc.split("\n")[0].strip()
        descriptions[prompt_type] = description

    return descriptions


def clear_prompt_cache() -> None:
    """
    Clear the prompt instance cache.
    This is useful when prompt configurations change.
    """
    global _prompt_instances

    # Get a list of prompt IDs to avoid modification during iteration
    prompt_ids = list(_prompt_instances.keys())

    for prompt_id in prompt_ids:
        logger.info(f"Removing prompt from cache: {prompt_id}")

    # Clear the cache
    _prompt_instances = {}
    logger.info("Prompt cache cleared")


def get_prompt_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for all available prompts.

    Returns:
        Dictionary mapping prompt IDs to their configurations
    """
    import os
    import glob

    configs_dir = "configs/prompts"
    if not os.path.exists(configs_dir):
        logger.warning(f"Prompts configuration directory not found: {configs_dir}")
        return {}

    configs = {}
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))

    for config_file in config_files:
        try:
            prompt_id = os.path.splitext(os.path.basename(config_file))[0]
            config = load_config(config_file)
            configs[prompt_id] = config
        except Exception as e:
            logger.error(f"Error loading prompt configuration from {config_file}: {str(e)}")

    return configs
