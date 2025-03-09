# src/config/settings.py

import os
import yaml
import json
from typing import Dict, Any
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger("settings")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the file format is not supported
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Determine file type from extension
    _, ext = os.path.splitext(config_path)

    try:
        with open(config_path, "r") as f:
            if ext.lower() in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif ext.lower() == ".json":
                config = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {ext}")
                raise ValueError(f"Unsupported configuration file format: {ext}")

        # Process environment variable references
        config = _process_env_vars(config)

        logger.debug(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML or JSON file.

    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file

    Raises:
        ValueError: If the file format is not supported
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Determine file type from extension
    _, ext = os.path.splitext(config_path)

    try:
        with open(config_path, "w") as f:
            if ext.lower() in [".yaml", ".yml"]:
                yaml.dump(config, f, default_flow_style=False)
            elif ext.lower() == ".json":
                json.dump(config, f, indent=2)
            else:
                logger.error(f"Unsupported configuration file format: {ext}")
                raise ValueError(f"Unsupported configuration file format: {ext}")

        logger.debug(f"Saved configuration to {config_path}")

    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        raise


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from a nested configuration dictionary using a dot-separated key path.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key
        default: Default value to return if the key does not exist

    Returns:
        Value at the specified key path, or the default value if not found

    Example:
        get_config_value(config, "model.parameters.temperature", 0.7)
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    result = base_config.copy()

    def _merge_dict(target, source):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                target[key] = _merge_dict(target[key], value)
            else:
                target[key] = value
        return target

    return _merge_dict(result, override_config)


def load_default_config() -> Dict[str, Any]:
    """
    Load the default system configuration.

    Returns:
        Default configuration dictionary
    """
    default_config_path = os.path.join(os.path.dirname(__file__), "default_config.yaml")

    try:
        return load_config(default_config_path)
    except (FileNotFoundError, ValueError):
        logger.warning("Default configuration not found, using empty configuration")
        return {}


def _process_env_vars(config: Any) -> Any:
    """
    Process environment variable references in the configuration.

    Args:
        config: Configuration value (can be dict, list, or primitive)

    Returns:
        Configuration with environment variables replaced
    """
    if isinstance(config, dict):
        return {key: _process_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_process_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        # Extract environment variable name
        env_var = config[2:-1]

        # Get value from environment
        value = os.environ.get(env_var)

        if value is None:
            logger.warning(f"Environment variable not found: {env_var}")
            return config

        return value
    else:
        return config
