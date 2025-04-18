# src/config/__init__.py
from src.config.settings import (
    load_config,
    save_config,
    get_config_value,
    merge_configs,
    load_default_config
)

from src.config.config_manager import ConfigManager
from src.config.validation import validate_config, validate_experiment_config

# Version of the config package
__version__ = "0.1.0"

__all__ = [
    'load_config',
    'save_config',
    'get_config_value',
    'merge_configs',
    'load_default_config',
    'ConfigManager',
    'validate_config',
    'validate_experiment_config'
]
