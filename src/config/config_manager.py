# src/config/config_manager.py

import os
import glob
from typing import Dict, Any
from src.config.settings import load_config, save_config, merge_configs, load_default_config
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger("config_manager")


class ConfigManager:
    """Manager for system configuration."""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Root directory for configuration files
        """
        self.config_dir = config_dir
        self.default_config = load_default_config()
        self.configs = {}
        self.experiment_config = None

    def load_experiment_config(self, experiment_id: str) -> Dict[str, Any]:
        """
        Load configuration for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Experiment configuration

        Raises:
            FileNotFoundError: If the experiment configuration file does not exist
        """
        experiment_path = os.path.join(self.config_dir, "experiments", f"{experiment_id}.yaml")
        experiment_config = load_config(experiment_path)

        # Store the current experiment configuration
        self.experiment_config = experiment_config

        # Load and merge component configurations
        full_config = self._load_component_configs(experiment_config)

        logger.info(f"Loaded experiment configuration for: {experiment_id}")
        return full_config

    def _load_component_configs(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and merge component configurations for an experiment.

        Args:
            experiment_config: Base experiment configuration

        Returns:
            Full configuration with all components
        """
        # Start with the default configuration
        full_config = self.default_config.copy()

        # Merge the experiment-level configuration
        full_config = merge_configs(full_config, experiment_config)

        # Load component configurations
        for component_type in ["agent", "model", "prompt", "evaluator"]:
            if component_type in experiment_config:
                component_id = experiment_config[component_type].get("id")
                if component_id:
                    try:
                        component_path = os.path.join(self.config_dir, f"{component_type}s", f"{component_id}.yaml")
                        component_config = load_config(component_path)

                        # Merge component configuration
                        full_config[component_type] = merge_configs(
                            full_config.get(component_type, {}),
                            component_config
                        )

                        # If component has specific overrides in experiment config, apply them
                        if "config" in experiment_config[component_type]:
                            if "config" not in full_config[component_type]:
                                full_config[component_type]["config"] = {}

                            full_config[component_type]["config"] = merge_configs(
                                full_config[component_type].get("config", {}),
                                experiment_config[component_type].get("config", {})
                            )

                        logger.debug(f"Loaded {component_type} configuration: {component_id}")
                    except FileNotFoundError:
                        logger.warning(f"{component_type.capitalize()} configuration not found: {component_id}")

        return full_config

    def get_component_configs(self, component_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get configurations for all components of a specific type.

        Args:
            component_type: Type of component (agent, model, prompt, evaluator)

        Returns:
            Dictionary mapping component IDs to their configurations
        """
        component_dir = os.path.join(self.config_dir, f"{component_type}s")
        if not os.path.exists(component_dir):
            logger.warning(f"Component directory not found: {component_dir}")
            return {}

        configs = {}
        config_files = glob.glob(os.path.join(component_dir, "*.yaml"))

        for config_file in config_files:
            try:
                component_id = os.path.splitext(os.path.basename(config_file))[0]
                config = load_config(config_file)
                configs[component_id] = config
            except Exception as e:
                logger.error(f"Error loading component configuration from {config_file}: {str(e)}")

        return configs

    def get_experiment_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configurations for all available experiments.

        Returns:
            Dictionary mapping experiment IDs to their configurations
        """
        experiment_dir = os.path.join(self.config_dir, "experiments")
        if not os.path.exists(experiment_dir):
            logger.warning(f"Experiments directory not found: {experiment_dir}")
            return {}

        configs = {}
        config_files = glob.glob(os.path.join(experiment_dir, "*.yaml"))

        for config_file in config_files:
            try:
                experiment_id = os.path.splitext(os.path.basename(config_file))[0]
                config = load_config(config_file)
                configs[experiment_id] = config
            except Exception as e:
                logger.error(f"Error loading experiment configuration from {config_file}: {str(e)}")

        return configs

    def save_experiment_config(self, experiment_id: str, config: Dict[str, Any]) -> str:
        """
        Save an experiment configuration.

        Args:
            experiment_id: ID of the experiment
            config: Experiment configuration

        Returns:
            Path to the saved configuration file
        """
        experiment_dir = os.path.join(self.config_dir, "experiments")
        os.makedirs(experiment_dir, exist_ok=True)

        config_path = os.path.join(experiment_dir, f"{experiment_id}.yaml")
        save_config(config, config_path)

        logger.info(f"Saved experiment configuration: {experiment_id}")
        return config_path

    def save_component_config(self, component_type: str, component_id: str, config: Dict[str, Any]) -> str:
        """
        Save a component configuration.

        Args:
            component_type: Type of component (agent, model, prompt, evaluator)
            component_id: ID of the component
            config: Component configuration

        Returns:
            Path to the saved configuration file
        """
        component_dir = os.path.join(self.config_dir, f"{component_type}s")
        os.makedirs(component_dir, exist_ok=True)

        config_path = os.path.join(component_dir, f"{component_id}.yaml")
        save_config(config, config_path)

        logger.info(f"Saved {component_type} configuration: {component_id}")
        return config_path

    def create_experiment_config(self, name: str, description: str,
                                 agent_id: str, model_id: str, prompt_id: str, evaluator_id: str,
                                 task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new experiment configuration.

        Args:
            name: Name of the experiment
            description: Description of the experiment
            agent_id: ID of the agent to use
            model_id: ID of the model to use
            prompt_id: ID of the prompt to use
            evaluator_id: ID of the evaluator to use
            task: Task details

        Returns:
            New experiment configuration
        """
        experiment_config = {
            "name": name,
            "description": description,
            "agent": {
                "id": agent_id
            },
            "model": {
                "id": model_id
            },
            "prompt": {
                "id": prompt_id
            },
            "evaluator": {
                "id": evaluator_id
            },
            "task": task
        }

        return experiment_config
