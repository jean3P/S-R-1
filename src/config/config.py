# config/config.py
import os
import yaml
from pathlib import Path


class Config:
    def __init__(self, config_path=None):
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Default configuration
        self.defaults = {
            "data": {
                "swe_bench_path": str(self.base_dir / "data" / "swe-bench-verified"),
                "cache_dir": str(self.base_dir / "data" / "cache"),
                "max_context_length": 9192,
            },
            "models": {
                "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
                "precision": "fp16",
                "max_new_tokens": 2048,
                "temperature": 0.2,
                "top_p": 0.95,
                "repo_cache_dir": str(self.base_dir / "models" / "cache"),
            },
            "reasoning": {
                "cot_steps": 5,
                "tot_breadth": 3,
                "tot_depth": 3,
                "reflection_iterations": 3,
            },
            "logging": {
                "log_dir": str(self.base_dir / "logs"),
                "log_level": "INFO",
            },
            "evaluation": {
                "metrics": ["success_rate", "code_quality", "execution_time", "patch_quality"],
                "results_dir": str(self.base_dir / "results"),
            }
        }

        # Load configuration from file if provided
        if config_path:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self._merge_configs(self.defaults, user_config)

        # Load model-specific configurations
        self.model_configs = self._load_model_configs()

    def _merge_configs(self, base, override):
        """Recursively merge override into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def _load_model_configs(self):
        """Load model-specific configurations from YAML file."""
        # Try multiple possible locations for the model configs
        possible_paths = [
            self.base_dir / "config" / "model_configs.yaml",
            self.base_dir / "configs" / "models" / "model_configs.yaml",
            Path("configs/models/model_configs.yaml")
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
                
        # If no config file found, log a warning and return empty dict
        print("Warning: No model configuration file found. Using default configurations.")
        return {}

    def get_model_config(self, model_name):
        """Get configuration for a specific model."""
        if model_name in self.model_configs:
            return self.model_configs[model_name]
        return {}

    def __getitem__(self, key):
        return self.defaults[key]

    def __setitem__(self, key, value):
        self.defaults[key] = value
