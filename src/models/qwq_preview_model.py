from .base_model import BaseModel
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class QwqPreviewModel(BaseModel):
    """Implementation for QwQ 32B Preview model."""

    def __init__(self, config):
        """Initialize the model with proper configuration."""
        # Pass the model name and config to the parent class
        try:
            super().__init__("qwq-preview", config)
            repo_id = self.model_config.get('repo_id', 'Unknown')
            logger.info(f"Initialized QwQ Preview model with repo_id: {repo_id}")
        except Exception as e:
            logger.error(f"Error initializing QwQ Preview model: {e}")
            raise

    def format_prompt(self, instruction: str) -> str:
        """Format instruction according to QwQ's prompt template."""
        template = self.model_config.get("prompt_template",
                                         "<|im_start|>system\nYou are a helpful AI assistant.\n<|im_end|>\n"
                                         "<|im_start|>user\n{instruction}\n<|im_end|>\n"
                                         "<|im_start|>assistant\n")
        return template.format(instruction=instruction)
