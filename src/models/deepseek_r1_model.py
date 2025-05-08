from .base_model import BaseModel
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DeepseekR1Model(BaseModel):
    """Implementation for DeepSeek R1 Distill Qwen 32B model."""

    def __init__(self, config):
        """Initialize the model with proper configuration."""
        # Pass the model name and config to the parent class
        try:
            super().__init__("deepseek-r1-distill", config)
            repo_id = self.model_config.get('repo_id', 'Unknown')
            logger.info(f"Initialized DeepSeek R1 model with repo_id: {repo_id}")
        except Exception as e:
            logger.error(f"Error initializing DeepSeek R1 model: {e}")
            raise

    def format_prompt(self, instruction: str) -> str:
        """Format instruction according to DeepSeek's prompt template."""
        template = self.model_config.get("prompt_template",
                                        "<｜begin▁of▁sentence｜>\nHuman: {instruction}\n\nAssistant:")
        return template.format(instruction=instruction)
