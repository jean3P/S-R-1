# models/deepseek_model.py
from .base_model import BaseModel
from typing import Dict, Any


class DeepseekModel(BaseModel):
    """Implementation for Deepseek models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("deepseek-7b", config)

    def format_prompt(self, instruction: str) -> str:
        """Format instruction according to Deepseek's prompt template."""
        template = self.model_config.get("prompt_template", "<｜begin▁of▁sentence｜>\nHuman: {instruction}\n\nAssistant:")
        return template.format(instruction=instruction)
