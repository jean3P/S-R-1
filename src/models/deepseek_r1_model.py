# src/models/deepseek_r1_model.py
from .base_model import BaseModel
from typing import Dict, Any


class DeepseekR1Model(BaseModel):
    """Implementation for DeepSeek R1 Distill Qwen 32B model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("deepseek-r1-distill", config)
        print(f"Initialized DeepSeek R1 model with repo_id: {self.model_config.get('repo_id')}")

    def format_prompt(self, instruction: str) -> str:
        """Format instruction according to DeepSeek's prompt template."""
        template = self.model_config.get("prompt_template",
                                        "<｜begin▁of▁sentence｜>\nHuman: {instruction}\n\nAssistant:")
        return template.format(instruction=instruction)
