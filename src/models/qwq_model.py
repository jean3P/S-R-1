# models/qwq_model.py
from .base_model import BaseModel
from typing import Dict, Any


class QwqModel(BaseModel):
    """Implementation for QWQ models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("yi-chat", config)

    def format_prompt(self, instruction: str) -> str:
        """Format instruction according to QWQ's prompt template."""
        template = self.model_config.get("prompt_template",
                                         "<|im_start|>system\nYou are a helpful AI assistant.\n<|im_end|>\n"
                                         "<|im_start|>user\n{instruction}\n<|im_end|>\n"
                                         "<|im_start|>assistant\n")
        return template.format(instruction=instruction)
