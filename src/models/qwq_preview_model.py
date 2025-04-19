# src/models/qwq_preview_model.py
from .base_model import BaseModel
from typing import Dict, Any


class QwqPreviewModel(BaseModel):
    """Implementation for QwQ 32B Preview model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("qwq-preview", config)
        print(f"Initialized QwQ Preview model with repo_id: {self.model_config.get('repo_id')}")

    def format_prompt(self, instruction: str) -> str:
        """Format instruction according to QwQ's prompt template."""
        template = self.model_config.get("prompt_template",
                                         "<|im_start|>system\nYou are a helpful AI assistant.\n<|im_end|>\n"
                                         "<|im_start|>user\n{instruction}\n<|im_end|>\n"
                                         "<|im_start|>assistant\n")
        return template.format(instruction=instruction)
