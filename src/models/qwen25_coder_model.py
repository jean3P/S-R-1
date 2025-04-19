# src/models/qwen25_coder_model.py
from .base_model import BaseModel
from typing import Dict, Any


class Qwen25CoderModel(BaseModel):
    """Implementation for Qwen 2.5 Coder 32B Instruct model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("qwen2-5-coder", config)
        print(f"Initialized Qwen 2.5 Coder model with repo_id: {self.model_config.get('repo_id')}")

    def format_prompt(self, instruction: str) -> str:
        """Format instruction according to Qwen 2.5's prompt template."""
        template = self.model_config.get("prompt_template",
                                         "<|im_start|>system\nYou are a helpful AI assistant.\n<|im_end|>\n"
                                         "<|im_start|>user\n{instruction}\n<|im_end|>\n"
                                         "<|im_start|>assistant\n")
        return template.format(instruction=instruction)
