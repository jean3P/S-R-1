# models/__init__.py
from .base_model import BaseModel
from .deepseek_model import DeepseekModel
from .qwen_model import QwenModel
from .qwen_coder_model import QwenCoderModel
from .qwq_model import QwqModel


# Factory function to create model instances
def create_model(model_name: str, config):
    """
    Factory function to create the appropriate model instance.

    Args:
        model_name: Name of the model to create.
        config: Configuration object.

    Returns:
        An instance of the requested model.
    """
    model_classes = {
        "deepseek-7b": DeepseekModel,
        "qwen-7b": QwenModel,
        "qwen-coder-7b": QwenCoderModel,
        "qwq-7b": QwqModel,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_classes.keys())}")

    return model_classes[model_name](config)

