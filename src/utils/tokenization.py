# src/utils/tokenization.py

from typing import List, Callable
import re
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger("tokenization")


# Default tokenizer function (simple approximation)
def _default_tokenizer(text: str) -> List[str]:
    """
    Simple default tokenizer that approximates tokens.
    This is a fallback and should not be used for accurate token counting.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens (approximate)
    """
    # Split on whitespace and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


# Registry to store tokenizer functions for different models
_tokenizer_registry = {}


def register_tokenizer(model_name: str, tokenizer_fn: Callable[[str], List[str]]) -> None:
    """
    Register a tokenizer function for a model.

    Args:
        model_name: Name of the model
        tokenizer_fn: Tokenizer function that takes text and returns tokens
    """
    _tokenizer_registry[model_name.lower()] = tokenizer_fn
    logger.debug(f"Registered tokenizer for model: {model_name}")


def get_tokenizer(model_name: str) -> Callable[[str], List[str]]:
    """
    Get tokenizer function for a model.

    Args:
        model_name: Name of the model

    Returns:
        Tokenizer function
    """
    # Try to find an exact match
    tokenizer = _tokenizer_registry.get(model_name.lower())

    # If not found, try to find a partial match
    if tokenizer is None:
        for registered_name, registered_tokenizer in _tokenizer_registry.items():
            if registered_name in model_name.lower():
                tokenizer = registered_tokenizer
                break

    # If still not found, use default tokenizer
    if tokenizer is None:
        logger.warning(f"No tokenizer found for model: {model_name}. Using default tokenizer.")
        tokenizer = _default_tokenizer

    return tokenizer


def count_tokens(text: str, model_name: str = "default") -> int:
    """
    Count the number of tokens in text for a specific model.

    Args:
        text: Text to count tokens in
        model_name: Name of the model

    Returns:
        Number of tokens
    """
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer(text)
    return len(tokens)


def truncate_to_token_limit(text: str, max_tokens: int, model_name: str = "default") -> str:
    """
    Truncate text to fit within a token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model_name: Name of the model

    Returns:
        Truncated text
    """
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer(text)

    if len(tokens) <= max_tokens:
        return text

    # This is an approximation since we can't easily reconstruct the text
    # For accurate truncation, use the model-specific tokenizer and detokenizer
    truncated_tokens = tokens[:max_tokens]
    truncated_text = " ".join(truncated_tokens)

    logger.warning(f"Text truncated from {len(tokens)} to {max_tokens} tokens.")
    return truncated_text


# Register model-specific tokenizers if available
try:
    from transformers import AutoTokenizer


    def _register_huggingface_tokenizers():
        """Register HuggingFace tokenizers for common models."""
        common_models = [
            "gpt2",
            "Qwen",
            "llama",
            "mistral",
            "Mixtral",
            "opt",
            "bloom"
        ]

        for model_name in common_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")

                def make_tokenizer_fn(tok):
                    return lambda text: tok.encode(text)

                register_tokenizer(model_name, make_tokenizer_fn(tokenizer))
                logger.info(f"Registered HuggingFace tokenizer for {model_name}")
            except Exception as e:
                logger.debug(f"Could not load HuggingFace tokenizer for {model_name}: {e}")


    # Register HuggingFace tokenizers
    _register_huggingface_tokenizers()
except ImportError:
    logger.debug("HuggingFace transformers not available. Using default tokenizer.")


# Approximate tokenizers for major models
def _gpt_tokenizer(text: str) -> List[str]:
    """
    Approximate tokenizer for GPT models.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens (approximate)
    """
    # GPT models generally use byte-pair encoding
    # This is a rough approximation: ~4 chars per token
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    tokens = []
    for word in words:
        # Approximate token count per word
        word_tokens = (len(word) + 3) // 4
        tokens.extend([word] * max(1, word_tokens))
    return tokens


def _claude_tokenizer(text: str) -> List[str]:
    """
    Approximate tokenizer for Claude models.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens (approximate)
    """
    # Claude models also use BPE but with different vocabulary
    # This is a rough approximation: ~4.5 chars per token
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    tokens = []
    for word in words:
        # Approximate token count per word
        word_tokens = (len(word) + 4) // 4
        tokens.extend([word] * max(1, word_tokens))
    return tokens


# Register approximate tokenizers
register_tokenizer("gpt", _gpt_tokenizer)
register_tokenizer("openai", _gpt_tokenizer)
register_tokenizer("claude", _claude_tokenizer)
register_tokenizer("anthropic", _claude_tokenizer)