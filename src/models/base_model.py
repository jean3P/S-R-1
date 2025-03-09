# src/models/base_model.py

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(ABC):
    """Abstract base class for all generative models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.

        Args:
            config: Model configuration
        """
        from src.utils.logging import get_logger

        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Initialize model-specific metrics
        self.metrics = {
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text based on the prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> Dict[str, Any]:
        """
        Tokenize the input text.

        Args:
            text: Input text

        Returns:
            Tokenization result
        """
        pass

    def _record_request(self, tokens_in: int, tokens_out: int, success: bool) -> None:
        """
        Record metrics for a generation request.

        Args:
            tokens_in: Number of input tokens
            tokens_out: Number of output tokens
            success: Whether the request was successful
        """
        self.metrics["total_tokens_in"] += tokens_in
        self.metrics["total_tokens_out"] += tokens_out
        self.metrics["total_requests"] += 1

        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current model metrics.

        Returns:
            Dictionary of model metrics
        """
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset the model metrics."""
        self.metrics = {
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }