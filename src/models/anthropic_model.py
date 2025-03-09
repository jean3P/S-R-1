# src/models/anthropic_model.py

import os
import time
from typing import Dict, Any
import anthropic

from src.models.base_model import BaseModel
from src.utils.tokenization import count_tokens


class AnthropicModel(BaseModel):
    """Implementation of BaseModel for Anthropic Claude models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Anthropic model.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Extract configuration
        self.model_name = config["model_name"]
        self.api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        self.api_version = config.get("api_version", "2023-06-01")

        # Generation parameters
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", None)
        self.max_tokens = config.get("max_tokens", 1024)
        self.stop_sequences = config.get("stop_sequences", None)

        # Rate limiting parameters
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.max_retry_delay = config.get("max_retry_delay", 60)

        # System message / prompt
        self.system_prompt = config.get("system_prompt", "")

        # Initialize client
        self._setup_client()

    def _setup_client(self) -> None:
        """Set up the Anthropic API client."""
        if not self.api_key:
            self.logger.error(
                "Anthropic API key not found. Set it in the config or as ANTHROPIC_API_KEY environment variable.")
            raise ValueError("Anthropic API key not found")

        # Initialize the client
        try:
            client_kwargs = {"api_key": self.api_key}

            if self.api_version:
                # Set version via anthropic.API_VERSION or headers depending on SDK version
                try:
                    client_kwargs["api_version"] = self.api_version
                except:
                    # For older versions of the SDK, try setting via headers
                    client_kwargs["headers"] = {"anthropic-version": self.api_version}

            self.client = anthropic.Anthropic(**client_kwargs)
            self.logger.info(f"Anthropic client initialized for model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing Anthropic client: {str(e)}")
            raise

    def generate(self, prompt: str) -> str:
        """
        Generate text using the model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # Count tokens for metrics
        tokens_in = count_tokens(prompt, "claude")  # Use Claude tokenizer approximation

        attempt = 0
        retry_delay = self.retry_delay

        while attempt < self.max_retries:
            attempt += 1

            try:
                start_time = time.time()

                # Prepare message parameters
                params = {
                    "model": self.model_name,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }

                # Add top_k if specified
                if self.top_k is not None:
                    params["top_k"] = self.top_k

                # Add stop sequences if specified
                if self.stop_sequences:
                    params["stop_sequences"] = self.stop_sequences

                # Check if system prompt is provided
                if self.system_prompt:
                    params["system"] = self.system_prompt
                    params["messages"] = [{"role": "user", "content": prompt}]

                    # Use messages API
                    response = self.client.messages.create(**params)
                    generated_text = response.content[0].text
                else:
                    # Use completion API with Claude prompt format
                    params["prompt"] = anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT

                    # Use completions API
                    response = self.client.completions.create(**params)
                    generated_text = response.completion

                generation_time = time.time() - start_time

                # Estimate output tokens (Anthropic doesn't always return token counts)
                tokens_out = count_tokens(generated_text, "claude")

                self.logger.info(f"Generated ~{tokens_out} tokens in {generation_time:.2f}s")

                # Record successful request
                self._record_request(tokens_in, tokens_out, True)

                return generated_text

            except anthropic.RateLimitError as e:
                if attempt < self.max_retries:
                    self.logger.warning(
                        f"Rate limit error, retrying in {retry_delay}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(retry_delay)
                    # Exponential backoff with jitter
                    retry_delay = min(retry_delay * 2, self.max_retry_delay) * (0.5 + 0.5 * (time.time() % 1))
                else:
                    self.logger.error(f"Rate limit error after {self.max_retries} attempts")
                    self._record_request(tokens_in, 0, False)
                    raise

            except anthropic.APIError as e:
                self.logger.error(f"Anthropic API error: {str(e)}")
                self._record_request(tokens_in, 0, False)
                raise

            except Exception as e:
                self.logger.error(f"Unexpected error in text generation: {str(e)}")
                self._record_request(tokens_in, 0, False)
                raise

        # If we've exhausted retries
        raise Exception(f"Failed to generate text after {self.max_retries} attempts")

    def tokenize(self, text: str) -> Dict[str, Any]:
        """
        Tokenize the input text (approximation for Anthropic models).

        Args:
            text: Input text

        Returns:
            Tokenization result (approximate)
        """
        # Anthropic doesn't expose tokenizer directly, this is an approximation
        token_count = count_tokens(text, "claude")
        return {
            "input_ids": list(range(token_count)),  # Placeholder
            "token_count": token_count
        }

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        return count_tokens(text, "claude")
