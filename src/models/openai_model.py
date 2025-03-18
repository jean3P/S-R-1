# src/models/openai_model.py

import os
import time
from typing import Dict, Any
import openai

from src.models.base_model import BaseModel
from src.utils.tokenization import count_tokens


class OpenAIModel(BaseModel):
    """Implementation of BaseModel for OpenAI API models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI model.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Extract configuration
        self.model_name = config["model_name"]
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.organization = config.get("organization") or os.environ.get("OPENAI_ORGANIZATION")
        self.api_base = config.get("api_base")
        self.api_version = config.get("api_version")
        self.api_type = config.get("api_type", "open_ai")

        # Generation parameters
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 1.0)
        self.max_tokens = config.get("max_tokens", 1024)
        self.presence_penalty = config.get("presence_penalty", 0.0)
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.stop_sequences = config.get("stop_sequences", None)

        # Rate limiting parameters
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.max_retry_delay = config.get("max_retry_delay", 60)

        # System message for chat models
        self.system_message = config.get("system_message", "You are a helpful AI assistant.")

        # Initialize client
        self._setup_client()

    def _setup_client(self) -> None:
        """Set up the OpenAI API client."""
        if not self.api_key:
            self.logger.error(
                "OpenAI API key not found. Set it in the config or as OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key not found")

        # Initialize the client
        client_kwargs = {"api_key": self.api_key}

        if self.organization:
            client_kwargs["organization"] = self.organization

        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        if self.api_version:
            client_kwargs["api_version"] = self.api_version

        if self.api_type != "open_ai":
            # Handle Azure OpenAI or other API types
            client_kwargs["api_type"] = self.api_type

        try:
            self.client = openai.OpenAI(**client_kwargs)
            self.logger.info(f"OpenAI client initialized for model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {str(e)}")
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
        tokens_in = count_tokens(prompt, "gpt")  # Use GPT tokenizer approximation

        # Check if this is a chat model
        is_chat_model = "gpt" in self.model_name.lower() and "turbo" in self.model_name.lower()
        is_gpt4 = "gpt-4" in self.model_name.lower()

        attempt = 0
        retry_delay = self.retry_delay

        while attempt < self.max_retries:
            attempt += 1

            try:
                start_time = time.time()

                if is_chat_model or is_gpt4:
                    # Format as a chat completion
                    messages = [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt}
                    ]

                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        stop=self.stop_sequences
                    )

                    generated_text = response.choices[0].message.content
                    tokens_out = response.usage.completion_tokens

                else:
                    # Format as a text completion
                    response = self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        stop=self.stop_sequences
                    )

                    generated_text = response.choices[0].text
                    tokens_out = response.usage.completion_tokens

                generation_time = time.time() - start_time
                self.logger.info(f"Generated {tokens_out} tokens in {generation_time:.2f}s")

                # Record successful request
                self._record_request(tokens_in, tokens_out, True)

                return generated_text

            except openai.RateLimitError as e:
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

            except openai.APIError as e:
                self.logger.error(f"OpenAI API error: {str(e)}")
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
        Tokenize the input text (approximation for OpenAI models).

        Args:
            text: Input text

        Returns:
            Tokenization result (approximate)
        """
        # OpenAI doesn't expose tokenizer directly, this is an approximation
        token_count = count_tokens(text, "gpt")
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
        return count_tokens(text, "gpt")
    