# src/models/huggingface_model.py


import time
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import TextIteratorStreamer

from src.models.base_model import BaseModel
from src.utils.tokenization import count_tokens


class HuggingFaceModel(BaseModel):
    """Implementation of BaseModel for HuggingFace models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace model.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        self.model_name = config["model_name"]
        self.device_map = config.get("device_map", "auto")
        self.use_fp16 = config.get("use_fp16", True)
        self.use_8bit = config.get("use_8bit", False)
        self.use_4bit = config.get("use_4bit", False)
        self.cache_dir = config.get("cache_dir")
        self.max_length = config.get("max_length", 2048)
        self.streaming = config.get("streaming", False)

        # Generation parameters
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 50)
        self.repetition_penalty = config.get("repetition_penalty", 1.1)
        self.do_sample = config.get("do_sample", True)

        # Load model and tokenizer
        self.load_model()

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        self.logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()

        try:
            # Determine the data type
            if self.use_fp16:
                dtype = torch.float16
            else:
                dtype = torch.float32

            # Determine quantization parameters
            quantization_config = None
            if self.use_8bit:
                self.logger.info("Using 8-bit quantization")
                quantization_config = {"load_in_8bit": True}
            elif self.use_4bit:
                self.logger.info("Using 4-bit quantization")
                quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": dtype}

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

            # Ensure the tokenizer has a pad token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token = "</s>"

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                quantization_config=quantization_config
            )

            # Create generator pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False,  # Don't return the prompt
                handle_long_generation="hole"  # Handle long generations
            )

            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")

            # Log model size and device placement
            model_size = sum(p.numel() for p in self.model.parameters()) / 1000000
            self.logger.info(f"Model size: {model_size:.2f}M parameters")
            self.logger.info(
                f"Model device map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'all on ' + str(next(self.model.parameters()).device)}")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, prompt: str) -> str:
        """
        Generate text using the model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        try:
            # Count tokens for metrics
            tokens_in = count_tokens(prompt, self.model_name)

            # Set up generation parameters
            generation_config = {
                "max_new_tokens": self.config.get("max_new_tokens", 512),
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }

            self.logger.info(f"Generating with {tokens_in} input tokens")

            # Generate text
            if self.streaming:
                # Streaming generation - not using pipeline to enable streaming
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                streamer = TextIteratorStreamer(self.tokenizer)

                # Generate asynchronously
                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "streamer": streamer,
                    **generation_config
                }

                # Start generation in a separate thread
                from threading import Thread
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                # Collect the generated text
                generated_text = ""
                for text in streamer:
                    generated_text += text

                thread.join()
            else:
                # Non-streaming generation using pipeline
                outputs = self.generator(
                    prompt,
                    **generation_config
                )

                generated_text = outputs[0]["generated_text"]

            # Count output tokens for metrics
            tokens_out = count_tokens(generated_text, self.model_name)

            # Record successful request
            self._record_request(tokens_in, tokens_out, True)

            return generated_text

        except Exception as e:
            self.logger.error(f"Error in text generation: {str(e)}")
            # Record failed request (estimate token count)
            self._record_request(tokens_in, 0, False)
            raise

    def tokenize(self, text: str) -> Dict[str, Any]:
        """
        Tokenize the input text.

        Args:
            text: Input text

        Returns:
            Tokenization result
        """
        return self.tokenizer(text, return_tensors="pt")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def clear_cuda_cache(self) -> None:
        """Clear CUDA cache to free up memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared")
            