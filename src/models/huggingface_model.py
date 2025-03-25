# src/models/huggingface_model.py


import time
import gc
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import TextIteratorStreamer, BitsAndBytesConfig

from src.models.base_model import BaseModel
from src.utils.tokenization import count_tokens


class HuggingFaceModel(BaseModel):
    """Implementation of BaseModel for HuggingFace models with memory optimization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HuggingFace model with memory optimizations.

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

        # Memory optimization options
        self.offload_folder = config.get("offload_folder", "offload_folder")
        self.enable_offloading = config.get("enable_offloading", True)
        self.low_cpu_mem_usage = config.get("low_cpu_mem_usage", True)

        # Generation parameters
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.95)
        self.top_k = config.get("top_k", 50)
        self.repetition_penalty = config.get("repetition_penalty", 1.1)
        self.do_sample = config.get("do_sample", True)

        # Load model and tokenizer
        self.load_model()

    def _memory_cleanup(self):
        """Perform memory cleanup operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared")

    def load_model(self) -> None:
        """Load the model and tokenizer with memory optimizations."""
        self.logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()

        try:
            # Clean up memory before loading
            self._memory_cleanup()

            # Determine the data type
            dtype = torch.float16 if self.use_fp16 else torch.float32
            self.logger.info(f"Using precision: {dtype}")

            # Load tokenizer first (less memory intensive)
            self.logger.info("Loading tokenizer...")
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
                self.logger.info(f"Set pad_token to {self.tokenizer.pad_token}")

            # Configure quantization for memory efficiency
            quantization_config = None
            if self.use_4bit:
                self.logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            elif self.use_8bit:
                self.logger.info("Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_has_fp16_weight=True
                )

            # Load model with memory optimizations
            self.logger.info("Loading model with memory optimizations...")
            load_kwargs = {
                "device_map": self.device_map,
                "torch_dtype": dtype,
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            }

            # Add quantization config if specified
            if quantization_config:
                load_kwargs["quantization_config"] = quantization_config

            # Add offload folder if enabled
            if self.enable_offloading:
                import os
                os.makedirs(self.offload_folder, exist_ok=True)
                load_kwargs["offload_folder"] = self.offload_folder

            # Try loading the model with progressively more aggressive memory optimizations
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **load_kwargs
                )
            except Exception as e:
                self.logger.warning(
                    f"Initial loading failed: {str(e)}. Trying with more aggressive memory optimizations...")

                # Clean up memory
                self._memory_cleanup()

                # If 4-bit failed or 8-bit failed, try with 8-bit
                if self.use_4bit and not self.use_8bit:
                    self.logger.info("Falling back to 8-bit quantization")
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **load_kwargs
                    )
                else:
                    # If all else fails, try the most aggressive memory saving
                    self.logger.info("Using maximum memory optimization settings")

                    # Force CPU offloading for some layers
                    if "device_map" in load_kwargs:
                        load_kwargs["device_map"] = "auto"

                    # Enable model offloading to disk
                    load_kwargs["offload_state_dict"] = True

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **load_kwargs
                    )

            # Create lightweight generator pipeline with memory optimizations
            self.logger.info("Creating text generation pipeline...")
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
        Generate text using the model with memory optimizations.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        try:
            # Clean up memory before generation
            self._memory_cleanup()

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

            # Handle very long prompts
            if tokens_in > self.max_length - 100:  # Leave room for generation
                self.logger.warning(f"Prompt too long ({tokens_in} tokens). Truncating to fit model context.")
                # Truncate from the beginning to preserve the most recent context
                prompt_tokens = self.tokenizer.encode(prompt)
                truncated_tokens = prompt_tokens[-(self.max_length - 100):]
                prompt = self.tokenizer.decode(truncated_tokens)
                tokens_in = len(truncated_tokens)
                self.logger.info(f"Truncated prompt to {tokens_in} tokens")

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
                # Non-streaming generation using pipeline with memory optimization
                try:
                    outputs = self.generator(
                        prompt,
                        **generation_config
                    )
                    generated_text = outputs[0]["generated_text"]
                except torch.cuda.OutOfMemoryError:
                    # Handle OOM during generation
                    self.logger.warning("CUDA out of memory during generation. Trying with reduced parameters.")
                    self._memory_cleanup()

                    # Reduce parameters to save memory
                    reduced_config = generation_config.copy()
                    reduced_config["max_new_tokens"] = min(256, reduced_config["max_new_tokens"])

                    outputs = self.generator(
                        prompt,
                        **reduced_config
                    )
                    generated_text = outputs[0]["generated_text"]

            # Count output tokens for metrics
            tokens_out = count_tokens(generated_text, self.model_name)

            # Record successful request
            self._record_request(tokens_in, tokens_out, True)

            # Clean up memory after generation
            self._memory_cleanup()

            return generated_text

        except Exception as e:
            self.logger.error(f"Error in text generation: {str(e)}")
            # Record failed request (estimate token count)
            self._record_request(tokens_in, 0, False)
            # Clean up on error
            self._memory_cleanup()
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
        self._memory_cleanup()

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts

        Returns:
            List of generated texts
        """
        return [self.generate(prompt) for prompt in prompts]

