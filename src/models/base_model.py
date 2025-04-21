# src/models/base_model.py
import os
import torch
import gc
import bitsandbytes
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
import logging
import optimum
from optimum.bettertransformer.transformation import BetterTransformer

logger = logging.getLogger(__name__)


class BaseModel:
    """Abstract base class for all language models."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the base model with memory optimizations.

        Args:
            model_name: Name of the model to use.
            config: Configuration dictionary for the model.
        """
        self.model_name = model_name
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config["models"]["device"]
        self.max_new_tokens = config["models"].get("max_new_tokens", 2048)
        self.temperature = config["models"].get("temperature", 0.2)
        self.top_p = config["models"].get("top_p", 0.95)

        # Get model-specific configuration
        self.model_config = config.get_model_config(model_name)

        # Memory optimization settings
        self.memory_efficient = True
        self.offload_to_cpu = True  # Enable offloading layers to CPU

        # Clear CUDA cache before loading model
        self._clear_memory()

        # Load model and tokenizer
        self._load_model_and_tokenizer()

    def _clear_memory(self):
        """Clear CUDA cache and run garbage collection to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Log memory stats if on CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                logger.info(f"GPU {i} memory: reserved={reserved:.2f}GB, allocated={allocated:.2f}GB")

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer with memory optimizations."""
        # Use model_name as repo_id if repo_id is not specified
        repo_id = self.model_config.get("repo_id", self.model_config.get("model_name"))
        if not repo_id:
            repo_id = self.model_name  # Fallback to the model name passed to __init__

        revision = self.model_config.get("revision", "main")
        trust_remote_code = self.model_config.get("trust_remote_code", False)
        cache_dir = self.config["models"].get("repo_cache_dir", "data/model_cache")

        # Get Hugging Face token from environment variable - try both common environment variable names
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        # If token is not set, log a warning
        if not hf_token and 'qwen' in repo_id.lower():
            logger.warning(
                f"Warning: HF_TOKEN environment variable not set. You may need to set it to access {repo_id}")
            logger.warning("Try: export HF_TOKEN=your_huggingface_token")

        try:
            # Load tokenizer with proper error handling
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                token=hf_token  # Add token for authentication
            )

            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Handle quantization - safely check if bitsandbytes is installed
            quantization_config = None
            if "quantization" in self.model_config:
                try:
                    import bitsandbytes
                    from transformers import BitsAndBytesConfig

                    quantization = self.model_config["quantization"]
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=quantization.get("bits", 4) == 4,
                        load_in_8bit=quantization.get("bits", 4) == 8,
                        llm_int8_threshold=quantization.get("threshold", 6.0),
                        llm_int8_has_fp16_weight=quantization.get("fp16_weight", False),
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=quantization.get("double_quant", True),
                        bnb_4bit_quant_type=quantization.get("quant_type", "nf4"),
                    )
                except ImportError:
                    logger.warning("Warning: bitsandbytes package not found. Quantization will be disabled.")
                    logger.warning("To enable quantization, install bitsandbytes: pip install bitsandbytes")
                    # Fall back to no quantization
                    self.model_config.pop("quantization", None)

            # Check GPU compute capability for Flash Attention support
            flash_attention_available = False
            use_flash_attention = self.model_config.get("use_flash_attention", True)

            if use_flash_attention and torch.cuda.is_available():
                try:
                    # Check if GPU architecture is supported
                    compute_capability = torch.cuda.get_device_capability(0)
                    # Ampere or newer (compute capability >= 8.0)
                    arch_supported = compute_capability[0] >= 8

                    if arch_supported:
                        import flash_attn
                        flash_attention_available = True
                    else:
                        logger.warning(
                            f"GPU architecture (compute capability {compute_capability[0]}.{compute_capability[1]}) doesn't support Flash Attention. Disabling it.")
                except ImportError:
                    logger.warning("Warning: flash_attn package not found. Flash attention will be disabled.")
                    logger.warning("To enable flash attention, install flash-attn: pip install flash-attn")

            # Set up device map for offloading to CPU if needed
            device_map = "auto" if self.device == "cuda" else None
            max_memory = None

            # Enable CPU offloading for large models
            if self.device == "cuda" and self.offload_to_cpu:
                # Calculate available GPU memory
                available_gpu_mem = {}
                for i in range(torch.cuda.device_count()):
                    # Reserve 2GB less than what's free to avoid OOM
                    free_mem = (torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)) - 2
                    available_gpu_mem[i] = f"{max(1, int(free_mem))}GiB"

                # Add CPU memory
                available_gpu_mem["cpu"] = "32GiB"  # Allow offloading to CPU
                max_memory = available_gpu_mem
                logger.info(f"Setting up memory offloading with: {max_memory}")

            # Check if the model supports gradient checkpointing
            supports_gradient_checkpointing = True
            if "qwen" in repo_id.lower():
                supports_gradient_checkpointing = False  # Qwen models don't support this parameter

            # Create model args dictionary
            model_args = {
                "revision": revision,
                "cache_dir": cache_dir,
                "trust_remote_code": trust_remote_code,
                "device_map": device_map,
                "max_memory": max_memory,
                "quantization_config": quantization_config,
                "torch_dtype": torch.float16 if self.config["models"]["precision"] == "fp16" else torch.float32,
                "token": hf_token,  # Add token for authentication
                "use_flash_attention_2": flash_attention_available,
                # Add memory optimizations
                "low_cpu_mem_usage": True,
                "offload_folder": "offload_folder",
                "offload_state_dict": self.offload_to_cpu,
            }

            # Add gradient checkpointing only if supported
            if supports_gradient_checkpointing:
                model_args["gradient_checkpointing"] = True

            # Load model with the appropriate args
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    **model_args
                )
            except (ValueError, RuntimeError) as e:
                # Check for Flash Attention error
                if "FlashAttention only supports Ampere GPUs or newer" in str(e):
                    logger.warning("FlashAttention error detected, trying again without Flash Attention")
                    model_args["use_flash_attention_2"] = False
                    self.model = AutoModelForCausalLM.from_pretrained(
                        repo_id,
                        **model_args
                    )
                # If quantization fails, try without it
                elif "quantization_config" in model_args and "quantization" in str(e).lower():
                    logger.warning(f"Quantization failed, trying without quantization: {str(e)}")
                    model_args.pop("quantization_config")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        repo_id,
                        **model_args
                    )
                else:
                    # Re-raise if it's not a known issue
                    raise

            # Set eval mode to save memory
            self.model.eval()

            # Clear cache after loading
            self._clear_memory()

        except Exception as e:
            if "is not a local folder and is not a valid model identifier" in str(e):
                raise Exception(f"Error loading model {repo_id}. This may be a private repository or does not exist. "
                                f"Set the HF_TOKEN environment variable with your Hugging Face token or check if the model ID is correct.") from e
            elif "No package metadata was found for bitsandbytes" in str(e):
                raise Exception(f"Error loading model {repo_id}. The bitsandbytes package is not installed properly. "
                                f"Try: pip install bitsandbytes==0.41.0 or disable quantization in the model config.") from e
            elif "CUDA out of memory" in str(e):
                # Provide helpful guidance for CUDA OOM errors
                available_mem = 0
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                        free_mem_gb = free_mem / (1024 ** 3)
                        available_mem = max(available_mem, free_mem_gb)

                raise Exception(f"CUDA out of memory while loading model {repo_id}. "
                                f"Available GPU memory: {available_mem:.2f}GB. "
                                f"Try using --cpu-only mode, or enable better memory management with "
                                f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True") from e
            else:
                raise e

    def format_prompt(self, instruction: str) -> str:
        """Format instruction according to model's prompt template."""
        template = self.model_config.get("prompt_template", "{instruction}")
        return template.format(instruction=instruction)

    @torch.no_grad()
    def generate(self, instruction: str, **kwargs) -> str:
        """
        Generate a response to the given instruction with memory optimization.

        Args:
            instruction: The instruction to respond to.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Generated text response.
        """
        # Clear memory before generation
        self._clear_memory()

        formatted_prompt = self.format_prompt(instruction)

        # Process input in efficient chunks if it's very long
        if len(formatted_prompt) > 12000 and self.memory_efficient:
            # Truncate to a reasonable length
            logger.warning(f"Truncating very long prompt from {len(formatted_prompt)} chars to 12000 chars")
            formatted_prompt = formatted_prompt[:12000]

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.temperature > 0,
            # Memory optimization for generation
            "use_cache": True,
            **kwargs
        }

        # Generate with memory optimization
        try:
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # If we hit OOM during generation, try to recover by clearing cache and reducing context
                logger.warning("CUDA OOM during generation. Attempting recovery by reducing context...")
                self._clear_memory()

                # Truncate the input further
                tokens = self.tokenizer.encode(formatted_prompt)
                max_length = min(len(tokens) // 2, 2000)  # Take half or at most 2000 tokens
                truncated_tokens = tokens[-max_length:]

                # Regenerate with truncated input
                inputs = {"input_ids": torch.tensor([truncated_tokens]).to(self.device)}
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )

                # Add a note about truncation
                truncation_note = "\n[Note: The input was too long and had to be truncated for processing. The response may not address all details from the original prompt.]\n\n"
            else:
                raise e

        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the response
        response = response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

        # Add truncation note if applicable
        if 'truncation_note' in locals():
            response = truncation_note + response

        # Clear memory after generation
        self._clear_memory()

        return response.strip()

    def get_logits(self, text: str) -> torch.Tensor:
        """Get the logits for the given text."""
        # Clear memory before operation
        self._clear_memory()

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Clear memory after operation
        self._clear_memory()

        return outputs.logits


class TextStreamer:
    """Helper class for text streaming during generation."""

    def __init__(self, tokenizer, skip_prompt=True, callback=None):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.callback = callback
        self.tokens_buffer = []
        self.prompt_processed = False

    def put(self, token_id):
        """Process a token."""
        self.tokens_buffer.append(token_id)

        # Skip tokens until we're past the prompt
        if self.skip_prompt and not self.prompt_processed:
            # Detect when we've moved past the prompt based on special tokens
            if token_id in [self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]:
                self.prompt_processed = True
            return

        # Decode the buffer
        text = self.tokenizer.decode(self.tokens_buffer)

        # Call the callback with the decoded text
        if self.callback:
            self.callback(text)

        # Clear the buffer
        self.tokens_buffer = []

    def end(self):
        """Process any remaining tokens."""
        if self.tokens_buffer and self.callback:
            text = self.tokenizer.decode(self.tokens_buffer)
            self.callback(text)
            self.tokens_buffer = []
