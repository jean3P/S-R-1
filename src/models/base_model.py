# src/models/base_model.py
import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseModel:
    """Abstract base class for all language models."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
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

        self._clear_memory()
        self._load_model_and_tokenizer()

    def _clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                logger.info(f"GPU {i} memory: reserved={reserved:.2f}GB, allocated={allocated:.2f}GB")

    def _load_model_and_tokenizer(self):
        repo_id = self.model_config.get("repo_id", self.model_name)
        revision = self.model_config.get("revision", "main")
        trust_remote_code = self.model_config.get("trust_remote_code", False)
        cache_dir = self.config["models"].get("repo_cache_dir", "data/model_cache")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        if not hf_token and 'qwen' in repo_id.lower():
            logger.warning(f"Warning: HF_TOKEN environment variable not set. You may need it for {repo_id}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                token=hf_token
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # ✅ Lazy-load bitsandbytes if needed
            quantization_config = None
            if "quantization" in self.model_config:
                try:
                    from transformers import BitsAndBytesConfig
                    import bitsandbytes

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
                    logger.warning("Quantization requested, but bitsandbytes not available.")
                    self.model_config.pop("quantization", None)

            # ✅ Flash Attention support
            flash_attention_available = False
            use_flash_attention = self.model_config.get("use_flash_attention", True)
            if use_flash_attention and torch.cuda.is_available():
                try:
                    compute_capability = torch.cuda.get_device_capability(0)
                    if compute_capability[0] >= 8:  # Ampere or newer
                        import flash_attn
                        flash_attention_available = True
                except ImportError:
                    logger.warning("flash_attn not found. Disabling flash attention.")
                    flash_attention_available = False

            device_map = "auto" if self.device == "cuda" else None
            max_memory = None

            if self.device == "cuda" and self.offload_to_cpu:
                available_gpu_mem = {}
                for i in range(torch.cuda.device_count()):
                    free_mem = (torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)) - 2
                    available_gpu_mem[i] = f"{max(1, int(free_mem))}GiB"
                available_gpu_mem["cpu"] = "32GiB"
                max_memory = available_gpu_mem
                logger.info(f"Using memory offloading: {max_memory}")

            supports_gradient_checkpointing = not "qwen" in repo_id.lower()

            model_args = {
                "revision": revision,
                "cache_dir": cache_dir,
                "trust_remote_code": trust_remote_code,
                "device_map": device_map,
                "max_memory": max_memory,
                "quantization_config": quantization_config,
                "torch_dtype": torch.float16 if self.config["models"]["precision"] == "fp16" else torch.float32,
                "token": hf_token,
                "use_flash_attention_2": flash_attention_available,
                "low_cpu_mem_usage": True,
                "offload_folder": "offload_folder",
                "offload_state_dict": self.offload_to_cpu,
            }

            if supports_gradient_checkpointing:
                model_args["gradient_checkpointing"] = True

            try:
                self.model = AutoModelForCausalLM.from_pretrained(repo_id, **model_args)
            except (ValueError, RuntimeError) as e:
                if "FlashAttention" in str(e):
                    logger.warning("Retrying without flash attention due to error.")
                    model_args["use_flash_attention_2"] = False
                    self.model = AutoModelForCausalLM.from_pretrained(repo_id, **model_args)
                elif "quantization_config" in model_args and "quantization" in str(e).lower():
                    logger.warning("Quantization error, retrying without it.")
                    model_args.pop("quantization_config")
                    self.model = AutoModelForCausalLM.from_pretrained(repo_id, **model_args)
                else:
                    raise

            self.model.eval()
            self._clear_memory()

        except Exception as e:
            raise RuntimeError(f"Error loading model {repo_id}: {str(e)}") from e

    def format_prompt(self, instruction: str) -> str:
        template = self.model_config.get("prompt_template", "{instruction}")
        return template.format(instruction=instruction)

    @torch.no_grad()
    def generate(self, instruction: str, **kwargs) -> str:
        self._clear_memory()
        formatted_prompt = self.format_prompt(instruction)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.temperature > 0,
            "use_cache": True,
            **kwargs
        }

        try:
            outputs = self.model.generate(**inputs, **generation_config)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("OOM: reducing input size and retrying...")
                self._clear_memory()
                tokens = self.tokenizer.encode(formatted_prompt)
                truncated_tokens = tokens[-2000:]
                inputs = {"input_ids": torch.tensor([truncated_tokens]).to(self.device)}
                outputs = self.model.generate(**inputs, **generation_config)
                truncation_note = "[Truncated input due to OOM]\n\n"
            else:
                raise

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        if 'truncation_note' in locals():
            response = truncation_note + response
        self._clear_memory()
        return response.strip()

    def get_logits(self, text: str) -> torch.Tensor:
        self._clear_memory()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
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
