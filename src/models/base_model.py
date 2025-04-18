# models/base_model.py
from abc import ABC
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any


class BaseModel(ABC):
    """Abstract base class for all language models."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the base model.

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

        # Load model and tokenizer
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        # Use model_name as repo_id if repo_id is not specified
        repo_id = self.model_config.get("repo_id", self.model_config.get("model_name"))
        if not repo_id:
            repo_id = self.model_name  # Fallback to the model name passed to __init__
            
        revision = self.model_config.get("revision", "main")
        trust_remote_code = self.model_config.get("trust_remote_code", False)
        cache_dir = self.config["models"].get("repo_cache_dir", "data/model_cache")

        # Determine which model class to use
        model_class_name = self.model_config.get("model_class", "AutoModelForCausalLM")
        tokenizer_class_name = self.model_config.get("tokenizer_class", "AutoTokenizer")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )

        # Prepare quantization config if needed
        quantization_config = None
        if "quantization" in self.model_config:
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

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if self.config["models"]["precision"] == "fp16" else torch.float32
        )

        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(self, instruction: str) -> str:
        """Format instruction according to model's prompt template."""
        template = self.model_config.get("prompt_template", "{instruction}")
        return template.format(instruction=instruction)

    @torch.no_grad()
    def generate(self, instruction: str, **kwargs) -> str:
        """
        Generate a response to the given instruction.

        Args:
            instruction: The instruction to respond to.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Generated text response.
        """
        formatted_prompt = self.format_prompt(instruction)

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.temperature > 0,
            **kwargs
        }

        # Generate response
        outputs = self.model.generate(
            **inputs,
            **generation_config
        )

        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the response
        response = response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

        return response.strip()

    def generate_with_streaming(self, instruction: str, callback=None, **kwargs) -> str:
        """
        Generate a response with streaming capability.

        Args:
            instruction: The instruction to respond to.
            callback: Optional callback function to process streaming tokens.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Generated text response.
        """
        formatted_prompt = self.format_prompt(instruction)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.temperature > 0,
            **kwargs
        }

        # Stream tokens if callback provided
        if callback:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, callback=callback)
            generation_config["streamer"] = streamer

        # Generate response
        outputs = self.model.generate(
            **inputs,
            **generation_config
        )

        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the response
        response = response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]

        return response.strip()

    def get_logits(self, text: str) -> torch.Tensor:
        """Get the logits for the given text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
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
