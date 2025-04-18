
from src.prompts.base_prompt import BasePrompt
from src.prompts.swe_bench_prompt import SWEBenchPrompt  # Import the SWE-bench prompt

from src.prompts.registry import (
    register_prompt,
    get_prompt,
    get_prompt_class,
    list_available_prompts,
    clear_prompt_cache,
    get_prompt_configs
)

# Register the prompts
register_prompt("swe_bench_prompt", SWEBenchPrompt)  # Register the SWE-bench prompt

# Version of the prompts package
__version__ = "0.1.0"
__all__ = [
    'BasePrompt',
    'SWEBenchPrompt',
    'register_prompt',
    'get_prompt',
    'get_prompt_class',
    'list_available_prompts',
    'clear_prompt_cache',
    'get_prompt_configs'
]
