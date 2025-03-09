# src/prompts/__init__.py
"""
Prompt templates for the AI system.

This package contains different types of prompt templates that can be used
to format inputs for language models. The main prompt types are:

- CodeGenerationPrompt: Templates for code generation and refinement
- ReasoningPrompt: Templates for multi-step reasoning

New prompt types can be added by implementing the BasePrompt interface and
registering them with the prompt registry.
"""

from src.prompts.base_prompt import BasePrompt
from src.prompts.code_generation import CodeGenerationPrompt
from src.prompts.reasoning import ReasoningPrompt
from src.prompts.template_utils import (
    TemplateLoader,
    format_string_template,
    extract_template_variables,
    process_template_includes,
    apply_template_filters
)
from src.prompts.registry import (
    register_prompt,
    get_prompt,
    get_prompt_class,
    list_available_prompts,
    clear_prompt_cache,
    get_prompt_configs
)

# Version of the prompts package
__version__ = "0.1.0"

__all__ = [
    'BasePrompt',
    'CodeGenerationPrompt',
    'ReasoningPrompt',
    'TemplateLoader',
    'format_string_template',
    'extract_template_variables',
    'process_template_includes',
    'apply_template_filters',
    'register_prompt',
    'get_prompt',
    'get_prompt_class',
    'list_available_prompts',
    'clear_prompt_cache',
    'get_prompt_configs'
]