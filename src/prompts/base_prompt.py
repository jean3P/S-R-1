# src/prompts/base_prompt.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BasePrompt(ABC):
    """Abstract base class for all prompt templates."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the prompt template.

        Args:
            config: Prompt configuration
        """
        from src.utils.logging import get_logger

        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.templates = config.get("templates", {})
        self.system_message = config.get("system_message", "")

    @abstractmethod
    def format_generation(self, prompt: str, context: Dict[str, Any] = None, task: Dict[str, Any] = None) -> str:
        """
        Format the generation prompt.

        Args:
            prompt: Input prompt
            context: Additional context (optional)
            task: Task details (optional)

        Returns:
            Formatted prompt
        """
        pass

    @abstractmethod
    def format_reflection(
            self,
            original_prompt: str,
            solution: str,
            output: str,
            errors: str,
            context: Dict[str, Any] = None,
            task: Dict[str, Any] = None
    ) -> str:
        """
        Format the reflection prompt.

        Args:
            original_prompt: Original prompt
            solution: Generated solution
            output: Execution output
            errors: Execution errors
            context: Additional context (optional)
            task: Task details (optional)

        Returns:
            Formatted reflection prompt
        """
        pass

    def _merge_variables(self, template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge template variables with defaults.

        Args:
            template_vars: Template variables

        Returns:
            Merged variables
        """
        # Start with default variables
        default_vars = self.config.get("default_variables", {})

        # Merge with provided variables (overriding defaults)
        variables = {**default_vars, **template_vars}

        return variables

    def _validate_template(self, template: str, variables: Dict[str, Any]) -> List[str]:
        """
        Validate template against provided variables.

        Args:
            template: Template string
            variables: Template variables

        Returns:
            List of missing variables (empty if valid)
        """
        import re

        # Find all placeholders in the template
        placeholders = re.findall(r'{([^{}]*)}', template)

        # Check if all placeholders have corresponding variables
        missing = []
        for placeholder in placeholders:
            # Handle formatting specifiers like {var:02d}
            var_name = placeholder.split(':', 1)[0]
            if var_name not in variables:
                missing.append(var_name)

        return missing

    def _format_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Format a template string with variables.

        Args:
            template: Template string
            variables: Template variables

        Returns:
            Formatted string
        """
        # Validate the template
        missing = self._validate_template(template, variables)
        if missing:
            missing_str = ", ".join(missing)
            self.logger.warning(f"Missing template variables: {missing_str}")

            # Provide placeholders for missing variables
            for var in missing:
                variables[var] = f"[{var}]"

        try:
            return template.format(**variables)
        except Exception as e:
            self.logger.error(f"Error formatting template: {str(e)}")
            # Return template with error message
            return template + f"\n\nError formatting template: {str(e)}"

    @abstractmethod
    def format_tot_reasoning(self, original_prompt: str, parent_reasoning: str,
                             depth: int, strategy: str, task: Dict[str, Any]) -> str:
        """
        Format a prompt for Tree of Thought reasoning.

        Args:
            original_prompt: Original problem statement
            parent_reasoning: Reasoning from parent node
            depth: Current reasoning depth
            strategy: Reasoning strategy to focus on
            task: Task details

        Returns:
            Formatted ToT reasoning prompt
        """
        pass
