# src/prompts/code_generation.py

from typing import Dict, Any
from src.prompts.base_prompt import BasePrompt


class CodeGenerationPrompt(BasePrompt):
    """Prompt templates for code generation and refinement."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the code generation prompt.

        Args:
            config: Prompt configuration
        """
        super().__init__(config)

        # Default templates if not provided in config
        if "generation" not in self.templates:
            self.templates["generation"] = (
                "# TASK: {prompt}\n"
                "# LANGUAGE: {language}\n"
                "{constraints}\n"
                "{examples}\n"
                "# Please only focus on providing the solution:"
            )

        if "reflection" not in self.templates:
            self.templates["reflection"] = (
                "{solution}\n\n"
                "# Execution Output:\n{output}\n"
                "# Execution Errors:\n{errors}\n\n"
                "# Based on the above, please refine the solution. Focus on:\n"
                "# 1. Fixing any errors\n"
                "# 2. Improving efficiency\n"
                "# 3. Enhancing readability\n"
                "# Provide the complete refined solution:"
            )

    def format_generation(self, prompt: str, task: Dict[str, Any]) -> str:
        """
        Format the code generation prompt.

        Args:
            prompt: Input prompt
            task: Task details

        Returns:
            Formatted prompt
        """
        # Use task details to enhance the prompt
        language = task.get("language", "python")
        constraints = task.get("constraints", [])
        examples = task.get("examples", [])

        # Format constraints as a string
        constraints_str = ""
        if constraints:
            constraints_str = "# Constraints:\n"
            for i, constraint in enumerate(constraints, 1):
                constraints_str += f"# {i}. {constraint}\n"

        # Format examples as a string
        examples_str = ""
        if examples:
            examples_str = "# Examples:\n"
            for i, example in enumerate(examples, 1):
                examples_str += f"# Example {i}:\n"
                examples_str += f"# Input: {example.get('input', '')}\n"
                examples_str += f"# Output: {example.get('output', '')}\n"

        # Prepare variables for template
        variables = {
            "prompt": prompt,
            "language": language,
            "constraints": constraints_str,
            "examples": examples_str
        }

        # Add any additional variables from the task
        for key, value in task.items():
            if key not in variables and isinstance(value, (str, int, float, bool)):
                variables[key] = value

        # Merge with default variables
        variables = self._merge_variables(variables)

        # Get the template
        template = self.templates["generation"]

        # Format the template
        formatted_prompt = self._format_template(template, variables)

        # Add system message if available and not already included
        if self.system_message and "{system_message}" not in template:
            formatted_prompt = f"{self.system_message}\n\n{formatted_prompt}"

        return formatted_prompt

    def format_reflection(
            self,
            original_prompt: str,
            solution: str,
            output: str,
            errors: str,
            task: Dict[str, Any]
    ) -> str:
        """
        Format the reflection prompt.

        Args:
            original_prompt: Original prompt
            solution: Generated solution
            output: Execution output
            errors: Execution errors
            task: Task details

        Returns:
            Formatted reflection prompt
        """
        # Prepare variables for template
        variables = {
            "original_prompt": original_prompt,
            "solution": solution,
            "output": output or "No output",
            "errors": errors or "No errors",
        }

        # Add task-specific refinement instructions
        if "refinement_instructions" in task:
            variables["refinement_instructions"] = task["refinement_instructions"]
        else:
            variables["refinement_instructions"] = (
                "1. Fixing any errors\n"
                "2. Improving efficiency\n"
                "3. Enhancing readability"
            )

        # Add language information
        variables["language"] = task.get("language", "python")

        # Add any additional variables from the task
        for key, value in task.items():
            if key not in variables and isinstance(value, (str, int, float, bool)):
                variables[key] = value

        # Merge with default variables
        variables = self._merge_variables(variables)

        # Get the template
        template = self.templates["reflection"]

        # Format the template
        formatted_prompt = self._format_template(template, variables)

        # Add system message if available and not already included
        if self.system_message and "{system_message}" not in template:
            formatted_prompt = f"{self.system_message}\n\n{formatted_prompt}"

        return formatted_prompt
