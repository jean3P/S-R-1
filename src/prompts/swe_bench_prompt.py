# src/prompts/swe_bench_prompt.py

from typing import Dict, Any
from src.prompts.base_prompt import BasePrompt


class SWEBenchPrompt(BasePrompt):
    """Prompt templates for SWE-bench problem solving."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SWE-bench prompt.

        Args:
            config: Prompt configuration
        """
        super().__init__(config)

        # Default templates if not provided in config
        if "generation" not in self.templates:
            self.templates["generation"] = (
                "# GitHub Issue: {issue_id}\n\n"
                "{problem_statement}\n\n"
                "# Repository Information\n"
                "Repository: {repo}\n"
                "Base commit: {base_commit}\n\n"
                "# Task\n"
                "Your task is to create a patch that fixes this issue. The patch should be in the format of a git diff.\n"
                "Focus on creating a minimal change that addresses the issue while maintaining the code's integrity.\n\n"
                "Please provide a patch in git diff format."
            )

        if "reflection" not in self.templates:
            self.templates["reflection"] = (
                "# GitHub Issue: {issue_id}\n\n"
                "{problem_statement}\n\n"
                "# Your Previous Solution\n"
                "{solution}\n\n"
                "# Test Results\n"
                "{output}\n\n"
                "# Errors\n"
                "{errors}\n\n"
                "# Task\n"
                "Based on the test results above, please refine your solution. The patch should be in the format of a git diff.\n"
                "Focus on creating a minimal change that addresses the issue while maintaining the code's integrity.\n"
                "Make sure your solution passes all the required tests.\n\n"
                "Please provide your refined patch in git diff format."
            )

    def format_generation(self, prompt: str, task: Dict[str, Any]) -> str:
        """
        Format the initial SWE-bench prompt.

        Args:
            prompt: Input prompt
            task: Task details

        Returns:
            Formatted prompt
        """
        # Extract task information
        repo_info = task.get("repo_info", {})

        # Prepare variables for template
        variables = {
            "issue_id": task.get("name", ""),
            "problem_statement": prompt,
            "repo": repo_info.get("repo", ""),
            "base_commit": repo_info.get("base_commit", "")
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
            solution: Generated solution (patch)
            output: Test execution output
            errors: Test execution errors
            task: Task details

        Returns:
            Formatted reflection prompt
        """
        # Extract task information
        repo_info = task.get("repo_info", {})

        # Prepare variables for template
        variables = {
            "issue_id": task.get("name", ""),
            "problem_statement": original_prompt,
            "solution": solution,
            "output": output or "No test output",
            "errors": errors or "No errors",
            "repo": repo_info.get("repo", ""),
            "base_commit": repo_info.get("base_commit", "")
        }

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

