# src/prompts/reasoning.py

from typing import Dict, Any, List
from src.prompts.base_prompt import BasePrompt


class ReasoningPrompt(BasePrompt):
    """Prompt templates for multi-step reasoning."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reasoning prompt.

        Args:
            config: Prompt configuration
        """
        super().__init__(config)

        # Default templates if not provided in config
        if "generation" not in self.templates:
            self.templates["generation"] = (
                "# Problem: {prompt}\n\n"
                "Think through this problem step-by-step to find the solution."
            )

        if "reasoning" not in self.templates:
            self.templates["reasoning"] = (
                "# Problem: {current_prompt}\n\n"
                "# Reasoning so far:\n{reasoning_history}\n\n"
                "# Current understanding: {current_understanding}\n\n"
                "Continue the reasoning process. What's the next step in solving this problem?"
            )

        if "solution" not in self.templates:
            self.templates["solution"] = (
                "# Problem: {initial_prompt}\n\n"
                "# Complete reasoning:\n{reasoning_history}\n\n"
                "Based on the reasoning above, provide the final solution to the problem."
            )

    def format_generation(self, prompt: str, task: Dict[str, Any]) -> str:
        """
        Format the initial reasoning prompt.

        Args:
            prompt: Input prompt
            task: Task details

        Returns:
            Formatted prompt
        """
        # Prepare variables for template
        variables = {
            "prompt": prompt,
            "problem_type": task.get("problem_type", "general"),
            "complexity": task.get("complexity", "medium"),
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
        Note: This is used differently for reasoning - it's more for compatibility
        with the BasePrompt interface. Use format_reasoning() instead.

        Args:
            original_prompt: Original prompt
            solution: Generated solution
            output: Execution output
            errors: Execution errors
            task: Task details

        Returns:
            Formatted reflection prompt
        """
        return self.format_reasoning(
            current_prompt=original_prompt,
            reasoning_state={
                "current_understanding": solution,
                "facts": [],
                "hypotheses": [],
            },
            reasoning_strategy=task.get("reasoning_strategy", "cot"),
            step=1,
            task=task
        )

    def format_reasoning(
            self,
            current_prompt: str,
            reasoning_state: Dict[str, Any],
            reasoning_strategy: str,
            step: int,
            task: Dict[str, Any]
    ) -> str:
        """
        Format the prompt for the next reasoning step.

        Args:
            current_prompt: Current problem description
            reasoning_state: Current reasoning state
            reasoning_strategy: Reasoning strategy (e.g., "cot", "tot")
            step: Current reasoning step
            task: Task details

        Returns:
            Formatted reasoning prompt
        """
        # Prepare variables for template
        variables = {
            "current_prompt": current_prompt,
            "current_understanding": reasoning_state.get("current_understanding", ""),
            "reasoning_strategy": reasoning_strategy,
            "step": step,
        }

        # Format reasoning history if available
        reasoning_history = ""
        if "reasoning_history" in reasoning_state:
            reasoning_history = reasoning_state["reasoning_history"]
        else:
            # Build from individual steps
            steps = reasoning_state.get("steps", [])
            for i, step_info in enumerate(steps, 1):
                reasoning_history += f"Step {i}: {step_info.get('thought', '')}\n\n"

        variables["reasoning_history"] = reasoning_history

        # Add facts and hypotheses if available
        facts = reasoning_state.get("facts", [])
        if facts:
            variables["facts"] = "\n".join([f"- {fact}" for fact in facts])
        else:
            variables["facts"] = "No facts discovered yet."

        hypotheses = reasoning_state.get("hypotheses", [])
        if hypotheses:
            variables["hypotheses"] = "\n".join([f"- {hyp}" for hyp in hypotheses])
        else:
            variables["hypotheses"] = "No hypotheses formed yet."

        # Add progress indicators if available
        variables["solution_progress"] = f"{reasoning_state.get('solution_progress', 0.0) * 100:.1f}%"
        variables["confidence"] = f"{reasoning_state.get('confidence', 0.0) * 100:.1f}%"

        # Add any additional variables from the task
        for key, value in task.items():
            if key not in variables and isinstance(value, (str, int, float, bool)):
                variables[key] = value

        # Merge with default variables
        variables = self._merge_variables(variables)

        # Get the appropriate template based on strategy
        template_key = f"reasoning_{reasoning_strategy}" if f"reasoning_{reasoning_strategy}" in self.templates else "reasoning"
        template = self.templates[template_key]

        # Format the template
        formatted_prompt = self._format_template(template, variables)

        # Add system message if available and not already included
        if self.system_message and "{system_message}" not in template:
            formatted_prompt = f"{self.system_message}\n\n{formatted_prompt}"

        return formatted_prompt

    def format_solution(
            self,
            initial_prompt: str,
            reasoning_steps: List[Dict[str, Any]],
            reasoning_state: Dict[str, Any],
            task: Dict[str, Any]
    ) -> str:
        """
        Format the prompt for generating the final solution.

        Args:
            initial_prompt: Initial problem description
            reasoning_steps: All reasoning steps
            reasoning_state: Current reasoning state
            task: Task details

        Returns:
            Formatted solution prompt
        """
        # Format reasoning history
        reasoning_history = ""
        for i, step in enumerate(reasoning_steps, 1):
            reasoning_history += f"Step {i}: {step.get('parsed_output', {}).get('thoughts', '')}\n\n"

        # Prepare variables for template
        variables = {
            "initial_prompt": initial_prompt,
            "reasoning_history": reasoning_history,
            "current_understanding": reasoning_state.get("current_understanding", ""),
            "facts": "\n".join([f"- {fact}" for fact in reasoning_state.get("facts", [])]),
            "hypotheses": "\n".join([f"- {hyp}" for hyp in reasoning_state.get("hypotheses", [])]),
            "solution_progress": f"{reasoning_state.get('solution_progress', 0.0) * 100:.1f}%",
            "confidence": f"{reasoning_state.get('confidence', 0.0) * 100:.1f}%",
        }

        # Add any additional variables from the task
        for key, value in task.items():
            if key not in variables and isinstance(value, (str, int, float, bool)):
                variables[key] = value

        # Merge with default variables
        variables = self._merge_variables(variables)

        # Get the template
        template = self.templates["solution"]

        # Format the template
        formatted_prompt = self._format_template(template, variables)

        # Add system message if available and not already included
        if self.system_message and "{system_message}" not in template:
            formatted_prompt = f"{self.system_message}\n\n{formatted_prompt}"

        return formatted_prompt
