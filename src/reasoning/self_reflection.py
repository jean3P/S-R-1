# src/reasoning/self_reflection.py

import re
import logging
from typing import Dict, Any, Tuple, List
import time

logger = logging.getLogger(__name__)


class SelfReflection:
    """
    Implementation of Self-Reflection for improved solution quality.
    """

    def __init__(self, config, model):
        """
        Initialize Self-Reflection.

        Args:
            config: Configuration object.
            model: Language model instance.
        """
        self.config = config
        self.model = model

        # Try to find the prompt template from various sources
        self.prompt_template = None

        # Check in model-specific configs
        model_configs = config.get_model_config(model.model_name)
        if model_configs and "self_reflection_template" in model_configs:
            self.prompt_template = model_configs.get("self_reflection_template", "")

        # Check in reflection configs
        reflection_config = config["self_reflection"] if "self_reflection" in config.defaults else {}
        if not self.prompt_template and "prompt_template" in reflection_config:
            self.prompt_template = reflection_config.get("prompt_template", "")

        # Load from file if not found
        if not self.prompt_template:
            from pathlib import Path
            import yaml

            # Try to load from prompts config - using Path objects directly
            prompt_paths = [
                Path("configs/prompts/self_reflection.yaml"),
                Path("src/configs/prompts/self_reflection.yaml")
            ]

            for path in prompt_paths:
                if path.exists():
                    try:
                        with open(path, 'r') as f:
                            prompt_config = yaml.safe_load(f)
                            if prompt_config and "prompt_template" in prompt_config:
                                self.prompt_template = prompt_config["prompt_template"]
                                break
                    except Exception as e:
                        logger.warning(f"Error loading prompt template from {path}: {e}")

        # Default template if nothing else works
        if not self.prompt_template:
            logger.warning("No SelfReflection prompt template found. Using default template.")
            self.prompt_template = """
                You are an expert software engineer reviewing a solution to a GitHub issue. You need to analyze the solution, reflect on it, and provide an improved version.

                GITHUB ISSUE:
                {issue_description}

                RELEVANT CODEBASE CONTEXT:
                {codebase_context}

                INITIAL SOLUTION:
                {solution}

                TASK:
                1. First, under "REFLECTION:", analyze the strengths and weaknesses of the solution, focusing on correctness, efficiency, and maintainability.
                2. Then, under "REVISED SOLUTION:", provide an improved implementation that addresses any issues identified.

                Make sure to wrap any code in ```python code blocks```.

                Begin your analysis:

                REFLECTION:
                """

        # Enhanced template with patch validation feedback
        self.enhanced_prompt_template = """
            You are an expert software engineer reviewing a solution to a GitHub issue. You need to analyze the solution, reflect on it, and provide an improved version.

            GITHUB ISSUE:
            {issue_description}

            RELEVANT CODEBASE CONTEXT:
            {codebase_context}

            INITIAL SOLUTION:
            {solution}

            TASK:
            1. First, under "REFLECTION:", analyze the strengths and weaknesses of the solution, focusing on correctness, efficiency, and maintainability.
            2. Pay special attention to creating a proper Git-formatted patch that modifies the correct files with accurate line numbers.
            3. Then, under "REVISED SOLUTION:", provide an improved implementation that addresses any issues identified.

            Make sure to wrap any code in ```python code blocks``` and ensure patches follow Git diff format.

            Begin your analysis:

            REFLECTION:
        """
        
        # Template that incorporates test patches and hints
        self.swe_bench_template = """
            You are an expert software engineer reviewing a solution to a GitHub issue. You need to analyze the solution, reflect on it, and provide an improved version.

            GITHUB ISSUE:
            {issue_description}

            RELEVANT CODEBASE CONTEXT:
            {codebase_context}

            INITIAL SOLUTION:
            {solution}

            {test_patch_section}

            {hints_section}

            {validation_feedback_section}

            TASK:
            1. First, under "REFLECTION:", analyze the strengths and weaknesses of the solution, focusing on correctness, efficiency, and maintainability.
            2. Pay special attention to creating a proper Git-formatted patch that modifies the correct files with accurate line numbers.
            3. Ensure your solution passes the test cases if test patches are provided.
            4. Then, under "REVISED SOLUTION:", provide an improved implementation that addresses any issues identified.

            Make sure to wrap any code in ```python code blocks``` and ensure patches follow Git diff format.

            Begin your analysis:

            REFLECTION:
        """

        self.iterations = config["reasoning"].get("reflection_iterations", 3)
        logger.info(f"SelfReflection initialized with model {model.model_name}")

    def refine_solution(self, solution: str, issue_description: str, codebase_context: str) -> Dict[str, Any]:
        """
        Refine a solution through multiple iterations of self-reflection.

        Args:
            solution: Initial solution to refine.
            issue_description: Description of the GitHub issue.
            codebase_context: Context from the codebase related to the issue.

        Returns:
            Dictionary containing the reflection iterations and final solution.
        """
        logger.info(f"Refining solution using Self-Reflection with {self.model.model_name}")

        # Ensure we have a valid solution
        if not solution or solution.strip() == "":
            logger.warning("Empty solution provided for reflection")
            solution = "# No initial solution provided"

        reflections = []
        current_solution = solution

        # Extract test patch and hints if present in the context
        test_patch = ""
        hints = ""
        validation_feedback = ""
        
        # Check for test patch in context
        test_patch_match = re.search(r'TEST PATCH:\n(.*?)(?=\n\n|\Z)', codebase_context, re.DOTALL)
        if test_patch_match:
            test_patch = test_patch_match.group(1)
            
        # Check for hints in issue description
        hints_match = re.search(r'ADDITIONAL HINTS:\n(.*?)(?=\n\n|\Z)', issue_description, re.DOTALL)
        if hints_match:
            hints = hints_match.group(1)
            
        # Check for validation feedback
        validation_match = re.search(r'PATCH VALIDATION FEEDBACK:\n(.*?)(?=\n\n|\Z)', codebase_context, re.DOTALL)
        if validation_match:
            validation_feedback = validation_match.group(1)
        
        # Determine which template to use
        has_validation_feedback = bool(validation_feedback)
        has_test_patch = bool(test_patch)
        has_hints = bool(hints)
        
        if has_test_patch or has_hints or has_validation_feedback:
            # Use the SWE-bench specific template
            prompt_template = self.swe_bench_template
        elif has_validation_feedback:
            prompt_template = self.enhanced_prompt_template
        else:
            prompt_template = self.prompt_template

        for i in range(self.iterations):
            logger.info(f"Self-reflection iteration {i + 1}/{self.iterations}")

            # Start timing
            start_time = time.time()

            # Prepare template sections
            test_patch_section = f"TEST PATCH TO PASS:\n{test_patch}" if has_test_patch else ""
            hints_section = f"ADDITIONAL HINTS:\n{hints}" if has_hints else ""
            validation_feedback_section = f"VALIDATION FEEDBACK:\n{validation_feedback}" if has_validation_feedback else ""
            
            # Format the prompt
            if prompt_template == self.swe_bench_template:
                prompt = prompt_template.format(
                    solution=current_solution,
                    issue_description=issue_description,
                    codebase_context=codebase_context,
                    test_patch_section=test_patch_section,
                    hints_section=hints_section,
                    validation_feedback_section=validation_feedback_section
                )
            else:
                prompt = prompt_template.format(
                    solution=current_solution,
                    issue_description=issue_description,
                    codebase_context=codebase_context
                )

            # Generate reflection and revised solution
            logger.debug(f"Self-reflection prompt: {prompt[:500]}...")
            response = self.model.generate(prompt)
            logger.debug(f"Model response: {response[:500]}...")

            # Calculate time taken
            reflection_time = time.time() - start_time

            # Parse the reflection and revised solution
            reflection, revised_solution = self._parse_response(response)

            logger.info(
                f"Reflection parsed, length: {len(reflection)}, revised solution length: {len(revised_solution)}"
            )

            reflections.append({
                "iteration": i + 1,
                "reflection": reflection,
                "solution": revised_solution,
                "reflection_time": reflection_time,
                "used_test_patch": has_test_patch,
                "used_hints": has_hints,
                "used_validation_feedback": has_validation_feedback
            })

            # Update the current solution for the next iteration
            current_solution = revised_solution

        return {
            "reflections": reflections,
            "final_solution": current_solution,
            "used_test_patch": has_test_patch,
            "used_hints": has_hints,
            "used_validation_feedback": has_validation_feedback
        }

    def _parse_response(self, response: str) -> tuple:
        """Parse the reflection and revised solution from the response."""
        if not response or response.strip() == "":
            logger.warning("Received empty response from model")
            return "No reflection provided", "No solution provided by model."

        # Common section markers
        reflection_markers = ["REFLECTION:", "ANALYSIS:", "EVALUATION:"]
        solution_markers = ["REVISED SOLUTION:", "IMPROVED SOLUTION:", "SOLUTION:", "IMPLEMENTATION:"]

        # Try finding the sections using markers
        reflection = ""
        revised_solution = ""

        # First check for standard format with explicit markers
        for r_marker in reflection_markers:
            if r_marker in response:
                parts = response.split(r_marker, 1)
                if len(parts) > 1:
                    reflection_part = parts[1]

                    # Try to find where the solution section starts
                    for s_marker in solution_markers:
                        if s_marker in reflection_part:
                            reflection_solution_parts = reflection_part.split(s_marker, 1)
                            reflection = reflection_solution_parts[0].strip()
                            if len(reflection_solution_parts) > 1:
                                revised_solution = reflection_solution_parts[1].strip()
                            break

                    # If we found the reflection but no solution marker, use the rest as reflection
                    if not revised_solution:
                        reflection = reflection_part.strip()

                    break

        # If we didn't find reflection markers, try just looking for solution markers
        if not reflection:
            for s_marker in solution_markers:
                if s_marker in response:
                    parts = response.split(s_marker, 1)
                    reflection = parts[0].strip()  # Everything before the solution marker
                    if len(parts) > 1:
                        revised_solution = parts[1].strip()
                    break
        # If no structured format detected, use heuristics
        if not reflection and not revised_solution:
            # Look for code blocks which often contain solutions
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)

            if code_blocks:
                # Use the text before first code block as reflection, and the code as solution
                code_start = response.find("```")
                if code_start > 0:
                    reflection = response[:code_start].strip()
                    revised_solution = "\n".join(code_blocks)
            else:
                # Split the response roughly in half if nothing else works
                lines = response.split("\n")
                mid_point = len(lines) // 2

                reflection = "\n".join(lines[:mid_point]).strip()
                revised_solution = "\n".join(lines[mid_point:]).strip()

        # Ensure we have something for both sections
        if not reflection:
            reflection = "No explicit reflection provided"

        if not revised_solution:
            # If we have no solution but have code blocks, use those
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
            if code_blocks:
                revised_solution = "\n".join(code_blocks)
            else:
                revised_solution = "No explicit solution provided"

        return reflection, revised_solution
