# src/reasoning/chain_of_thought.py

import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ChainOfThought:
    """
    Implementation of Chain of Thought reasoning for solving GitHub issues.
    """

    def __init__(self, config, model):
        """
        Initialize Tree of Thought reasoning.

        Args:
            config: Configuration object.
            model: Language model instance.
        """
        self.config = config
        self.model = model

        # Load ToT-specific configurations
        model_configs = config.get_model_config(model.model_name)

        # Try to find the prompt template from various sources
        self.prompt_template = None

        # Check in model configs
        if "tree_of_thought" in config.model_configs:
            self.tot_config = config.model_configs["tree_of_thought"]
            if "prompt_template" in self.tot_config:
                self.prompt_template = self.tot_config.get("prompt_template", "")
        else:
            self.tot_config = {}

        # If not found, load from file
        if not self.prompt_template:
            from pathlib import Path
            import yaml

            # Try to load from prompts config
            prompt_paths = [
                Path("configs/prompts/tree_of_thought.yaml"),
                Path("src/configs/prompts/tree_of_thought.yaml")
            ]

            for path in prompt_paths:
                if path.exists():
                    try:
                        with open(path, 'r') as f:
                            prompt_config = yaml.safe_load(f)
                            if prompt_config and "prompt_template" in prompt_config["tree_of_thought"]:
                                self.prompt_template = prompt_config["tree_of_thought"]["prompt_template"]
                                break
                    except Exception as e:
                        logger.warning(f"Error loading prompt template from {path}: {e}")

        # Default template if nothing else works
        if not self.prompt_template:
            logger.warning("No TreeOfThought prompt template found. Using default template.")
            self.prompt_template = """
                You are an expert software engineer tasked with fixing a GitHub issue. You'll explore multiple approaches before determining the best solution.

                GITHUB ISSUE:
                {issue_description}

                RELEVANT CODEBASE CONTEXT:
                {codebase_context}

                Please solve this issue using a tree-of-thought approach:

                First, explore 3 different ways of understanding the problem, labeled as BRANCH 1, BRANCH 2, and BRANCH 3.

                Then, for the most promising branch, propose 3 different solution approaches, labeled as SOLUTION 1, SOLUTION 2, and SOLUTION 3.

                Finally, provide a detailed implementation of the best solution under "IMPLEMENTATION:".

                Make sure to wrap any code in ```python code blocks```.

                Begin your analysis now:
                """

        # ToT parameters
        self.tot_breadth = config["reasoning"].get("tot_breadth", 3)
        self.tot_depth = config["reasoning"].get("tot_depth", 3)

    def solve(self, issue_description: str, codebase_context: str) -> Dict[str, Any]:
        """
        Solve a GitHub issue using Chain of Thought reasoning.

        Args:
            issue_description: Description of the GitHub issue.
            codebase_context: Context from the codebase related to the issue.

        Returns:
            Dictionary containing the solution steps and the final solution.
        """
        logger.info(f"Solving issue using Chain of Thought reasoning with {self.model.model_name}")

        # Format the prompt with issue description
        instruction = self._format_instruction(issue_description, codebase_context)

        # Generate the solution
        logger.debug(f"Chain of Thought prompt: {instruction[:500]}...")
        response = self.model.generate(instruction)
        logger.debug(f"Model response: {response[:500]}...")

        # Parse the solution steps and final solution
        solution_data = self._parse_response(response)
        logger.info(
            f"Solution parsed, steps: {len(solution_data.get('steps', []))}, solution length: {len(solution_data.get('solution', ''))}")

        return solution_data

    def _format_instruction(self, issue_description: str, codebase_context: str) -> str:
        """Format the instruction for the model."""
        # Format using the CoT template
        instruction = self.prompt_template.format(
            issue_description=issue_description,
            codebase_context=codebase_context
        )
        return instruction

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into steps and solution."""
        if not response or response.strip() == "":
            logger.warning("Received empty response from model")
            return {
                "steps": [],
                "solution": "No solution provided by model."
            }

        # Try to identify sections in the response
        solution_markers = ["SOLUTION:", "IMPLEMENTATION:", "FINAL SOLUTION:", "CODE:"]
        steps = []
        solution = ""

        # First check if we have a structured response with steps and solution
        lines = response.split('\n')
        current_step = None
        current_content = []
        found_solution = False

        for line in lines:
            line_stripped = line.strip()

            # Check for step markers like "1.", "Step 1:", etc.
            if re.match(r"^(?:Step\s*)?[1-9][0-9]?\.?\s", line_stripped):
                # Save previous step if exists
                if current_step is not None:
                    steps.append({
                        "name": current_step,
                        "content": "\n".join(current_content)
                    })
                # Start new step
                current_step = line_stripped
                current_content = []

            # Check for solution markers
            elif any(marker in line_stripped for marker in solution_markers):
                # Save previous step if exists
                if current_step is not None:
                    steps.append({
                        "name": current_step,
                        "content": "\n".join(current_content)
                    })
                # Start collecting the solution
                solution = line + "\n"
                current_step = None
                current_content = []
                found_solution = True

            # Continue collecting content for current section
            elif current_step is not None:
                current_content.append(line)
            elif found_solution:
                solution += line + "\n"

        # Add the last step if it exists
        if current_step is not None:
            steps.append({
                "name": current_step,
                "content": "\n".join(current_content)
            })

        # If no structured format was detected, use heuristics to extract solution
        if not steps and not found_solution:
            # Look for code blocks which often contain solutions
            code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
            if code_blocks:
                solution = "\n".join(code_blocks)
            else:
                # Use the whole response as the solution
                solution = response

        return {
            "steps": steps,
            "solution": solution.strip()
        }
