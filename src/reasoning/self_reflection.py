# src/reasoning/self_reflection.py

import re
import logging
import time
from typing import Dict, Any
import torch

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
        self.current_issue = None

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

        # Template that incorporates validation feedback
        self.validation_prompt_template = """
            You are an expert software engineer reviewing a solution to a GitHub issue. You need to analyze the solution, fix the patch formatting issues, and provide an improved implementation.

            GITHUB ISSUE:
            {issue_description}

            RELEVANT CODEBASE CONTEXT:
            {codebase_context}

            CURRENT SOLUTION:
            {solution}

            PATCH VALIDATION FEEDBACK:
            {validation_feedback}

            TASK:
            1. First, under "REFLECTION:", analyze why the patch is invalid based on the validation feedback.
            2. Focus on fixing the Git patch formatting issues.
            3. Make sure the file paths, line numbers, and context match the actual files.
            4. Then, under "REVISED SOLUTION:", provide a correctly formatted patch that will apply cleanly.

            Ensure your patch follows the proper Git diff format:
            ```
            diff --git a/path/to/file b/path/to/file
            --- a/path/to/file
            +++ b/path/to/file
            @@ -start,count +start,count @@
             context line
            -removed line
            +added line
             context line
            ```

            Begin your analysis:

            REFLECTION:
        """

        self.iterations = config["reasoning"].get("reflection_iterations", 3)
        logger.info(f"SelfReflection initialized with model {model.model_name}")

        # Initialize patch formatter and validator if needed
        self.patch_formatter = None
        self.patch_validator = None
        self.initialize_patch_tools()

    def initialize_patch_tools(self):
        """Initialize patch formatting and validation tools if available."""
        try:
            from ..utils.enhanced_patch_formatter import EnhancedPatchFormatter
            from ..utils.patch_validator import PatchValidator

            self.patch_formatter = EnhancedPatchFormatter(self.config)
            self.patch_validator = PatchValidator(self.config)
            logger.info("Patch formatting and validation tools initialized")
        except ImportError as e:
            logger.warning(f"Could not initialize patch tools: {e}")

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

        # Extract repo name and issue ID for patch validation
        repo_name = self._extract_repo_from_context(codebase_context)
        if hasattr(self, 'current_issue') and self.current_issue:
            issue_id = self.current_issue.get("instance_id") or self.current_issue.get("id")
        else:
            # Fallback to extraction if issue object not available
            issue_id = self._extract_issue_id_from_context(issue_description)

        # Initial patch extraction and validation
        has_validation_feedback = False
        validation_feedback = ""

        # Extract and validate initial patch if tools are available
        if self.patch_formatter and self.patch_validator:
            initial_patch = self._extract_patch(current_solution)
            if initial_patch:
                formatted_initial_patch = self.patch_formatter.format_patch(initial_patch, repo_name)
                initial_validation = self.patch_validator.validate_patch(formatted_initial_patch, issue_id)
                validation_feedback = initial_validation.get("feedback", "")
                has_validation_feedback = True
                logger.info(f"Initial patch validation: success={initial_validation.get('success', False)}")

        for i in range(self.iterations):
            logger.info(f"Self-reflection iteration {i + 1}/{self.iterations}")

            # Start timing
            start_time = time.time()

            # Select appropriate template based on validation feedback
            if has_validation_feedback:
                prompt = self.validation_prompt_template.format(
                    solution=current_solution,
                    issue_description=issue_description,
                    codebase_context=codebase_context,
                    validation_feedback=validation_feedback
                )
            else:
                prompt = self.prompt_template.format(
                    solution=current_solution,
                    issue_description=issue_description,
                    codebase_context=codebase_context
                )

            # Generate reflection and revised solution
            logger.debug(f"Self-reflection prompt: {prompt[:500]}...")

            # Clear memory before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            response = self.model.generate(prompt)
            logger.debug(f"Model response: {response[:500]}...")

            # Calculate time taken
            reflection_time = time.time() - start_time

            # Parse the reflection and revised solution
            reflection, revised_solution = self._parse_response(response)

            logger.info(
                f"Reflection parsed, length: {len(reflection)}, revised solution length: {len(revised_solution)}"
            )

            # Extract and validate patch from revised solution
            patch_info = {}
            if self.patch_formatter and self.patch_validator:
                # Extract patch
                patch = self._extract_patch(revised_solution)

                if patch:
                    # Format patch
                    formatted_patch = self.patch_formatter.format_patch(patch, repo_name)

                    if self.current_issue:
                        issue_id = self.current_issue.get("instance_id") or self.current_issue.get("id")
                        logger.debug(f"Using issue ID from issue object: {issue_id}")
                    else:
                        issue_id = self._extract_issue_id_from_context(issue_description)
                        logger.debug(f"Using extracted issue ID: {issue_id}")

                    # Validate patch
                    validation_result = self.patch_validator.validate_patch(formatted_patch, issue_id)

                    validation_feedback = validation_result.get("feedback", "")
                    has_validation_feedback = True

                    # Store patch info
                    patch_info = {
                        "raw_patch": patch,
                        "formatted_patch": formatted_patch,
                        "validation": validation_result,
                        "success": validation_result.get("success", False)
                    }

                    logger.info(
                        f"Patch validation (iteration {i + 1}): success={validation_result.get('success', False)}")
                else:
                    validation_feedback = "No valid patch found in the solution. Please provide a properly formatted Git patch."
                    has_validation_feedback = True
                    patch_info = {
                        "error": "No valid patch extracted"
                    }

            # Store reflection data
            reflection_data = {
                "iteration": i + 1,
                "reflection": reflection,
                "solution": revised_solution,
                "reflection_time": reflection_time
            }

            # Add patch info if available
            if patch_info:
                reflection_data.update(patch_info)

            reflections.append(reflection_data)

            # Update the current solution for the next iteration
            current_solution = revised_solution

            # Check if we've found a valid patch and can stop early
            if patch_info.get("success", False):
                logger.info(f"Valid patch found in iteration {i + 1}. Stopping reflections early.")
                break

            # Clear memory after iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Find the best iteration (one with successful validation or last one)
        best_iteration = None
        for r in reflections:
            if r.get("success", False):
                best_iteration = r
                break

        if not best_iteration and reflections:
            best_iteration = reflections[-1]

        final_solution = best_iteration["solution"] if best_iteration else current_solution

        return {
            "reflections": reflections,
            "final_solution": final_solution,
            "success": best_iteration.get("success", False) if best_iteration else False,
            "formatted_patch": best_iteration.get("formatted_patch", "") if best_iteration else ""
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

    def _extract_patch(self, solution: str) -> str:
        """Extract a Git patch from the solution text."""
        # Look for a git diff format
        diff_pattern = r'(diff --git.*?)(?:\Z|(?=^```|\n\n\n))'
        diff_match = re.search(diff_pattern, solution, re.MULTILINE | re.DOTALL)

        if diff_match:
            return diff_match.group(1).strip()

        # Look for content inside code blocks that might contain patches
        code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
        code_match = re.search(code_block_pattern, solution, re.MULTILINE | re.DOTALL)

        if code_match:
            content = code_match.group(1).strip()
            if content.startswith('diff --git') or ('---' in content and '+++' in content):
                return content

        # No valid patch found
        return ""

    def _extract_repo_from_context(self, context: str) -> str:
        """Extract repository name from context."""
        repo_match = re.search(r'Repository:\s*([a-zA-Z0-9_\-/.]+)', context, re.IGNORECASE)
        if repo_match:
            return repo_match.group(1)

        # Try to find a repo-like path
        path_match = re.search(r'([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)\.git', context)
        if path_match:
            return f"{path_match.group(1)}/{path_match.group(2)}"

        # Try to extract from file paths
        file_paths = re.findall(r'[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-/]+\.[a-zA-Z0-9]+', context)
        if file_paths:
            # Get the first component of the first file path
            parts = file_paths[0].split('/')
            if len(parts) > 1:
                return parts[0]

        return "unknown_repo"

    def _extract_issue_id_from_context(self, description: str) -> str:
        """Extract issue ID from description."""
        # If we have current_issue, use its ID
        if hasattr(self, 'current_issue') and self.current_issue:
            issue_id = self.current_issue.get("instance_id") or self.current_issue.get("id")
            if issue_id:
                return issue_id

        # More specific pattern for SWE-bench format: repo__issue
        swebench_match = re.search(r'([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)__(\d+)', description)
        if swebench_match:
            return f"{swebench_match.group(1)}/{swebench_match.group(2)}__{swebench_match.group(3)}"

        # Look for common issue ID patterns
        issue_match = re.search(r'(?:issue|#)[\s:]*([a-zA-Z0-9_\-/.]+__\d+)', description, re.IGNORECASE)
        if issue_match:
            return issue_match.group(1)

        # Try to extract GitHub issue format
        github_issue = re.search(r'([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+)#(\d+)', description)
        if github_issue:
            return f"{github_issue.group(1)}__{github_issue.group(2)}"

        # If no match, log a warning and return None - don't generate a placeholder
        logger.warning(f"Could not extract valid issue ID from description")
        return None
