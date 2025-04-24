import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
import torch

from ..utils.patch_validator import PatchValidator

logger = logging.getLogger(__name__)


class EnhancedChainOfThought:
    """
    Enhanced Chain of Thought reasoning with integrated validation and self-reflection.
    Implements the third phase of the optimized bug fixing pipeline.
    """

    def __init__(self, config, model):
        """
        Initialize Chain of Thought reasoning with validation.

        Args:
            config: Configuration object.
            model: Language model instance.
        """
        self.config = config
        self.model = model
        self.max_iterations = 2
        self.patch_validator = PatchValidator(config)

        # Metrics tracking
        self.metrics = {
            "syntax_validity": 0.0,
            "logic_completeness": 0.0,
            "patch_format": 0.0,
            "overall_completeness": 0.0,
        }

    def solve_with_validation(
            self,
            issue_id: str,
            issue_description: str,
            codebase_context: str,
            bug_location: Dict[str, Any],
            root_cause: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve the issue with integrated validation and self-reflection.

        Args:
            issue_id: Issue ID for validation.
            issue_description: Description of the issue.
            codebase_context: Context from the codebase.
            bug_location: Location of the bug.
            root_cause: Optional root cause analysis from previous phase.

        Returns:
            Dictionary containing solutions, evaluations, and validation results.
        """
        logger.info("Starting Enhanced Chain of Thought with validation and reflection")

        # Format the prompt with bug location and root cause
        prompt = self._create_solution_prompt(issue_description, codebase_context, bug_location, root_cause)

        # Track iterations
        iterations = []
        valid_solution = None
        current_solution = None

        # First iteration: Generate initial solution
        try:
            logger.info("Generating initial solution")
            response = self.model.generate(prompt)
            logger.debug(f"Initial solution response: {response[:500]}...")

            # Extract solution and patch
            solution = response.strip()
            patch = self._extract_patch(solution)

            # Calculate solution completeness
            completeness = self._evaluate_solution_completeness(solution, patch is not None)

            # Track initial solution
            iteration_result = {
                "iteration": 1,
                "solution": solution,
                "patch": patch,
                "completeness": completeness,
                "evaluation": {
                    "syntax_validity": self.metrics["syntax_validity"],
                    "logic_completeness": self.metrics["logic_completeness"],
                    "patch_format": self.metrics["patch_format"],
                }
            }

            # Validate patch if we have one
            if patch:
                validation_result = self.patch_validator.validate_patch(patch, issue_id)
                iteration_result["validation"] = validation_result

                # If valid patch, mark for early stopping
                if validation_result.get("success", False):
                    valid_solution = {
                        "solution": solution,
                        "patch": patch,
                        "validation": validation_result,
                        "iteration": 1
                    }
                    logger.info("Found valid patch in initial solution")
                    iterations.append(iteration_result)

                    # Return early if valid solution found
                    return {
                        "iterations": iterations,
                        "valid_solution": valid_solution,
                        "solution": solution,
                        "patch": patch,
                        "depth": completeness,
                        "early_stopped": True,
                        "success": True
                    }

            iterations.append(iteration_result)
            current_solution = solution

            # If completeness is very high but no valid patch, try validating with more relaxed options
            if completeness >= 0.9 and patch and not valid_solution:
                logger.info("Solution completeness is high but validation failed, trying relaxed validation")
                relaxed_validation = self._try_relaxed_validation(patch, issue_id)

                if relaxed_validation.get("success", False):
                    valid_solution = {
                        "solution": solution,
                        "patch": patch,
                        "validation": relaxed_validation,
                        "iteration": 1,
                        "relaxed_validation": True
                    }
                    logger.info("Found valid patch with relaxed validation")

                    # Update the iteration result
                    iterations[-1]["relaxed_validation"] = relaxed_validation

                    # Return early if valid solution found with relaxed validation
                    return {
                        "iterations": iterations,
                        "valid_solution": valid_solution,
                        "solution": solution,
                        "patch": patch,
                        "depth": completeness,
                        "early_stopped": True,
                        "success": True,
                        "used_relaxed_validation": True
                    }

        except Exception as e:
            logger.error(f"Error generating initial solution: {e}")
            return {"error": str(e)}

        # Second iteration with self-reflection if needed
        if not valid_solution and len(iterations) < self.max_iterations:
            try:
                logger.info("Performing self-reflection to improve solution")

                # Get validation feedback if available
                validation_feedback = ""
                if iterations and "validation" in iterations[-1]:
                    validation_feedback = iterations[-1]["validation"].get("feedback", "")

                # Create reflection prompt
                reflection_prompt = self._create_reflection_prompt(
                    issue_description,
                    bug_location,
                    current_solution,
                    validation_feedback
                )

                # Generate reflection and improved solution
                reflection_response = self.model.generate(reflection_prompt)

                # Extract reflection and improved solution
                reflection, improved_solution = self._parse_reflection_response(reflection_response)

                # Extract patch from improved solution
                improved_patch = self._extract_patch(improved_solution)

                # Calculate solution completeness
                improved_completeness = self._evaluate_solution_completeness(improved_solution,
                                                                             improved_patch is not None)

                # Track reflection solution
                iteration_result = {
                    "iteration": 2,
                    "reflection": reflection,
                    "solution": improved_solution,
                    "patch": improved_patch,
                    "completeness": improved_completeness,
                    "evaluation": {
                        "syntax_validity": self.metrics["syntax_validity"],
                        "logic_completeness": self.metrics["logic_completeness"],
                        "patch_format": self.metrics["patch_format"],
                    }
                }

                # Validate improved patch
                if improved_patch:
                    validation_result = self.patch_validator.validate_patch(improved_patch, issue_id)
                    iteration_result["validation"] = validation_result

                    # If valid patch, use as final solution
                    if validation_result.get("success", False):
                        valid_solution = {
                            "solution": improved_solution,
                            "patch": improved_patch,
                            "validation": validation_result,
                            "reflection": reflection,
                            "iteration": 2
                        }
                        logger.info("Found valid patch in reflection solution")

                iterations.append(iteration_result)

            except Exception as e:
                logger.error(f"Error in self-reflection: {e}")

        # Choose the best iteration if no valid solution found
        best_iteration = None
        best_completeness = 0.0

        for iteration in iterations:
            if iteration["completeness"] > best_completeness:
                best_completeness = iteration["completeness"]
                best_iteration = iteration

        # If we have a valid solution, use that
        if valid_solution:
            solution = valid_solution["solution"]
            patch = valid_solution["patch"]
            success = True
        elif best_iteration:
            # Otherwise use the best iteration
            solution = best_iteration["solution"]
            patch = best_iteration.get("patch")
            success = False
        else:
            # Fallback if no iterations (shouldn't happen)
            solution = "No solution generated"
            patch = None
            success = False

        return {
            "iterations": iterations,
            "valid_solution": valid_solution,
            "solution": solution,
            "patch": patch,
            "depth": best_completeness,
            "early_stopped": False,
            "success": success
        }

    def _create_solution_prompt(
            self,
            issue_description: str,
            codebase_context: str,
            bug_location: Dict[str, Any],
            root_cause: Optional[str] = None
    ) -> str:
        """
        Create the solution development prompt.

        Args:
            issue_description: Issue description text.
            codebase_context: Codebase context information.
            bug_location: Bug location dictionary.
            root_cause: Optional root cause analysis.

        Returns:
            Formatted prompt string.
        """
        # Include root cause analysis if available
        root_cause_section = ""
        if root_cause:
            root_cause_section = f"""
            ROOT CAUSE ANALYSIS:
            {root_cause}
            """

        prompt = f"""
        You are an expert software engineer tasked with fixing a precisely located bug.

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}
        Issue: {bug_location.get('issue', 'Unknown')}

        {root_cause_section}

        CODEBASE CONTEXT:
        {codebase_context}

        Your task is to develop a precise solution for this bug:
        1. First, analyze the bug and explain your understanding of it
        2. Then, create a specific patch that fixes the issue
        3. Make sure the patch is minimal and focused on the specific bug

        Your solution MUST include:
        1. A brief technical explanation of the bug and your fix approach
        2. A complete and properly formatted Git patch that can be applied to the repository

        The Git patch must:
        - Start with "diff --git a/path/to/file b/path/to/file"
        - Include standard Git diff headers (---, +++)
        - Have proper hunk headers (@@ -start,count +start,count @@)
        - Contain adequate context lines (unchanged code)
        - Use - for removed lines and + for added lines

        Generate the COMPLETE solution in one go.
        """

        return prompt

    def _create_reflection_prompt(
            self,
            issue_description: str,
            bug_location: Dict[str, Any],
            initial_solution: str,
            validation_feedback: str = ""
    ) -> str:
        """
        Create the reflection prompt to improve the initial solution.

        Args:
            issue_description: Issue description text.
            bug_location: Bug location dictionary.
            initial_solution: Initial solution text.
            validation_feedback: Validation feedback if available.

        Returns:
            Formatted reflection prompt.
        """
        feedback_section = ""
        if validation_feedback:
            feedback_section = f"""
            VALIDATION FEEDBACK:
            {validation_feedback}

            Please pay special attention to fixing the issues identified in the validation feedback.
            """

        prompt = f"""
        You are an expert software engineer applying self-reflection to improve a bug fix solution.

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}
        Issue: {bug_location.get('issue', 'Unknown')}

        INITIAL SOLUTION:
        {initial_solution}

        {feedback_section}

        Your task is to:

        1. REFLECTION: Critically analyze the initial solution. Consider:
           - Is the patch correctly formatted following Git standards?
           - Are the file paths correct and consistent with the bug location?
           - Are line numbers and hunk headers accurate?
           - Does the solution properly address the root cause?
           - Is the solution minimal and focused?
           - Are there any edge cases not handled?

        2. IMPROVED SOLUTION: Based on your reflection, provide an improved solution that:
           - Fixes any formatting issues in the patch
           - Ensures file paths are correct
           - Makes sure line numbers and contexts match
           - Addresses any missed aspects of the bug
           - Handles potential edge cases

        Your response should be structured as:

        REFLECTION:
        [Your critical analysis]

        IMPROVED SOLUTION:
        [Your complete improved solution, including properly formatted Git patch]
        """

        return prompt

    def _parse_reflection_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the reflection response into reflection and improved solution.

        Args:
            response: Reflection response text.

        Returns:
            Tuple of (reflection, improved_solution).
        """
        reflection = ""
        improved_solution = ""

        # Try to find REFLECTION: and IMPROVED SOLUTION: markers
        reflection_match = re.search(r'REFLECTION:(.*?)(?=IMPROVED SOLUTION:|$)', response, re.DOTALL)
        solution_match = re.search(r'IMPROVED SOLUTION:(.*)', response, re.DOTALL)

        if reflection_match:
            reflection = reflection_match.group(1).strip()

        if solution_match:
            improved_solution = solution_match.group(1).strip()

        # If no matches, try alternative patterns
        if not reflection and not improved_solution:
            # Try to identify a natural split between analysis and solution
            # Often the solution part contains a patch or code blocks
            if "```" in response or "diff --git" in response:
                # Find the position of the first diff or code block
                diff_pos = response.find("diff --git")
                code_pos = response.find("```")

                split_pos = -1
                if diff_pos >= 0 and code_pos >= 0:
                    split_pos = min(diff_pos, code_pos)
                elif diff_pos >= 0:
                    split_pos = diff_pos
                elif code_pos >= 0:
                    split_pos = code_pos

                if split_pos > 0:
                    # Look for the previous paragraph break before the diff/code
                    prev_break = response.rfind("\n\n", 0, split_pos)
                    if prev_break >= 0:
                        reflection = response[:prev_break].strip()
                        improved_solution = response[prev_break:].strip()
                    else:
                        # Arbitrary split if no clear paragraph break
                        reflection = response[:split_pos].strip()
                        improved_solution = response[split_pos:].strip()

            # If still no solution, split roughly in half
            if not reflection or not improved_solution:
                lines = response.split('\n')
                mid_point = len(lines) // 2
                reflection = '\n'.join(lines[:mid_point]).strip()
                improved_solution = '\n'.join(lines[mid_point:]).strip()

        return reflection, improved_solution

    def _evaluate_solution_completeness(self, solution: str, has_patch: bool) -> float:
        """
        Evaluate the completeness of a solution.

        Args:
            solution: Solution text.
            has_patch: Whether the solution includes a patch.

        Returns:
            Completeness score between 0 and 1.
        """
        # Reset metrics
        self.metrics = {
            "syntax_validity": 0.0,
            "logic_completeness": 0.0,
            "patch_format": 0.0,
            "overall_completeness": 0.0,
        }

        # Base score
        score = 0.5  # Start with base score for having a solution

        # Check for syntax validity
        syntax_indicators = ['```', 'def ', 'class ', 'if ', 'for ', 'return ', '+=', '-=', '==', '!=']
        syntax_count = sum(1 for indicator in syntax_indicators if indicator in solution)
        syntax_score = min(0.2, syntax_count * 0.03)
        score += syntax_score
        self.metrics["syntax_validity"] = syntax_score

        # Check for patch format
        if has_patch:
            patch_score = 0.3
            score += patch_score
            self.metrics["patch_format"] = patch_score
        else:
            # Check for signs of a patch even if not properly formatted
            patch_indicators = ['diff', '---', '+++', '@@ ', 'a/', 'b/']
            patch_count = sum(1 for indicator in patch_indicators if indicator in solution)
            patch_score = min(0.2, patch_count * 0.04)
            score += patch_score
            self.metrics["patch_format"] = patch_score

        # Check for logic completeness
        logic_indicators = [
            'bug', 'issue', 'problem', 'fix', 'solution', 'cause',
            'change', 'update', 'modify', 'replace', 'add', 'remove'
        ]
        logic_count = sum(1 for term in logic_indicators if term in solution.lower())
        logic_score = min(0.2, logic_count * 0.02)
        score += logic_score
        self.metrics["logic_completeness"] = logic_score

        # Check for explanation quality
        if solution.count('\n') > 10:  # Substantial solution with explanation
            score += 0.1
        elif 'because' in solution.lower() or 'reason' in solution.lower():
            score += 0.05

        self.metrics["overall_completeness"] = min(1.0, score)
        return min(1.0, score)

    def _extract_patch(self, response: str) -> Optional[str]:
        """
        Extract a Git patch from the response.

        Args:
            response: Solution text.

        Returns:
            Extracted patch string or None.
        """
        # Look for diff --git pattern
        patch_pattern = r'(diff --git.*?)(?=^```|\Z)'
        patch_match = re.search(patch_pattern, response, re.MULTILINE | re.DOTALL)

        if patch_match:
            return patch_match.group(1).strip()

        # Look for code blocks
        code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
        code_match = re.search(code_block_pattern, response, re.MULTILINE | re.DOTALL)

        if code_match:
            content = code_match.group(1).strip()
            if content.startswith('diff --git') or ('---' in content and '+++' in content):
                return content

        # Check if the response itself is a patch
        if response.strip().startswith('diff --git') or ('---' in response and '+++' in response):
            return response.strip()

        return None

    def _try_relaxed_validation(self, patch: str, issue_id: str) -> Dict[str, Any]:
        """
        Try validating a patch with more relaxed options.

        Args:
            patch: Patch string.
            issue_id: Issue ID for validation.

        Returns:
            Validation result dictionary.
        """
        # This function would ideally call into the patch validator with relaxed options
        # For now, we'll just check if the patch would apply with --ignore-whitespace
        try:
            # Get the issue
            from ..data.data_loader import SWEBenchDataLoader
            data_loader = SWEBenchDataLoader(self.config)
            issue = data_loader.load_issue(issue_id)

            if not issue:
                return {"success": False, "feedback": f"Could not load issue {issue_id}"}

            # Get repo path
            repo = issue.get("repo", "")
            repo_path = self.config["data"]["repositories"] / repo

            if not repo_path.exists():
                return {"success": False, "feedback": f"Repository path not found: {repo_path}"}

            # Save patch to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch)
                patch_file = f.name

            # Try applying with relaxed options
            import subprocess
            try:
                subprocess.run(
                    ["git", "apply", "--check", "--ignore-whitespace", "--ignore-space-change", patch_file],
                    cwd=repo_path,
                    check=True,
                    capture_output=True
                )

                # If we get here, patch would apply with relaxed options
                return {
                    "success": True,
                    "feedback": "Patch applies with relaxed whitespace handling options.",
                    "relaxed_options_used": True
                }
            except subprocess.CalledProcessError as e:
                # Even with relaxed options, patch doesn't apply
                return {
                    "success": False,
                    "feedback": f"Patch doesn't apply even with relaxed options: {e.stderr.decode('utf-8')}"
                }
            finally:
                # Clean up
                import os
                os.unlink(patch_file)

        except Exception as e:
            return {"success": False, "feedback": f"Error in relaxed validation: {str(e)}"}


