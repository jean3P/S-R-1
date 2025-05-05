# src/solution/robust_bug_fixing_pipeline.py

import logging
import re
import time
import hashlib
import json
import tempfile
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
import gc

from ..data.astropy_synthetic_dataloader import AstropySyntheticDataLoader
from ..models import create_model
from ..utils.patch_validator import PatchValidator

logger = logging.getLogger(__name__)


class RobustBugFixingPipeline:
    """
    Implements a robust bug fixing pipeline with Chain of Thought, Self-Reflection,
    Instrumentation, Patch Ranking, and Caching to solve bugs in Python code.
    """

    def __init__(self, config, model_name=None):
        """
        Initialize the robust bug fixing pipeline.

        Args:
            config: Configuration object.
            model_name: Name of the model to use, or None to use default.
        """
        self.config = config
        self.model_name = model_name or config.get("default_model", "deepseek-r1-distill")
        self.model = None  # Lazy-load model on demand

        # Initialize core components
        self.data_loader = AstropySyntheticDataLoader(config)
        self.patch_validator = PatchValidator(config)

        # Configure parameters
        self.max_iterations = config.get("max_iterations", 5)
        self.test_timeout = config.get("test_timeout", 300)  # Timeout for test execution in seconds

        # Cached patches to avoid re-testing the same patch
        self.failed_patch_hashes = set()

        # Directory for storing results
        self.results_dir = Path(config.get("evaluation", {}).get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Memory optimization
        self.use_memory_optimization = config.get("memory_efficient", True)

        logger.info(f"Initialized RobustBugFixingPipeline with model {self.model_name}")

    def _get_model(self):
        """Lazy-load model when needed."""
        if self.model is None:
            logger.info(f"Initializing model: {self.model_name}")
            self._run_memory_cleanup()
            self.model = create_model(self.model_name, self.config)
        return self.model

    def _run_memory_cleanup(self):
        """Run memory cleanup operations."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(f"GPU {i} memory: allocated={allocated:.2f}GB (max: {max_allocated:.2f}GB), "
                            f"reserved={reserved:.2f}GB")

    def solve_bug(self, bug_id: str) -> Dict[str, Any]:
        """
        Solve a bug using CoT, self-reflection, instrumentation, patch ranking, and caching.

        Args:
            bug_id: Bug ID to solve.

        Returns:
            Dictionary with solution results.
        """
        logger.info(f"Starting robust bug fixing pipeline for bug {bug_id}")
        start_time = time.time()

        # Load bug data
        bug_data = self._load_bug_data(bug_id)
        if not bug_data or "error" in bug_data:
            error_message = bug_data.get("error", f"Failed to load data for bug {bug_id}")
            logger.error(error_message)
            return {
                "bug_id": bug_id,
                "status": "error",
                "error_message": error_message,
                "processing_time": time.time() - start_time
            }
        # Get branch name from the issue and checkout first
        branch_name = bug_id
        if branch_name:
            # Checkout the branch with the bug to ensure we're testing the right code
            checkout_success = self._git_checkout_branch(branch_name)

        # Initialize tracking
        iteration_logs = []
        self.failed_patch_hashes = set()  # Reset cache for this bug
        iteration_count = 0

        # Track if we found a solution
        solution_found = False
        final_patch = None

        # Main iteration loop
        while iteration_count < self.max_iterations and not solution_found:
            iteration_count += 1
            logger.info(f"Starting iteration {iteration_count}/{self.max_iterations}")

            # Clear memory before generation
            self._run_memory_cleanup()

            try:
                # Phase selection: First iteration is CoT, others are Self-Reflection
                if iteration_count == 1:
                    phase = "CoT"
                    # Generate patch using Chain of Thought reasoning
                    patch, explanation = self._generate_patch_with_cot(bug_data)
                else:
                    phase = "Self-Reflection"
                    # Get previous iteration data
                    prev_iteration = iteration_logs[-1]
                    prev_patch = prev_iteration.get("patch_text", "")
                    prev_error = prev_iteration.get("test_result", {}).get("error_message", "")

                    # Generate improved patch using self-reflection
                    patch, explanation = self._generate_patch_with_reflection(
                        bug_data,
                        prev_patch,
                        prev_error,
                        iteration_logs
                    )

                # Calculate patch hash for caching
                patch_hash = self._calculate_patch_hash(patch)

                # Skip if we've already tested this patch
                if patch_hash in self.failed_patch_hashes:
                    logger.info(f"Skipping duplicate patch (hash: {patch_hash[:8]}...)")

                    # Create a log entry for the skipped attempt
                    iteration_log = {
                        "iteration": iteration_count,
                        "phase": phase,
                        "patch_hash": patch_hash,
                        "patch_text": patch,
                        "explanation": explanation,
                        "skipped": True,
                        "reason": "Duplicate patch",
                        "test_result": {"status": "skipped"}
                    }
                    iteration_logs.append(iteration_log)
                    continue

                # Determine instrumentation points based on the bug
                instrumentation = self._generate_instrumentation(bug_data, patch)

                # Apply the patch and instrumentation
                patched_path, original_content = self._apply_patch_with_instrumentation(bug_data, patch,
                                                                                        instrumentation)

                # Run the test
                test_result = self._run_test(bug_data, patched_path)

                # Check if the test passes
                if test_result.get("status") == "pass":
                    solution_found = True
                    final_patch = patch
                else:
                    # Add to failed hashes
                    self.failed_patch_hashes.add(patch_hash)

                # Create a log entry for this iteration
                iteration_log = {
                    "iteration": iteration_count,
                    "phase": phase,
                    "patch_hash": patch_hash,
                    "patch_text": patch,
                    "explanation": explanation,
                    "instrumentation": instrumentation,
                    "test_result": test_result
                }
                iteration_logs.append(iteration_log)

                # Restore original file
                if patched_path and original_content is not None:
                    with open(patched_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)

            except Exception as e:
                logger.error(f"Error in iteration {iteration_count}: {str(e)}", exc_info=True)

                # Log the error
                iteration_log = {
                    "iteration": iteration_count,
                    "phase": phase if 'phase' in locals() else "unknown",
                    "error": str(e),
                    "test_result": {"status": "error", "error_message": str(e)}
                }
                iteration_logs.append(iteration_log)

                # Don't add to failed hashes for errors

        # Prepare final result
        processing_time = time.time() - start_time

        if solution_found:
            result = {
                "bug_id": bug_id,
                "status": "passed",
                "final_patch": final_patch,
                "iterations": iteration_count,
                "history": iteration_logs,
                "processing_time": processing_time
            }
        else:
            result = {
                "bug_id": bug_id,
                "status": "no_solution",
                "iterations": iteration_count,
                "history": iteration_logs,
                "processing_time": processing_time
            }

        # Save results
        self._save_results(bug_id, result)

        logger.info(f"Completed bug fixing for {bug_id} with status {result['status']} in {processing_time:.2f}s")
        return result

    def _git_checkout_branch(self, branch_name: str) -> bool:
        """
        Checkout a specific git branch for analysis.

        Args:
            branch_name: Name of the branch to checkout

        Returns:
            True if checkout successful, False otherwise
        """
        try:
            import subprocess
            repo_path = Path("/storage/homefs/jp22b083/SSI/S-R-1/data/repositories/astropy/astropy")

            # Make sure we're not going to try to checkout a non-git dir
            if not os.path.exists(os.path.join(repo_path, ".git")):
                logger.error(f"Not a git repository: {repo_path}")
                return False

            # Regular checkout (local branch exists)
            result = subprocess.run(
                ["git", "checkout", branch_name],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"Git checkout failed: {result.stderr}")

            logger.info(f"Git checkout branch: {branch_name} {'successful' if result.returncode == 0 else 'failed'}")
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error checking out branch {branch_name}: {str(e)}")
            return False

    def _load_bug_data(self, bug_id: str) -> Dict[str, Any]:
        """
        Load bug data including location, context, and test information.

        Args:
            bug_id: Bug ID to load data for.

        Returns:
            Dictionary with bug data.
        """
        # Load issue
        issue = self.data_loader.load_issue(bug_id)
        if not issue:
            return {"error": f"Issue {bug_id} not found"}

        # Get bug location from a JSON file or other sources
        bug_location = self._load_bug_location(bug_id)
        if not bug_location or "error" in bug_location:
            return {"error": f"Failed to load bug location: {bug_location.get('error', 'Unknown error')}"}

        # Extract essential information
        bug_data = {
            "bug_id": bug_id,
            "impl_file_path": bug_location.get("file", issue.get("impl_file_path", "")),
            "impl_function_name": bug_location.get("function", issue.get("impl_function_name", "")),
            "line_start": bug_location.get("line_start", 0),
            "line_end": bug_location.get("line_end", 0),
            "bug_lines": bug_location.get("bug_lines", []),
            "code_content": bug_location.get("code_content", ""),
            "problem_statement": issue.get("problem_statement", bug_location.get("bug_description", "")),
            "hint_text": issue.get("hint_text", ""),
            "test_file_path": bug_location.get("test_file", issue.get("test_file_path", "")),
            "test_function_name": bug_location.get("test_function", issue.get("test_function_name", "")),
            "path_env": issue.get("path_env", "python"),
            "repo_path": Path("/storage/homefs/jp22b083/SSI/S-R-1/data/repositories/astropy/astropy")
        }

        # If bug lines is empty but we have line_start and line_end, create a range
        if not bug_data["bug_lines"] and bug_data["line_start"] > 0 and bug_data["line_end"] > 0:
            bug_data["bug_lines"] = list(range(bug_data["line_start"], bug_data["line_end"] + 1))

        return bug_data

    def _load_bug_location(self, bug_id: str) -> Dict[str, Any]:
        """
        Load the bug location from a JSON file.

        Args:
            bug_id: Bug ID to load.

        Returns:
            Dictionary with bug location.
        """
        # Common JSON file paths for bug locations
        bug_locations_json_paths = [
            "/storage/homefs/jp22b083/SSI/S-R-1/results/enhanced_bug_detector/bug_locations.json"
        ]

        # Try loading from the combined JSON file
        for path in bug_locations_json_paths:
            try:
                with open(path, 'r') as f:
                    all_bug_locations = json.load(f)
                    # Find matching issue by ID
                    for bug_location in all_bug_locations:
                        if bug_location.get("issue_id") == bug_id:
                            logger.info(f"Found bug location for issue {bug_id}")
                            # Extract only the necessary information as specified
                            return {
                                "file": bug_location.get("file", ""),
                                "function": bug_location.get("function", ""),
                                "line_start": bug_location.get("line_start", 0),
                                "line_end": bug_location.get("line_end", 0),
                                "code_content": bug_location.get("code_content", ""),
                                "bug_type": bug_location.get("bug_type", ""),
                                "bug_description": bug_location.get("bug_description", ""),
                                "bug_lines": bug_location.get("alternative_bug_lines", []),
                                "issue_id": bug_id,
                                "test_file": bug_location.get("test_file", ""),
                                "test_function": bug_location.get("test_function", "")
                            }
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Error loading bug locations from {path}: {str(e)}")
                continue

        # Fallback: Try to extract basic info from issue
        logger.warning(f"No bug location file found for {bug_id}, using issue data")
        issue = self.data_loader.load_issue(bug_id)

        if issue:
            failing_code = issue.get("FAIL_TO_PASS", "")
            function_name = self._extract_function_name(failing_code)

            return {
                "file": issue.get("impl_file_path", "unknown_file.py"),
                "function": function_name or issue.get("impl_function_name", "unknown_function"),
                "code_content": failing_code,
                "bug_description": "Bug location could not be found in detector output",
                "issue_id": bug_id
            }

        # Error case
        return {
            "error": f"Could not load bug location for issue {bug_id}",
            "file": None,
            "function": None
        }

    def _extract_function_name(self, code: str) -> str:
        """Extract the function name from a code snippet."""
        # Try to extract method name from both function and class definitions
        pattern = r"(class|def)\s+([a-zA-Z0-9_]+)\s*[\(:]"
        matches = re.findall(pattern, code)

        if matches:
            for match_type, name in matches:
                # Return class name if it's a class definition
                if match_type == 'class':
                    return name

            # Otherwise return the first function/method name
            for match_type, name in matches:
                if match_type == 'def':
                    return name

        return ""

    def _generate_patch_with_cot(self, bug_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate a patch using Chain of Thought reasoning.

        Args:
            bug_data: Bug data dictionary.

        Returns:
            Tuple of (patch, explanation).
        """
        logger.info("Generating patch using Chain of Thought reasoning")
        model = self._get_model()

        # Create a comprehensive prompt with all bug details
        prompt = self._create_cot_prompt(bug_data)

        # Generate the solution
        response = model.generate(prompt)

        # Extract patch and explanation
        patch, explanation = self._extract_patch_and_explanation(response)

        logger.info(f"Generated patch with CoT ({len(patch)} chars, explanation: {len(explanation)} chars)")
        return patch, explanation

    def _generate_patch_with_reflection(
            self,
            bug_data: Dict[str, Any],
            previous_patch: str,
            error_message: str,
            iteration_logs: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Generate an improved patch through self-reflection.

        Args:
            bug_data: Bug data dictionary.
            previous_patch: Previous patch that failed.
            error_message: Error message from test run.
            iteration_logs: Logs of all previous iterations.

        Returns:
            Tuple of (improved_patch, explanation).
        """
        logger.info("Generating improved patch through self-reflection")
        model = self._get_model()

        # Create self-reflection prompt with test feedback and previous attempts
        prompt = self._create_reflection_prompt(bug_data, previous_patch, error_message, iteration_logs)

        # Generate the improved solution
        response = model.generate(prompt)

        # Extract patch and explanation
        improved_patch, explanation = self._extract_patch_and_explanation(response)

        logger.info(f"Generated patch with self-reflection ({len(improved_patch)} chars)")
        return improved_patch, explanation

    def _generate_instrumentation(self, bug_data: Dict[str, Any], patch: str) -> List[str]:
        """
        Generate instrumentation code to insert at bug location.

        Args:
            bug_data: Bug data dictionary.
            patch: The patch to be applied.

        Returns:
            List of instrumentation lines to insert.
        """
        # Basic instrumentation: print statements for key variables or expressions
        instrumentation = []

        # Identify key variables from the bug data and patch
        variables = self._extract_key_variables(bug_data, patch)

        for var in variables:
            # Add print statements for debugging
            instrumentation.append(f"print(f'DEBUG: {var} = {{type({var})}} {{repr({var})}}')")

            # Add assertion if we can infer constraints
        constraints = self._extract_constraints(bug_data, patch)
        for constraint in constraints:
            instrumentation.append(
                f"assert {constraint}, f'Constraint failed: {constraint} ({{type({constraint})}} {{repr({constraint})}})'")

        logger.info(f"Generated {len(instrumentation)} instrumentation points")
        return instrumentation

    def _extract_key_variables(self, bug_data: Dict[str, Any], patch: str) -> List[str]:
        """Extract key variables that should be monitored."""
        variables = []

        # Look for variables in the diff
        var_pattern = r'[+-]\s*(?!(?:def|class|import|from|print|return|if|else|elif|for|while))(\w+)[\s=\(\[]'
        removed_vars = re.findall(
            r'-\s*(?!(?:def|class|import|from|print|return|if|else|elif|for|while))(\w+)[\s=\(\[]', patch)
        added_vars = re.findall(r'\+\s*(?!(?:def|class|import|from|print|return|if|else|elif|for|while))(\w+)[\s=\(\[]',
                                patch)

        # Variables that are both added and removed are likely important
        for var in set(removed_vars) & set(added_vars):
            if var not in ['self', 'None', 'True', 'False'] and var.isalnum():
                variables.append(var)

        # Variables in conditional statements
        if_pattern = r'[+-]\s*if\s+(.+?):'
        if_matches = re.findall(if_pattern, patch)
        for match in if_matches:
            # Extract variables from the condition
            cond_vars = re.findall(r'(\w+)(?:\s*(?:[=!<>]=|[<>])\s*|\s+in\s+|\s+not\s+in\s+|\s+is\s+|\s+is\s+not\s+)',
                                   match)
            for var in cond_vars:
                if var not in ['self', 'None', 'True', 'False'] and var.isalnum() and var not in variables:
                    variables.append(var)

        # If no variables found, try to infer from line context
        if not variables and bug_data["bug_lines"]:
            # Extract code lines at bug location
            code_lines = bug_data["code_content"].splitlines()
            for line_num in bug_data["bug_lines"]:
                if 0 <= line_num - 1 < len(code_lines):
                    line = code_lines[line_num - 1]
                    # Look for assignment or condition
                    line_vars = re.findall(r'(\w+)(?:\s*=\s*|\s+in\s+|\s*(?:[=!<>]=|[<>])\s*)', line)
                    for var in line_vars:
                        if var not in ['self', 'None', 'True', 'False'] and var.isalnum() and var not in variables:
                            variables.append(var)

        # Limit to at most 3 variables for clarity
        return variables[:3]

    def _extract_constraints(self, bug_data: Dict[str, Any], patch: str) -> List[str]:
        """Extract constraints that should be asserted."""
        constraints = []

        # Look for common patterns in the patch
        # 1. Type checks or instance checks
        type_checks = re.findall(r'[+-]\s*(?:isinstance|type)\((\w+)[,\s]', patch)
        for var in type_checks:
            if var not in ['self', 'None', 'True', 'False']:
                # Generic type check
                constraints.append(f"{var} is not None")

        # 2. Parentheses changes often imply operator precedence issues
        paren_pattern = r'[+-][^()]+(\([^()]+\))'
        if re.search(paren_pattern, patch):
            # Look for added or changed parentheses
            added_parens = re.findall(r'\+[^()]+(\([^()]+\))', patch)
            removed_parens = re.findall(r'-[^()]+(\([^()]+\))', patch)

            if added_parens or removed_parens:
                # Extract the condition inside parentheses
                for paren_expr in added_parens:
                    # Clean up expression
                    expr = paren_expr.strip('()')
                    if ' and ' in expr or ' or ' in expr or ' not ' in expr:
                        # Logical expression
                        constraints.append(expr)

        return constraints[:2]  # Limit to 2 constraints

    def _create_cot_prompt(self, bug_data: Dict[str, Any]) -> str:
        """
        Create a Chain of Thought prompt for patch generation.

        Args:
            bug_data: Bug data dictionary.

        Returns:
            Formatted prompt string.
        """
        # Format bug information
        file_path = "astropy" + "/"+ bug_data.get("impl_file_path", "")
        function_name = bug_data.get("impl_function_name", "")
        bug_lines = bug_data.get("bug_lines", [])
        bug_lines_str = ", ".join(map(str, bug_lines)) if bug_lines else "Unknown"
        problem_statement = bug_data.get("problem_statement", "")
        hint_text = bug_data.get("hint_text", "")
        code_content = bug_data.get("code_content", "")
        test_file = bug_data.get("test_file_path", "")
        test_function = bug_data.get("test_function_name", "")

        # Construct the prompt
        prompt = f"""
        You are an expert software engineer tasked with fixing a precisely located bug in Python code.

        # BUG INFORMATION
        - File: {file_path}
        - Function: {function_name}
        - Bug Lines: {bug_lines_str}
        - Problem: {problem_statement}
        {f"- Hint: {hint_text}" if hint_text else ""}
        
        # TEST INFORMATION
        - Test File: {test_file}
        - Test Function: {test_function}

        # CODE WITH BUG
        ```python
        {code_content}
        ```

        # YOUR TASK
        1. Think through what this function is doing and where the bug lies.
        2. Explain the root cause of the bug in plain terms.
        3. Describe why the test is failing.
        4. Propose a patch to fix the bug in the function.

        # Response Format
        ## Root Cause Analysis
        [Provide your detailed analysis of the bug here]

        ## Fix Explanation
        [Explain your solution approach]

        ## Patch
        ```diff
        [Provide a proper Git diff patch here]
        ```

        The patch must:
        - Start with "diff --git a/[filepath] b/[filepath]"
        - Use correct hunk headers (@@ -line,count +line,count @@)
        - Include proper diff headers (---, +++)
        - Use - for removed lines and + for added lines
        - Include adequate context (unchanged lines)
        - Fix only the identified logic bug

        Be precise. Think step-by-step.
        """

        return prompt

    def _create_reflection_prompt(
            self,
            bug_data: Dict[str, Any],
            previous_patch: str,
            error_message: str,
            iteration_logs: List[Dict[str, Any]]
    ) -> str:
        """
        Create a self-reflection prompt for improved patch generation.

        Args:
            bug_data: Bug data dictionary.
            previous_patch: Previous patch that failed.
            error_message: Error message from test run.
            iteration_logs: Logs of all previous iterations.

        Returns:
            Formatted prompt string.
        """
        # Format bug information
        file_path = "astropy" + "/"+  bug_data.get("impl_file_path", "")
        function_name = bug_data.get("impl_function_name", "")
        bug_lines = bug_data.get("bug_lines", [])
        bug_lines_str = ", ".join(map(str, bug_lines)) if bug_lines else "Unknown"
        problem_statement = bug_data.get("problem_statement", "")
        hint_text = bug_data.get("hint_text", "")
        code_content = bug_data.get("code_content", "")
        test_function = bug_data.get("test_function_name", "")

        # Extract test results from previous iterations
        test_feedback = []
        for log in iteration_logs:
            test_result = log.get("test_result", {})
            if test_result.get("status") != "pass":
                feedback = f"Iteration {log.get('iteration')}: {test_result.get('error_message', 'Unknown error')}"
                test_feedback.append(feedback)

        test_feedback_str = "\n".join(test_feedback)

        # Construct the prompt
        prompt = f"""
        You previously attempted to fix a bug in the function `{function_name}`, but the patch failed to pass the test `{test_function}`.

        # ORIGINAL CODE WITH BUG
        ```python
        {code_content}
        ```
        
        # BUG INFORMATION
        - File: {file_path}
        - Bug Lines: {bug_lines_str}
        - Problem: {problem_statement}
        
        # PREVIOUS PATCH ATTEMPT (FAILED)
        ```diff
        {previous_patch}
        ```

        # Test Result
        Latest error message: {error_message}

        {f"- Additional Hints: {hint_text}" if hint_text else ""}
        {f"### Test Feedback History {test_feedback_str}" if test_feedback else ""}

        # YOUR TASK
        1. Critically analyze why the previous patch failed
        2. Think about what the test failure is telling you
        3. Identify what assumptions were incorrect in your previous approach
        4. Generate an improved patch that addresses the issues in the function

        # Task Instructions
        - Reflect on why the previous patch failed.
        - Revise your assumptions and explain what needs to change.
        - Propose a new patch that better addresses the problem.
        
        # Response Format

        ## Reflection on Previous Attempt
        [Why did the last fix fail? What was misunderstood?]

        ## Improved Approach
        [What will you do differently in this new patch?]

        ## Improved Patch
        ```diff
        [Valid unified Git diff]
        ```

        The patch must:
        - Start with diff --git a/{file_path} b/{file_path}
        - Include valid headers and within the scope of the function
        - Directly target the bug without affecting unrelated lines

        Be analytical. Learn from the failure. Then correct it.
        """

        return prompt

    def _extract_patch_and_explanation(self, response: str) -> Tuple[str, str]:
        """
        Extract the patch and explanation from the model response.

        Args:
            response: Model response text.

        Returns:
            Tuple of (patch, explanation).
        """
        # Extract the patch
        patch_pattern = r'```diff\n(.*?)```'
        patch_match = re.search(patch_pattern, response, re.DOTALL)

        if patch_match:
            patch = patch_match.group(1).strip()
        else:
            # Fallback: look for any code block that might contain a patch
            code_block_pattern = r'```(?:.*?)\n(diff --git.*?)```'
            code_match = re.search(code_block_pattern, response, re.DOTALL)

            if code_match:
                patch = code_match.group(1).strip()
            else:
                # Last resort: look for diff directly
                diff_pattern = r'(diff --git.*?)(?=\n\n|\n#|\Z)'
                diff_match = re.search(diff_pattern, response, re.DOTALL)
                if diff_match:
                    patch = diff_match.group(1).strip()
                else:
                    # No patch found
                    patch = ""

        # Make sure the patch starts with diff --git
        if patch and not patch.startswith("diff --git"):
            # Try to fix the patch
            if "diff --git" in patch:
                patch = patch[patch.find("diff --git"):]

        # Extract the explanation
        explanation = ""

        # Look for structured sections
        root_cause_pattern = r'##\s*Root Cause Analysis\s*(.*?)(?=##|\Z)'
        fix_pattern = r'##\s*Fix Explanation\s*(.*?)(?=##|\Z)'
        reflection_pattern = r'##\s*Reflection on Previous Attempt\s*(.*?)(?=##|\Z)'
        improved_pattern = r'##\s*Improved Approach\s*(.*?)(?=##|\Z)'

        root_cause_match = re.search(root_cause_pattern, response, re.DOTALL)
        fix_match = re.search(fix_pattern, response, re.DOTALL)
        reflection_match = re.search(reflection_pattern, response, re.DOTALL)
        improved_match = re.search(improved_pattern, response, re.DOTALL)

        if root_cause_match:
            explanation += root_cause_match.group(1).strip() + "\n\n"

        if fix_match:
            explanation += fix_match.group(1).strip() + "\n\n"

        if reflection_match:
            explanation += reflection_match.group(1).strip() + "\n\n"

        if improved_match:
            explanation += improved_match.group(1).strip() + "\n\n"

        # If we couldn't find structured sections, extract everything before the patch
        if not explanation:
            if patch:
                parts = response.split("```diff")
                if parts and len(parts) > 0:
                    explanation = parts[0].strip()
            else:
                explanation = response.strip()

        return patch, explanation.strip()

    def _calculate_patch_hash(self, patch: str) -> str:
        """
        Calculate a hash for the patch to detect duplicates.

        Args:
            patch: Patch string.

        Returns:
            Patch hash.
        """
        # Normalize the patch to ignore whitespace differences
        normalized_patch = self._normalize_patch(patch)

        # Calculate hash
        return hashlib.sha256(normalized_patch.encode('utf-8')).hexdigest()

    def _normalize_patch(self, patch: str) -> str:
        """
        Normalize a patch to ignore insignificant differences.

        Args:
            patch: Original patch string.

        Returns:
            Normalized patch string.
        """
        lines = patch.splitlines()
        normalized_lines = []

        for line in lines:
            # Skip diff headers and hunk headers
            if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++') or line.startswith(
                    '@@'):
                continue

            # Keep only content lines with changes
            if line.startswith('+') or line.startswith('-'):
                # Remove whitespace
                normalized_line = re.sub(r'\s+', '', line)
                normalized_lines.append(normalized_line)

        return '\n'.join(normalized_lines)

    def _apply_patch_with_instrumentation(
            self,
            bug_data: Dict[str, Any],
            patch: str,
            instrumentation: List[str]
    ) -> Tuple[str, str]:
        """
        Apply the patch and instrumentation to the target file.

        Args:
            bug_data: Bug data dictionary.
            patch: Patch string.
            instrumentation: List of instrumentation lines.

        Returns:
            Tuple of (patched_file_path, original_content).
        """
        # Get file path
        file_path = bug_data.get("impl_file_path", "")
        if not file_path:
            logger.error("No implementation file path specified")
            return None, None

        # Get repo path
        repo_path = bug_data.get("repo_path")
        if not repo_path:
            logger.error("No repository path specified")
            return None, None

        # Full path to the file
        full_path = repo_path / "astropy" / file_path

        # Check if file exists
        if not os.path.exists(full_path):
            logger.error(f"File not found: {full_path}")
            return None, None

        # Read original content
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return None, None

        # Apply the patch using git apply
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as temp_patch_file:
            temp_patch_file.write(patch)
            temp_patch_path = temp_patch_file.name

        try:
            # Apply the patch
            subprocess.run(
                ["git", "apply", "--check", temp_patch_path],
                cwd=repo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            subprocess.run(
                ["git", "apply", temp_patch_path],
                cwd=repo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            logger.info(f"Successfully applied patch to {file_path}")

            # Now apply instrumentation
            if instrumentation:
                # Read the patched content
                with open(full_path, 'r', encoding='utf-8') as f:
                    patched_content = f.read()

                # Find the best place to insert instrumentation
                lines = patched_content.splitlines()
                bug_lines = bug_data.get("bug_lines", [])

                # Default to the first line of the function
                insert_line = bug_data.get("line_start", 0)

                if bug_lines:
                    # Use the first bug line
                    insert_line = min(bug_lines)

                # Adjust for 0-based indexing
                if insert_line > 0:
                    insert_line -= 1

                # Ensure valid line index
                if 0 <= insert_line < len(lines):
                    # Get indentation
                    indent_match = re.match(r'^(\s*)', lines[insert_line])
                    indent = indent_match.group(1) if indent_match else '    '

                    # Add instrumentation lines with proper indentation
                    instrumentation_code = [f"{indent}{line}" for line in instrumentation]

                    # Insert after the line
                    lines.insert(insert_line + 1, "\n".join(instrumentation_code))

                    # Write back the modified content
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(lines))

                    logger.info(f"Added {len(instrumentation)} instrumentation lines")
                else:
                    logger.warning(f"Invalid line index for instrumentation: {insert_line}")

            # Clean up temp file
            os.unlink(temp_patch_path)

            return str(full_path), original_content

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply patch: {e.stderr}")
            # Clean up temp file
            os.unlink(temp_patch_path)
            return None, None
        except Exception as e:
            logger.error(f"Error applying patch: {e}")
            # Clean up temp file
            os.unlink(temp_patch_path)
            return None, None

    def _run_test(self, bug_data: Dict[str, Any], patched_path: Optional[str]) -> Dict[str, Any]:
        """
        Run the test to validate the patch.

        Args:
            bug_data: Bug data dictionary.
            patched_path: Path to the patched file or None if patch failed.

        Returns:
            Dictionary with test result.
        """
        if not patched_path:
            return {
                "status": "error",
                "error_message": "Failed to apply patch",
                "output": ""
            }

        # Get test information
        test_file = bug_data.get("test_file_path", "")
        test_function = bug_data.get("test_function_name", "")
        python_path = bug_data.get("path_env", "python")
        repo_path = bug_data.get("repo_path")

        if not test_file or not repo_path:
            return {
                "status": "error",
                "error_message": "Missing test information",
                "output": ""
            }

        # Construct test command
        if test_function:
            test_path = f"{test_file}::{test_function}"
        else:
            test_path = test_file

        test_cmd = [python_path, "-m", "pytest", test_path, "-v"]

        try:
            # Run the test
            logger.info(f"Running test: {' '.join(test_cmd)}")
            result = subprocess.run(
                test_cmd,
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.test_timeout
            )

            # Process the result
            test_passed = result.returncode == 0
            test_output = result.stdout + "\n" + result.stderr

            # Create structured result
            status = "pass" if test_passed else "fail"
            error_message = self._extract_error_message(test_output) if not test_passed else ""

            test_result = {
                "status": status,
                "error_message": error_message,
                "output": test_output,
                "returncode": result.returncode
            }

            logger.info(f"Test result: {status}")
            return test_result

        except subprocess.TimeoutExpired:
            logger.warning(f"Test execution timed out after {self.test_timeout} seconds")
            return {
                "status": "timeout",
                "error_message": f"Test execution timed out after {self.test_timeout} seconds",
                "output": "Test timed out"
            }
        except Exception as e:
            logger.error(f"Error running test: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "output": str(e)
            }

    def _extract_error_message(self, test_output: str) -> str:
        """
        Extract error message from test output.

        Args:
            test_output: Test output text.

        Returns:
            Extracted error message.
        """
        # Look for error lines
        error_patterns = [
            r'E\s+(.*Error:.+)',
            r'ERROR:.+',
            r'FAILED.+',
            r'AssertionError:.+'
        ]

        for pattern in error_patterns:
            error_match = re.search(pattern, test_output)
            if error_match:
                return error_match.group(0).strip()

        # If no specific error pattern found, extract the first traceback line
        traceback_match = re.search(r'Traceback.+?:\n(.+)', test_output, re.DOTALL)
        if traceback_match:
            # Get first non-empty line
            lines = traceback_match.group(1).strip().split('\n')
            for line in lines:
                if line.strip():
                    return line.strip()

        return "Unknown error"

    def _save_results(self, bug_id: str, result: Dict[str, Any]) -> None:
        """
        Save the results to a file.

        Args:
            bug_id: Bug ID.
            result: Result dictionary.
        """
        # Create result directory if it doesn't exist
        result_dir = self.results_dir / "bugs"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        result_file = result_dir / f"{bug_id}.json"

        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved results to {result_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
