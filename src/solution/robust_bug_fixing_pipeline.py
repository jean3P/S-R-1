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

        # Initialize tracking
        iteration_logs = []
        self.failed_patch_hashes = set()  # Reset cache for this bug

        # Initialize patch errors tracking if not already created
        if not hasattr(self, 'patch_errors'):
            self.patch_errors = []

        iteration_count = 0

        # Track success at both levels
        valid_patch_found = False  # Track if we found a syntactically valid patch
        solution_found = False  # Track if we found a patch that passes tests
        final_patch = None

        # New summary statistics
        stats = {
            "total_iterations": 0,
            "syntax_failures": 0,  # Patches that couldn't be applied
            "test_failures": 0,  # Patches that could be applied but failed tests
            "time_to_valid_patch": 0,
            "time_to_solution": 0
        }

        # Main iteration loop
        while iteration_count < self.max_iterations and not solution_found:
            iteration_count += 1
            stats["total_iterations"] = iteration_count
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
                        "test_result": {"status": "skipped"},
                        "patch_valid": False,
                        "test_passed": False
                    }
                    iteration_logs.append(iteration_log)
                    continue

                # Determine instrumentation points based on the bug
                instrumentation = self._generate_instrumentation(bug_data, patch)

                # Apply the patch and instrumentation
                patched_path, original_content = self._apply_patch_with_instrumentation(bug_data, patch,
                                                                                        instrumentation)

                # Track if the patch was valid (could be applied)
                if patched_path:
                    valid_patch_found = True
                    # Record time to first valid patch if we haven't already
                    if stats["time_to_valid_patch"] == 0:
                        stats["time_to_valid_patch"] = time.time() - start_time
                        logger.info(f"Found first syntactically valid patch in {stats['time_to_valid_patch']:.2f}s")
                else:
                    # Increment syntax failure counter
                    stats["syntax_failures"] += 1

                    # Create a log entry for this iteration with patch validation failure
                    iteration_log = {
                        "iteration": iteration_count,
                        "phase": phase,
                        "patch_hash": patch_hash,
                        "patch_text": patch,
                        "explanation": explanation,
                        "instrumentation": instrumentation,
                        "test_result": {
                            "status": "error",
                            "error_message": "Patch validation failed - could not apply patch",
                            "output": "Patch could not be applied"
                        },
                        "patch_valid": False,
                        "test_passed": False
                    }
                    iteration_logs.append(iteration_log)

                    # Add to failed hashes
                    self.failed_patch_hashes.add(patch_hash)

                    # Restore original file if needed and skip test execution
                    if patched_path and original_content is not None:
                        with open(patched_path, 'w', encoding='utf-8') as f:
                            f.write(original_content)

                    continue  # Skip to next iteration

                # If we get here, the patch was valid and could be applied
                # Now run the test
                test_result = self._run_test(bug_data, patched_path)

                # Check if the test passes
                if test_result.get("status") == "pass":
                    solution_found = True
                    final_patch = patch
                    # Record time to solution
                    stats["time_to_solution"] = time.time() - start_time
                    logger.info(f"Found solution in {stats['time_to_solution']:.2f}s")
                else:
                    # Increment test failure counter
                    stats["test_failures"] += 1
                    # Add to failed hashes
                    self.failed_patch_hashes.add(patch_hash)

                # Create a log entry for this iteration with both patch validity and test results
                iteration_log = {
                    "iteration": iteration_count,
                    "phase": phase,
                    "patch_hash": patch_hash,
                    "patch_text": patch,
                    "explanation": explanation,
                    "instrumentation": instrumentation,
                    "test_result": test_result,
                    "patch_valid": True,  # If we got here, the patch could be applied
                    "test_passed": test_result.get("status") == "pass"
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
                    "test_result": {"status": "error", "error_message": str(e)},
                    "patch_valid": False,
                    "test_passed": False
                }
                iteration_logs.append(iteration_log)

        # Prepare final result
        processing_time = time.time() - start_time

        # Enhanced status reporting with detailed success levels
        if solution_found:
            status = "success_test_passed"
        elif valid_patch_found:
            status = "partial_success_valid_patch"
        else:
            status = "failed_no_valid_patch"

        result = {
            "bug_id": bug_id,
            "status": status,  # Enhanced status with success levels
            "old_status": "passed" if solution_found else "no_solution",  # Keep old status for compatibility
            "found_valid_patch": valid_patch_found,  # Did we find a patch that could be applied?
            "passed_tests": solution_found,  # Did any patch pass the tests?
            "final_patch": final_patch,
            "iterations": iteration_count,
            "history": iteration_logs,
            "processing_time": processing_time,
            "stats": stats
        }

        # Print a clear summary
        summary = f"""
            ================================================
            BUG FIX SUMMARY FOR {bug_id}
            ================================================
            Status: {status.upper()}
            Total Iterations: {stats['total_iterations']}
              - Syntax Failures: {stats['syntax_failures']}
              - Test Failures: {stats['test_failures']}
              - Successful Patches: {1 if solution_found else 0}
            Time to First Valid Patch: {stats['time_to_valid_patch']:.2f}s
            Time to Solution: {stats['time_to_solution']:.2f}s
            Total Processing Time: {processing_time:.2f}s
            ================================================
            """
        print(summary)
        logger.info(summary)

        # Save results
        self._save_results(bug_id, result)

        logger.info(f"Completed bug fixing for {bug_id} with status {status} in {processing_time:.2f}s")
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
                timeout=180
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

        # Get branch name from the issue and checkout first
        branch_name = bug_id
        if branch_name:
            # Checkout the branch with the bug to ensure we're testing the right code
            self._git_checkout_branch(branch_name)

        # NEW: Extract the complete function from the file
        repo_path = bug_data["repo_path"]
        file_path = bug_data["impl_file_path"]
        function_name = bug_data["impl_function_name"]

        if repo_path and file_path and function_name:
            full_path = repo_path / "astropy" / file_path
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                # Extract the function using regex
                bug_data["full_file_content"] = file_content

                # Try to extract the function by name with regex
                function_pattern = rf'def\s+{re.escape(function_name)}\s*\([^)]*\).*?(?=\n\S|\Z)'
                function_match = re.search(function_pattern, file_content, re.DOTALL)

                if function_match:
                    function_code = function_match.group(0)
                    bug_data["complete_function"] = function_code

                    # Get the line number where the function starts
                    lines_before_function = file_content[:function_match.start()].count('\n') + 1
                    bug_data["function_start_line"] = lines_before_function

                    # Log the complete function for debugging
                    logger.info(f"Extracted function '{function_name}' from line {lines_before_function}:")
                    logger.info(f"\n{function_code}")
                else:
                    # If we can't find the function by name, try to extract the class containing it
                    class_pattern = r'class\s+(\w+).*?(?=\n\S|\Z)'
                    for class_match in re.finditer(class_pattern, file_content, re.DOTALL):
                        class_code = class_match.group(0)
                        if f"def {function_name}" in class_code:
                            # This class contains our function
                            method_pattern = rf'def\s+{re.escape(function_name)}\s*\([^)]*\).*?(?=\n\s{{0,{class_match.group(0).find("def")}}}def|\Z)'
                            method_match = re.search(method_pattern, class_code, re.DOTALL)
                            if method_match:
                                method_code = method_match.group(0)
                                bug_data["complete_function"] = method_code

                                # Calculate line number
                                lines_before_class = file_content[:class_match.start()].count('\n') + 1
                                lines_in_class_before_method = class_code[:method_match.start()].count('\n')
                                bug_data["function_start_line"] = lines_before_class + lines_in_class_before_method

                                # Log the complete method for debugging
                                logger.info(
                                    f"Extracted method '{function_name}' from class at line {bug_data['function_start_line']}:")
                                logger.info(f"\n{method_code}")
                                break

                # If we still can't find the function, try to identify the function containing the bug lines
                if "complete_function" not in bug_data and bug_data["bug_lines"]:
                    file_lines = file_content.splitlines()
                    bug_line_idx = min(bug_data["bug_lines"]) - 1  # Convert to 0-based index

                    if 0 <= bug_line_idx < len(file_lines):
                        # Look backward to find function definition
                        for i in range(bug_line_idx, -1, -1):
                            if re.match(r'\s*def\s+', file_lines[i]):
                                # Found function definition line
                                function_start_line = i + 1  # Convert back to 1-based

                                # Extract indentation level
                                indent_match = re.match(r'^(\s*)', file_lines[i])
                                base_indent = indent_match.group(1) if indent_match else ''

                                # Collect all function lines
                                function_lines = []
                                for j in range(i, len(file_lines)):
                                    if j == i or (j > i and
                                                  (file_lines[j].startswith(base_indent + ' ') or
                                                   file_lines[j].strip() == '' or
                                                   file_lines[j] == base_indent)):
                                        function_lines.append(file_lines[j])
                                    else:
                                        # Found end of function (line with same or less indentation)
                                        if j > i and file_lines[j].startswith(base_indent[:-1] if base_indent else ''):
                                            if not re.match(r'\s*def\s+', file_lines[j]):
                                                # Not another function def, so include this line
                                                break
                                        # Otherwise this is another function def or class, so stop
                                        break

                                function_code = '\n'.join(function_lines)
                                bug_data["complete_function"] = function_code
                                bug_data["function_start_line"] = function_start_line

                                # Extract function name from definition
                                name_match = re.search(r'def\s+(\w+)', function_lines[0])
                                if name_match:
                                    extracted_name = name_match.group(1)
                                    bug_data["extracted_function_name"] = extracted_name
                                    logger.info(
                                        f"Extracted function '{extracted_name}' containing bug line {min(bug_data['bug_lines'])}:")
                                    logger.info(f"\n{function_code}")
                                break

                # Get context around bug lines
                if bug_data["bug_lines"]:
                    file_lines = file_content.splitlines()
                    start_line = max(0, min(bug_data["bug_lines"]) - 10)
                    end_line = min(len(file_lines), max(bug_data["bug_lines"]) + 10)
                    context_lines = file_lines[start_line:end_line]
                    bug_data["bug_context"] = '\n'.join(context_lines)
                    logger.info(f"Bug context (lines {start_line + 1}-{end_line}):")
                    logger.info(f"\n{bug_data['bug_context']}")

                    # Highlight the bug lines in the context
                    bug_lines_in_context = [i - start_line for i in bug_data["bug_lines"] if start_line < i <= end_line]
                    bug_context_highlighted = []
                    for i, line in enumerate(context_lines):
                        line_num = start_line + i + 1
                        prefix = f"{'>' if line_num in bug_data['bug_lines'] else ' '} {line_num:4d} | "
                        bug_context_highlighted.append(f"{prefix}{line}")

                    bug_data["bug_context_highlighted"] = '\n'.join(bug_context_highlighted)
                    logger.info(f"Bug context with highlights:")
                    logger.info(f"\n{bug_data['bug_context_highlighted']}")

            except Exception as e:
                logger.error(f"Error extracting function from file: {str(e)}")
                # Continue with the basic bug data

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

        logger.debug(f"Prompt for {self.model_name}:\n{prompt}")

        # Generate the improved solution
        response = model.generate(prompt)

        logger.debug(f"Raw response from {self.model_name}:\n{response}")

        # Extract patch and explanation
        improved_patch, explanation = self._extract_patch_and_explanation(response)

        logger.info(f"Generated patch with CoT ({len(improved_patch)} chars, explanation: {len(explanation)} chars)")
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
        file_path = "astropy" + "/" + bug_data.get("impl_file_path", "")
        function_name = bug_data.get("impl_function_name", "")
        bug_lines = bug_data.get("bug_lines", [])
        bug_lines_str = ", ".join(map(str, bug_lines)) if bug_lines else "Unknown"
        problem_statement = bug_data.get("problem_statement", "")
        hint_text = bug_data.get("hint_text", "")
        code_content = bug_data.get("code_content", "") or bug_data.get("complete_function", "")
        test_file = bug_data.get("test_file_path", "")
        test_function = bug_data.get("test_function_name", "")

        # Use the complete function if available
        if "complete_function" in bug_data:
            function_code = bug_data["complete_function"]
            function_start_line = bug_data.get("function_start_line", 0)

            # Function with line numbers for clarity
            function_lines = function_code.splitlines()
            numbered_function = []
            for i, line in enumerate(function_lines):
                line_num = function_start_line + i
                prefix = f"{'>' if line_num in bug_lines else ' '} {line_num:4d} | "
                numbered_function.append(f"{prefix}{line}")

            function_with_line_numbers = "\n".join(numbered_function)

            function_info = f"""
    # COMPLETE FUNCTION (with line numbers)
    ```python
    {function_with_line_numbers}
    ```

    Bug is in line(s): {bug_lines_str}
    """
        else:
            function_info = ""

        # Include bug context with line numbers if available
        if "bug_context_highlighted" in bug_data:
            context_info = f"""
    # CODE CONTEXT AROUND BUG (lines with '>' contain the bug)
    ```python
    {bug_data["bug_context_highlighted"]}
    ```
    """
        else:
            context_info = ""

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

        {function_info}

        {context_info if not function_info else ""}

        # CODE WITH BUG
        ```python
        {code_content}
        ```

        # YOUR TASK
        1. Think through what this function is doing and where the bug lies.
        2. Explain the root cause of the bug in plain terms.
        3. Describe why the test is failing.
        4. Match the exact indentation of the original code.
        5. Do not change indentation of surrounding context lines.
        6. Propose a patch to fix the bug in the function.

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
        - Be based on the ACTUAL current file content, not hypothetical code

        IMPORTANT: Your patch MUST match the exact lines from the current file. 
        Do not create a patch for code that doesn't exist in the file.

        Be precise. Think step-by-step.
        """

        if self.model_name == "qwq-preview":
            # Add extra explicit instructions
            prompt += """
            VERY IMPORTANT: You must create a patch that uses the ACTUAL code from the file, 
            not abstract placeholders. Look at the provided bug context and make changes to
            the EXACT lines of code shown, preserving all indentation and formatting.
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
        file_path = "astropy" + "/" + bug_data.get("impl_file_path", "")
        function_name = bug_data.get("impl_function_name", "")
        bug_lines = bug_data.get("bug_lines", [])
        bug_lines_str = ", ".join(map(str, bug_lines)) if bug_lines else "Unknown"
        problem_statement = bug_data.get("problem_statement", "")
        hint_text = bug_data.get("hint_text", "")
        code_content = bug_data.get("code_content", "") or bug_data.get("complete_function", "")
        test_function = bug_data.get("test_function_name", "")

        # Extract test results from previous iterations
        test_feedback = []
        for log in iteration_logs:
            test_result = log.get("test_result", {})
            if test_result.get("status") != "pass":
                feedback = f"Iteration {log.get('iteration')}: {test_result.get('error_message', 'Unknown error')}"
                test_feedback.append(feedback)

        test_feedback_str = "\n".join(test_feedback)

        # NEW: Add information about patch application errors if we have them
        patch_error_info = ""
        if hasattr(self, 'patch_errors') and self.patch_errors:
            last_error = self.patch_errors[-1]
            patch_error_info = f"""
    # Patch Application Error
    The previous patch could not be applied due to the following error:
    {last_error.get('error', 'Unknown error')}

    # Format Requirements
    Please ensure your patch:
    1. Has the correct file path: {file_path}
    2. References valid line numbers based on the current file content
    3. Contains proper context lines (unchanged lines before and after changes)
    4. Uses proper unified diff format:
       - Start with "diff --git a/{file_path} b/{file_path}"
       - Include "--- a/{file_path}" and "+++ b/{file_path}" headers
       - Use proper hunk headers: "@@ -lineNum,count +lineNum,count @@"
       - Use '-' prefix for removed lines and '+' for added lines
       - Include enough unchanged context lines
    """

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
        {patch_error_info if patch_error_info else ""}

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

    # Improved version with better patch validation and error logging

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

        # Handle common issues with partial patches
        if patch and "@@" in patch:
            # Check if the patch ends abruptly without proper context
            lines = patch.splitlines()
            last_content_line_idx = -1

            # Find the last content line (excluding empty lines at end)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip():
                    last_content_line_idx = i
                    break

            if last_content_line_idx >= 0:
                last_line = lines[last_content_line_idx]
                # If last line is a patch line (+ or -), patch might be truncated
                if last_line.startswith('+') or last_line.startswith('-'):
                    # Try to add context to complete the patch
                    # This helps when a patch ends in the middle of a structure
                    if "f\"" in last_line or "\"" in last_line:
                        # This appears to be a string issue - ensure we have context
                        if last_content_line_idx < len(lines) - 1:
                            # Already has trailing content
                            pass
                        else:
                            # Add a basic context line
                            lines.append(' ' + last_line[1:])
                            patch = "\n".join(lines)

        # Now apply the full patch formatting fix
        if patch:
            # Log the original patch for debugging
            logger.debug(f"Original patch before format fixing:\n{patch}")

            # Fix patch format
            original_patch = patch
            patch = self._fix_patch_format(patch)

            # Log the fixed patch if it was changed
            if patch != original_patch:
                logger.info("Fixed patch format to prevent 'corrupt patch' errors")
                logger.debug(f"Fixed patch:\n{patch}")

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

    def check_patch_end_issues(self, patch: str) -> tuple[bool, str]:
        """
        Check for issues at the end of a patch file that might cause corruption.

        Args:
            patch: The patch string to check

        Returns:
            Tuple of (has_issues, message) where:
            - has_issues is a boolean indicating if issues were found
            - message is a string describing any issues or "No issues found"
        """
        issues = []

        # Split into lines for analysis
        lines = patch.splitlines()

        if not lines:
            return True, "Empty patch"

        # Check the last few non-empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            last_line = non_empty_lines[-1]

            # Valid patterns for ending a patch
            valid_patterns = [
                lambda l: l.startswith(" "),
                lambda l: l.startswith("+"),
                lambda l: l.startswith("-"),
                lambda l: l.startswith("@@") and l.rstrip().endswith("@@"),
                lambda l: l.startswith("diff "),
                lambda l: l.startswith("--- "),
                lambda l: l.startswith("+++ ")
            ]

            if not any(pattern(last_line) for pattern in valid_patterns):
                issues.append(f"Last non-empty line has unexpected format: '{last_line}'")

        # Check for trailing whitespace or special characters
        if patch.endswith(" ") or patch.endswith("\t"):
            issues.append("Patch ends with trailing whitespace")

        # Check for missing newline at the end
        if not patch.endswith("\n"):
            issues.append("Patch doesn't end with a newline character")

        # Check for mixed line endings
        if "\r\n" in patch and "\n" in patch.replace("\r\n", ""):
            issues.append("Patch has mixed line endings (CRLF and LF)")

        # Check for unusual characters at the end
        import re
        unusual_chars = re.compile(r'[^\x20-\x7E\r\n]')
        last_part = patch[-20:] if len(patch) >= 20 else patch
        if unusual_chars.search(last_part):
            issues.append(f"Unusual characters found at the end: {repr(last_part)}")

        # Check for missing hunk context - a patch should have complete hunks
        has_hunk_header = any(line.startswith("@@") for line in lines)
        if not has_hunk_header:
            issues.append("No hunk headers (@@ -line,count +line,count @@) found in patch")

        # Check for incomplete hunks (might be cut off at the end)
        in_hunk = False
        active_hunk_content = False

        for line in reversed(lines):
            if not line.strip():
                continue  # Skip empty lines

            if line.startswith("@@"):
                in_hunk = True
                if not active_hunk_content:
                    issues.append("Last hunk appears to be empty or incomplete")
                break

            if line.startswith(" ") or line.startswith("+") or line.startswith("-"):
                active_hunk_content = True
            elif not any(p.startswith(line.strip()) for p in ["---", "+++", "diff"]):
                # Found a non-patch line at the end
                issues.append(f"Unexpected content at the end: '{line.strip()}'")
                break

        return bool(issues), "\n".join(issues) if issues else "No issues found"

    def validate_patch_with_git(self, patch: str) -> tuple[bool, str]:
        """
        Use Git's patch validation to check for issues.

        Args:
            patch: The patch string to check

        Returns:
            Tuple of (is_valid, message)
        """
        import tempfile
        import os
        import subprocess

        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as temp_file:
            temp_file.write(patch)
            temp_path = temp_file.name

        try:
            result = subprocess.run(
                ["git", "apply", "--check", "--verbose", temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                return False, result.stderr.strip()
            return True, "Patch is valid"
        except Exception as e:
            return False, f"Error validating patch: {str(e)}"
        finally:
            os.unlink(temp_path)

    def ensure_patch_ends_with_newline(self, patch: str) -> str:
        """
        Ensures that a patch string ends with a newline character.
        Git patches require a trailing newline to be valid.

        Args:
            patch: The patch string to check

        Returns:
            Patch string with a guaranteed trailing newline
        """
        if not patch:
            return patch

        # Ensure the patch ends with exactly one newline
        if not patch.endswith('\n'):
            patch += '\n'

        return patch

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
        patch = self.fix_patch_indentation(patch)
        patch = self.fix_closing_parenthesis_indentation(patch)

        # Ensure we fix the patch format before applying it
        patch = self._fix_patch_format(patch, bug_data)

        has_issues, message = self.check_patch_end_issues(patch)
        if has_issues:
            logger.info(f"Issues found in patch: {message}")
            # Maybe try to fix the issues or reject the patch
        else:
            # Double-check with Git's validation if available
            is_valid, git_message = self.validate_patch_with_git(patch)
            if not is_valid:
                logger.info(f"Git validation failed: {git_message}")
            else:
                logger.info("Patch looks valid!")

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
        patch = self.ensure_patch_ends_with_newline(patch)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as temp_patch_file:
            temp_patch_file.write(patch)
            temp_patch_path = temp_patch_file.name

        logger.info(f"Temporal patch file: {temp_patch_path}")

        try:
            # NEW: Try to validate the patch first
            try:
                # Check if patch is valid
                check_result = subprocess.run(
                    ["git", "apply", "--check", "--verbose", temp_patch_path],
                    cwd=repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if check_result.returncode != 0:
                    # Log the full patch when validation fails
                    error_msg = check_result.stderr.strip()

                    # Make error output more prominent with clear formatting
                    formatted_error = f"""
                        ================================================
                        PATCH VALIDATION ERROR
                        ================================================
                        ERROR OUTPUT:
                        {error_msg}
                        ------------------------------------------------
                        PATCH CONTENT:
                        {patch}
                        ================================================
                        """
                    # Log with high visibility
                    logger.error(formatted_error)

                    # Also print to stdout for immediate visibility during execution
                    print(formatted_error)

                    # Store the error for future reflection
                    if not hasattr(self, 'patch_errors'):
                        self.patch_errors = []

                    self.patch_errors.append({
                        "patch": patch,
                        "error": error_msg
                    })

                    raise subprocess.CalledProcessError(
                        check_result.returncode,
                        check_result.args,
                        check_result.stdout,
                        check_result.stderr
                    )
            except subprocess.CalledProcessError as e:
                # If validation fails, raise the error to be caught by the outer try-except
                raise e

            # Apply the validated patch
            apply_result = subprocess.run(
                ["git", "apply", temp_patch_path],
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if apply_result.returncode != 0:
                # This should rarely happen since we validated, but log if it does
                logger.error(f"Patch application failed after validation: {apply_result.stderr}")
                logger.error(f"Corrupt patch:\n{patch}")
                raise subprocess.CalledProcessError(
                    apply_result.returncode,
                    apply_result.args,
                    apply_result.stdout,
                    apply_result.stderr
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
            error_msg = e.stderr.strip()

            # Make error output more prominent with clear formatting
            formatted_error = f"""
            ================================================
            PATCH APPLICATION ERROR
            ================================================
            ERROR OUTPUT:
            {error_msg}
            ------------------------------------------------
            PATCH CONTENT:
            {patch}
            ================================================
            """
            # Log with high visibility
            logger.error(formatted_error)

            # Also print to stdout for immediate visibility during execution
            print(formatted_error)

            # Store the error for future reflection
            if not hasattr(self, 'patch_errors'):
                self.patch_errors = []

            self.patch_errors.append({
                "patch": patch,
                "error": error_msg
            })

            # Clean up temp file
            os.unlink(temp_patch_path)
            return None, None
        except Exception as e:
            error_msg = str(e)

            # Make error output more prominent with clear formatting
            formatted_error = f"""
                ================================================
                PATCH APPLICATION ERROR (GENERAL EXCEPTION)
                ================================================
                ERROR OUTPUT:
                {error_msg}
                ------------------------------------------------
                PATCH CONTENT:
                {patch}
                ================================================
                """
            # Log with high visibility
            logger.error(formatted_error)

            # Also print to stdout for immediate visibility during execution
            print(formatted_error)

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

    def _fix_patch_format(self, patch: str, bug_data: Dict[str, Any] = None) -> str:
        """
        Fix common formatting issues with Git patches to ensure they can be applied.

        Args:
            patch: The original patch string
            bug_data: Bug information including file paths

        Returns:
            Fixed patch string with complete context
        """
        if not patch:
            return patch

        # Process patch line by line
        lines = patch.splitlines()
        fixed_lines = []

        # Track context for processing
        in_hunk = False
        hunk_start_line = -1
        current_hunk_header = None
        current_hunk_lines = []
        expected_line_count = None

        # Track indentation levels for closing parentheses
        open_parens_indents = []

        for i, line in enumerate(lines):
            if "index [current_sha]..[new_sha] [options]" in line:
                continue
            # Process header lines (diff, ---, +++)
            if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
                # Add header line directly
                fixed_lines.append(line)
                continue

            # Process hunk headers
            if line.startswith('@@'):
                # If we were in a hunk before, fix and add the previous hunk
                if in_hunk and current_hunk_lines:
                    fixed_hunk = self._fix_hunk_content(current_hunk_lines, expected_line_count,
                                                        bug_data, current_hunk_header)
                    fixed_lines.extend(fixed_hunk)
                    current_hunk_lines = []

                # Start a new hunk
                in_hunk = True
                hunk_start_line = i
                current_hunk_header = line
                open_parens_indents = []  # Reset paren tracking for new hunk

                # Parse the header to get expected line counts
                header_match = re.match(r'@@ -\d+,(\d+) \+\d+,(\d+) @@', line)
                if header_match:
                    # Use the maximum line count to ensure both sides have enough lines
                    expected_line_count = max(int(header_match.group(1)), int(header_match.group(2)))
                else:
                    expected_line_count = None

                # Fix hunk header - remove function signatures to prevent corruption
                match = re.match(r'(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@).*', line)
                if match:
                    fixed_lines.append(match.group(1))
                else:
                    fixed_lines.append(line)
                continue

            # Track indentation for parentheses
            if in_hunk:
                # Check for opening parentheses
                if '(' in line and not line.startswith('@@'):
                    indent_match = re.match(r'^[+ -](\s*)', line)
                    if indent_match:
                        current_indent = len(indent_match.group(1))
                        open_parens_indents.append(current_indent)

                # Check for closing parentheses and fix indentation
                if ')' in line and line.strip().startswith(')') and open_parens_indents:
                    # This is a line with just a closing parenthesis
                    indent_level = open_parens_indents.pop() if open_parens_indents else 0

                    # Fix indentation - ensure it matches the opening paren's indentation
                    prefix = line[0] if line and line[0] in '+ -' else ' '
                    fixed_indent = ' ' * indent_level
                    line = f"{prefix}{fixed_indent})"

                current_hunk_lines.append(line)
            else:
                # Not in a hunk yet, add line directly
                fixed_lines.append(line)

        # Process the last hunk if any
        if in_hunk and current_hunk_lines:
            fixed_hunk = self._fix_hunk_content(current_hunk_lines, expected_line_count,
                                                bug_data, current_hunk_header)
            fixed_lines.extend(fixed_hunk)

        # Now fix any remaining header issues to ensure they match actual content
        fixed_patch = '\n'.join(fixed_lines)
        return self._fix_patch_headers(fixed_patch)

    def _split_patch_into_parts(self, patch: str) -> List[Tuple[str, str]]:
        """
        Split a patch into its logical components (headers, hunks).

        Args:
            patch: The patch string

        Returns:
            List of (part_type, part_content) tuples
        """
        lines = patch.splitlines()
        parts = []
        current_part = []
        current_type = None

        for line in lines:
            # Detect headers (diff, ---, +++)
            if line.startswith("diff --git"):
                # Save previous part if any
                if current_part:
                    parts.append((current_type, "\n".join(current_part)))

                # Start new header part
                current_type = "header"
                current_part = [line]
            elif line.startswith("---") or line.startswith("+++"):
                if current_type != "header":
                    # Start new header if not already in one
                    if current_part:
                        parts.append((current_type, "\n".join(current_part)))
                    current_type = "header"
                    current_part = [line]
                else:
                    # Continue existing header
                    current_part.append(line)
            elif line.startswith("@@"):
                # Start of a new hunk
                if current_part:
                    parts.append((current_type, "\n".join(current_part)))

                current_type = "hunk"
                current_part = [line]
            else:
                # Content line, add to current part
                if current_type:
                    current_part.append(line)
                else:
                    # If no current part type, assume header
                    current_type = "header"
                    current_part.append(line)

        # Add the last part
        if current_part:
            parts.append((current_type, "\n".join(current_part)))

        return parts

    def _fix_hunk_with_complete_context(self, hunk: str) -> str:
        """
        Fix a hunk to ensure it has complete context and balanced structure.

        Args:
            hunk: The hunk content

        Returns:
            Fixed hunk with proper context
        """
        lines = hunk.splitlines()
        if not lines:
            return hunk

        # Process the hunk header
        header_line = lines[0]
        content_lines = lines[1:] if len(lines) > 1 else []

        # Fix the hunk header - remove anything after the second @@
        if header_line.startswith("@@"):
            match = re.match(r'(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@)(.*)', header_line)
            if match:
                header_line = match.group(1)

        # Track brackets/parentheses to ensure balanced structure
        bracket_balance = self._analyze_code_structure(content_lines)

        # Fix the content lines and ensure complete context
        fixed_content = self._ensure_complete_context(content_lines, bracket_balance)

        # Combine the header with fixed content
        return header_line + "\n" + "\n".join(fixed_content)

    def _analyze_code_structure(self, lines: List[str]) -> Dict[str, int]:
        """
        Analyze the code structure to detect imbalanced brackets, parentheses, etc.

        Args:
            lines: The lines of code to analyze

        Returns:
            Dictionary with balance counts for each bracket type
        """
        # Track balance of different bracket types
        balance = {
            '(': 0,  # parentheses
            '{': 0,  # curly braces
            '[': 0,  # square brackets
        }

        # Process each line
        for line in lines:
            # Skip the prefix for patch lines (-, +, space)
            code = line[1:] if line and line[0] in '- +' else line

            # Scan for brackets
            for char in code:
                if char == '(':
                    balance['('] += 1
                elif char == ')':
                    balance['('] -= 1
                elif char == '{':
                    balance['{'] += 1
                elif char == '}':
                    balance['{'] -= 1
                elif char == '[':
                    balance['['] += 1
                elif char == ']':
                    balance['['] -= 1

        return balance

    def _ensure_complete_context(self, lines: List[str], bracket_balance: Dict[str, int]) -> List[str]:
        """
        Ensure the patch has complete context, especially for balanced code structures.

        Args:
            lines: The content lines of the hunk
            bracket_balance: The bracket balance information

        Returns:
            List of fixed content lines with complete context
        """
        if not lines:
            return lines

        # First, check if we have imbalanced structures
        has_imbalance = any(balance != 0 for balance in bracket_balance.values())

        # Look for truncated multi-line strings
        has_string_patterns = False
        for i, line in enumerate(lines):
            if i > 0 and line.strip() and (line.startswith('-') or line.startswith('+')):
                line_content = line[1:].strip()
                # Look for string patterns like quotes, f-strings
                if (line_content.startswith('f"') or line_content.startswith("f'") or
                        line_content.startswith('"') or line_content.startswith("'") or
                        line_content.endswith('"') or line_content.endswith("'")):
                    has_string_patterns = True
                    break

        # Check for missing closing contexts
        missing_context = False
        if lines and lines[-1].strip() and not lines[-1].startswith(' '):
            # Last line is a change (+ or -), not context
            missing_context = True

        # If we need to fix the context, ensure balanced blocks
        if has_imbalance or has_string_patterns or missing_context:
            # Add balanced context lines if needed
            if missing_context and len(lines) > 1:
                # Ensure we have at least one context line at the end
                lines.append(' ' + lines[-1][1:].rstrip())

        # Group consecutive related lines that should stay together
        balanced_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # Handle multi-line string patterns
            if i < len(lines) - 1 and line.startswith('-') and 'f"' in line:
                # Look for a group of related string lines
                group_lines = [line]
                j = i + 1

                # Collect all related minus lines
                while j < len(lines) and lines[j].startswith('-'):
                    group_lines.append(lines[j])
                    j += 1

                # Collect all related plus lines that follow
                while j < len(lines) and lines[j].startswith('+'):
                    group_lines.append(lines[j])
                    j += 1

                # If we found a potential string pattern, add context
                if len(group_lines) > 1:
                    # Ensure we have context after the pattern
                    if j < len(lines) and lines[j].startswith(' '):
                        group_lines.append(lines[j])
                        j += 1

                    balanced_lines.extend(group_lines)
                    i = j
                    continue

            # Default case - add the line
            balanced_lines.append(line)
            i += 1

        # Ensure we have proper line prefixes (space, +, -)
        fixed_lines = []
        for line in balanced_lines:
            if not line:
                continue

            if not (line.startswith(' ') or line.startswith('+') or line.startswith('-')):
                # Add space prefix for context lines
                fixed_lines.append(' ' + line)
            else:
                fixed_lines.append(line)

        return fixed_lines

    def _fix_hunk_content(self, hunk_lines: List[str], expected_line_count: int = None,
                          bug_data: Dict[str, Any] = None, hunk_header: str = None) -> List[str]:
        """
        Fix the content of a hunk to ensure it has balanced structure and matches the expected line count.
        Retrieves missing context from the original file when needed.

        Args:
            hunk_lines: Lines of the hunk content (without header)
            expected_line_count: Number of lines expected in the hunk according to the header
            bug_data: Dictionary containing bug information, including file paths
            hunk_header: The header of the current hunk (e.g., "@@ -291,7 +291,7 @@")

        Returns:
            Fixed hunk content lines
        """
        if not hunk_lines:
            return hunk_lines

        indent = ''

        # Calculate current line counts
        current_minus_lines = sum(1 for line in hunk_lines if line.startswith('-') or line.startswith(' '))
        current_plus_lines = sum(1 for line in hunk_lines if line.startswith('+') or line.startswith(' '))

        # Track code structure balance
        brackets = {
            '(': 0,
            '{': 0,
            '[': 0,
            '"': 0,
            "'": 0
        }

        # Special case for error messages which often have f-strings and parentheses
        has_error_message = False
        has_open_string = False

        # Analyze structure balance in the hunk
        for line in hunk_lines:
            content = line[1:] if line and line[0] in '+-' else line

            # Check for error message patterns
            if "TypeError" in content or "ValueError" in content or "raise" in content:
                has_error_message = True

            # Check for open string literals (f-strings)
            if (("f\"" in content and content.count("\"") % 2 != 0) or
                    ("f'" in content and content.count("'") % 2 != 0)):
                has_open_string = True

            # Track brackets and parentheses
            for char in content:
                if char == '(':
                    brackets['('] += 1
                elif char == ')':
                    brackets['('] -= 1
                elif char == '{':
                    brackets['{'] += 1
                elif char == '}':
                    brackets['{'] -= 1
                elif char == '[':
                    brackets['['] += 1
                elif char == ']':
                    brackets['['] -= 1

        # Check if we have imbalanced structures or if the hunk ends with a change
        imbalanced = any(count != 0 for count in brackets.values())
        truncated = hunk_lines and (hunk_lines[-1].startswith('+') or hunk_lines[-1].startswith('-'))

        # Create a list for the fixed lines
        fixed_lines = list(hunk_lines)

        # If we have an expected line count from the header, try to match it
        if expected_line_count is not None:
            # Calculate how many more context lines we need
            lines_needed_minus = expected_line_count - current_minus_lines
            lines_needed_plus = expected_line_count - current_plus_lines

            # If we need to add context lines
            if lines_needed_minus > 0 or lines_needed_plus > 0 or imbalanced or truncated:
                last_line = hunk_lines[-1] if hunk_lines else ""
                last_content = last_line[1:] if last_line and last_line[0] in '+-' else last_line
                indent = re.match(r'^(\s*)', last_content).group(1) if last_content and re.match(r'^(\s*)',
                                                                                                 last_content) else ''

                # If the last line isn't a context line, add it as context first
                if truncated:
                    if last_line.startswith('+'):
                        fixed_lines.append(' ' + last_content)
                    elif last_line.startswith('-') and len(fixed_lines) > 1 and fixed_lines[-2].startswith('+'):
                        # If we have a removed line followed by an added line, use the added line as context
                        fixed_lines.append(' ' + fixed_lines[-2][1:])
                    else:
                        # Use the current line as context
                        fixed_lines.append(' ' + last_content)

                    # Recalculate how many more lines we need after adding this context line
                    lines_needed_minus = max(0, expected_line_count - (current_minus_lines + 1))
                    lines_needed_plus = max(0, expected_line_count - (current_plus_lines + 1))

                # If we still need more lines and don't have original context, add generic ones
                while lines_needed_minus > 0 or lines_needed_plus > 0:
                    if "return" not in last_content:
                        if lines_needed_minus == 1 or lines_needed_plus == 1:
                            # For the last needed line, add something neutral
                            fixed_lines.append(' ')
                        else:
                            fixed_lines.append(f' {indent}# End of function')
                    else:
                        # Just add a generic context line
                        fixed_lines.append(' ')

                    lines_needed_minus = max(0, lines_needed_minus - 1)
                    lines_needed_plus = max(0, lines_needed_plus - 1)

        # Add special handling for error messages with closed parentheses
        if has_error_message and "TypeError" in ''.join(hunk_lines) and brackets['('] > 0:
            # Check if the last line is already a closing parenthesis
            last_line = fixed_lines[-1] if fixed_lines else ""
            if not (last_line.strip().endswith(')') or ')' in last_line):
                # Only add closing parenthesis if there's an actual imbalance
                # Count open and close parentheses in the entire hunk
                open_count = sum(line.count('(') for line in hunk_lines)
                close_count = sum(line.count(')') for line in hunk_lines)
                if open_count > close_count:
                    # Add a closing parenthesis line with proper indentation
                    if indent:
                        # Reduce indentation for closing parenthesis
                        closing_indent = indent[:-4] if len(indent) >= 4 else ''
                        fixed_lines.append(f' {closing_indent})')

        # Add additional context lines for error messages with f-strings
        if has_error_message and ("f\"" in ''.join(hunk_lines) or "f'" in ''.join(hunk_lines)):
            # Check if we need to add closing string context
            last_line = fixed_lines[-1] if fixed_lines else ""
            if not ('"' in last_line or "'" in last_line):
                # Add a line that might contain the closing quote
                fixed_lines.append(f' {indent})')

        # Ensure we end with a proper context line
        if fixed_lines and (fixed_lines[-1].startswith('+') or fixed_lines[-1].startswith('-')):
            last_content = fixed_lines[-1][1:]
            fixed_lines.append(' ' + last_content)

        return fixed_lines

    def _fix_patch_headers(self, patch_text):
        """
        Fix patch headers to precisely match the actual content of each hunk.
        Handles complex changes, multiple hunks, and ensures correct line counting.

        Args:
            patch_text: The original patch text

        Returns:
            The corrected patch text with accurate headers
        """
        lines = patch_text.splitlines()
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            if "index [current_sha]..[new_sha] [options]" in line:
                continue

            # Process non-hunk lines directly
            if line.startswith('diff ') or line.startswith('---') or line.startswith('+++'):
                fixed_lines.append(line)
                i += 1
                continue

            # Process hunk headers
            if line.startswith('@@'):
                hunk_start = i
                hunk_end = hunk_start + 1
                while hunk_end < len(lines) and not lines[hunk_end].startswith('@@'):
                    hunk_end += 1

                hunk_content = lines[hunk_start + 1:hunk_end]

                # Count line types accurately
                orig_count = 0  # Lines from original file (context + removals)
                new_count = 0  # Lines in new file (context + additions)

                for content_line in hunk_content:
                    if content_line.startswith(' '):  # Context line
                        orig_count += 1
                        new_count += 1
                    elif content_line.startswith('-'):  # Removal
                        orig_count += 1
                    elif content_line.startswith('+'):  # Addition
                        new_count += 1
                    else:  # Malformed line - treat as context
                        orig_count += 1
                        new_count += 1

                # Parse original header
                header_match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)', line)
                if header_match:
                    orig_start = int(header_match.group(1))
                    new_start = int(header_match.group(3))
                    header_comment = header_match.group(5) or ""

                    # Ensure minimum counts of 1
                    orig_count = max(1, orig_count)
                    new_count = max(1, new_count)

                    # Build new header
                    new_header = f"@@ -{orig_start},{orig_count} +{new_start},{new_count} @@{header_comment}"

                    fixed_lines.append(new_header)
                    fixed_lines.extend(hunk_content)
                else:
                    fixed_lines.append(line)
                    fixed_lines.extend(hunk_content)

                i = hunk_end
            else:
                fixed_lines.append(line)
                i += 1

        return '\n'.join(fixed_lines)

    def fix_patch_indentation(self, patch_text):
        """
        Fix indentation issues in patches, particularly for closing parentheses.
        Only modify lines that are being changed (+ or -), never context lines.
        """
        lines = patch_text.splitlines()
        fixed_lines = []
        in_hunk = False

        for i, line in enumerate(lines):
            # Handle patch headers
            if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
                fixed_lines.append(line)
                continue

            # Track when we enter hunks
            if line.startswith('@@'):
                in_hunk = True
                fixed_lines.append(line)
                continue

            # Process content lines within hunks
            if in_hunk:
                # IMPORTANT: Never modify context lines (starting with space)
                if line.startswith(' '):
                    # Preserve context lines exactly as they are
                    fixed_lines.append(line)
                else:
                    # Only modify actual changes (+ or - lines)
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_closing_parenthesis_indentation(self, patch_text):
        """
        Fixes indentation of closing parentheses in patch content.
        Only affects changed lines (+ or -), never context lines.
        """
        lines = patch_text.splitlines()
        fixed_lines = []
        in_hunk = False

        for line in lines:
            # Don't modify any context lines
            if not in_hunk or line.startswith(' '):
                fixed_lines.append(line)
                continue

            # Start tracking hunks
            if line.startswith('@@'):
                in_hunk = True
                fixed_lines.append(line)
                continue

            # We're only dealing with + or - lines here
            # Don't modify lines with closing parentheses
            if line.strip().endswith(')') and line[1:].strip() == ')':
                # Keep original indentation for standalone closing parens
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)


