# src/evaluators/swe_bench_evaluator.py

import os
import subprocess
import time
import tempfile
from typing import Dict, Any, Tuple, List

from src.evaluators.base_evaluator import BaseEvaluator


class SWEBenchEvaluator(BaseEvaluator):
    """Evaluator for SWE-bench problems."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SWE-bench evaluator.

        Args:
            config: Evaluator configuration
        """
        super().__init__(config)

        # Extract configuration
        self.repos_dir = config.get("repos_dir", "data/repositories")
        self.timeout = config.get("timeout", 300)  # Longer timeout for complex tests
        self.python_path = config.get("python_path", "python")
        self.venv_dir = config.get("venv_dir", "data/venvs")

    def evaluate(self, code: str, task: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Evaluate SWE-bench code by applying the patch and running tests.

        Args:
            code: Patch to apply
            task: Task details including repo info and test info

        Returns:
            Tuple of (output, errors)
        """
        # The issue might be here - checking if task is None or if it's coming from config
        if task is None:
            task = self.config.get("task")

        task = self.config.get("task")
        if not task:
            return "", "Error: Task information is required for SWE-bench evaluation"

        self.logger.info(f"Evaluating SWE-bench solution for {task.get('name')}")

        repo_info = task.get("repo_info", {})
        repo_name = repo_info.get("repo")
        base_commit = repo_info.get("base_commit")

        if not repo_name or not base_commit:
            return "", "Error: Repository information is missing"

        # Setup
        start_time = time.time()

        try:
            # 1. Prepare repository
            repo_path = self._prepare_repository(repo_name, base_commit)
            self.logger.info(f"Repository prepared at {repo_path}")

            # 2. Apply the patch
            patch_result = self._apply_patch(repo_path, code)
            if not patch_result["success"]:
                errors = f"Error applying patch: {patch_result['error']}"
                self._record_evaluation(False, time.time() - start_time)
                return "", errors

            self.logger.info("Patch applied successfully")

            # 3. Run tests
            test_info = task.get("test_info", {})
            fail_to_pass = test_info.get("fail_to_pass", [])
            pass_to_pass = test_info.get("pass_to_pass", [])

            test_results = self._run_tests(repo_path, fail_to_pass, pass_to_pass)
            output = test_results["output"]
            errors = test_results["errors"]
            test_success = test_results["success"]

            execution_time = time.time() - start_time
            self.logger.info(f"Evaluation completed in {execution_time:.2f}s, success: {test_success}")

            # Record metrics
            self._record_evaluation(test_success, execution_time)

            return output, errors

        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            execution_time = time.time() - start_time

            # Record metrics for failed execution
            self._record_evaluation(False, execution_time)

            return "", f"Evaluator Error: {str(e)}"

    def _prepare_repository(self, repo_name: str, commit_hash: str) -> str:
        """
        Prepare the repository for evaluation.

        Args:
            repo_name: Repository name (e.g., "owner/repo")
            commit_hash: Commit hash to checkout

        Returns:
            Path to the prepared repository
        """
        # Create the repositories directory if it doesn't exist
        os.makedirs(self.repos_dir, exist_ok=True)

        # Form repository path
        repo_path = os.path.join(self.repos_dir, repo_name.replace("/", "_"))

        # Clone repository if it doesn't exist
        if not os.path.exists(repo_path):
            self.logger.info(f"Cloning repository {repo_name}...")
            clone_cmd = ["git", "clone", f"https://github.com/{repo_name}", repo_path]
            subprocess.run(clone_cmd, check=True)

        # Checkout the specific commit
        self.logger.info(f"Checking out commit {commit_hash}...")
        checkout_cmd = ["git", "-C", repo_path, "checkout", commit_hash]
        subprocess.run(checkout_cmd, check=True)

        return repo_path

    def _apply_patch(self, repo_path: str, patch_code: str) -> Dict[str, Any]:
        """
        Apply a patch to the repository using git, assuming patch is valid.

        Returns:
            Dict with 'success' and optional 'error'.
        """
        result = {
            "success": False,
            "error": None
        }

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as patch_file:
                patch_file.write(patch_code)
                patch_file.flush()
                patch_path = patch_file.name

            # Try applying with -p1 (standard for diffs with a/ and b/ prefixes)
            apply_cmd = ["git", "-C", repo_path, "apply", "--whitespace=nowarn", patch_path]
            proc = subprocess.run(apply_cmd, capture_output=True, text=True)

            os.unlink(patch_path)

            if proc.returncode == 0:
                result["success"] = True
            else:
                result["error"] = proc.stderr.strip()

            return result

        except Exception as e:
            result["error"] = str(e)
            return result

    def _run_tests(self, repo_path: str, fail_to_pass: List[str], pass_to_pass: List[str]) -> Dict[str, Any]:
        """
        Run tests in the repository.

        Args:
            repo_path: Path to the repository
            fail_to_pass: Tests that should fail before and pass after the patch
            pass_to_pass: Tests that should pass both before and after the patch

        Returns:
            Dictionary with test results
        """
        results = {
            "success": False,
            "output": "",
            "errors": "",
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }

        all_tests = fail_to_pass + pass_to_pass
        results["tests_run"] = len(all_tests)
        output_lines = []
        error_lines = []

        # Run each test
        for test in all_tests:
            try:
                # Determine the test framework (unittest, pytest, etc.)
                if "pytest" in test:
                    test_cmd = ["pytest", test]
                else:
                    test_cmd = [self.python_path, "-m", "unittest", test]

                # Run the test
                proc = subprocess.run(
                    test_cmd,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                is_pass = proc.returncode == 0

                # Record test result
                test_result = f"Test {test}: {'PASS' if is_pass else 'FAIL'}"
                output_lines.append(test_result)

                if is_pass:
                    results["tests_passed"] += 1
                else:
                    results["tests_failed"] += 1
                    error_lines.append(f"Test {test} failed:")
                    error_lines.append(proc.stderr)

            except subprocess.TimeoutExpired:
                output_lines.append(f"Test {test}: TIMEOUT")
                results["tests_failed"] += 1
                error_lines.append(f"Test {test} timed out after {self.timeout} seconds")

            except Exception as e:
                output_lines.append(f"Test {test}: ERROR")
                results["tests_failed"] += 1
                error_lines.append(f"Error running test {test}: {str(e)}")

        # Check if all fail_to_pass tests now pass
        fail_to_pass_success = True
        for test in fail_to_pass:
            test_result_line = next((line for line in output_lines if line.startswith(f"Test {test}:")), "")
            if "PASS" not in test_result_line:
                fail_to_pass_success = False
                break

        # Check if all pass_to_pass tests still pass
        pass_to_pass_success = True
        for test in pass_to_pass:
            test_result_line = next((line for line in output_lines if line.startswith(f"Test {test}:")), "")
            if "PASS" not in test_result_line:
                pass_to_pass_success = False
                break

        # Set overall success
        results["success"] = fail_to_pass_success and pass_to_pass_success

        # Compile output and errors
        results["output"] = "\n".join(output_lines)
        results["errors"] = "\n".join(error_lines)

        return results



