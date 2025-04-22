# src/evaluation/evaluator.py

import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List
import difflib

from ..data.data_loader import SWEBenchDataLoader

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for assessing the quality of solutions.
    """

    def __init__(self, config):
        """
        Initialize the evaluator.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.metrics = config["evaluation"]["metrics"]

    def evaluate_solution(self, issue: Dict[str, Any], solution_patch: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate a solution against ground truth, including test score.

        Args:
            issue: Issue dictionary.
            solution_patch: Generated solution patch.
            ground_truth: Ground truth patch.

        Returns:
            Dictionary containing evaluation metrics.
        """
        results = {}

        # Evaluate success rate
        if "success_rate" in self.metrics:
            results["success_rate"] = self._evaluate_success_rate(solution_patch, ground_truth)

        # Evaluate code quality
        if "code_quality" in self.metrics:
            results["code_quality"] = self._evaluate_code_quality(solution_patch)

        # Evaluate patch quality
        if "patch_quality" in self.metrics:
            results["patch_quality"] = self._evaluate_patch_quality(solution_patch, ground_truth)

        # Evaluate test success (if test information is available)
        if "test_success" in self.metrics:
            results["test_success"] = self._evaluate_test_success(issue, solution_patch)

        # Compute overall score
        results["overall_score"] = self._compute_overall_score(results)

        return results

    def _evaluate_test_success(self, issue: Dict[str, Any], solution_patch: str) -> float:
        """
        Evaluate if the patch passes the tests that should be fixed.

        Args:
            issue: Issue dictionary with metadata
            solution_patch: The solution patch to test

        Returns:
            Score between 0.0 and 1.0 indicating test success
        """
        # Make sure environment is set up properly
        data_loader = SWEBenchDataLoader(self.config)
        if not data_loader.prepare_repository_for_testing(issue):
            logger.warning("Could not prepare testing environment, returning 0 test score")
            return 0.0

        # Get the tests that should pass after applying the patch
        fail_to_pass = data_loader.get_fail_to_pass_tests(issue)
        pass_to_pass = data_loader.get_pass_to_pass_tests(issue)

        if not fail_to_pass and not pass_to_pass:
            logger.info("No tests specified for this issue, skipping test evaluation")
            return 0.5  # Neutral score when no tests available

        # Apply patch and run tests
        test_results = self._run_tests(issue, solution_patch, fail_to_pass, pass_to_pass)

        # Calculate test success score
        score = 0.0

        # Weight for fail-to-pass tests (these are the most important)
        if fail_to_pass:
            fail_to_pass_success = test_results.get("passing", 0) / len(fail_to_pass)
            score += fail_to_pass_success * 0.8  # 80% of the score

        # Weight for pass-to-pass tests (these should continue passing)
        if pass_to_pass:
            pass_to_pass_success = test_results.get("pass_to_pass_maintained", 0) / len(pass_to_pass)
            score += pass_to_pass_success * 0.2  # 20% of the score
        elif fail_to_pass:  # If no pass-to-pass tests, allocate full weight to fail-to-pass
            score = test_results.get("passing", 0) / len(fail_to_pass)

        return score

    def _evaluate_success_rate(self, solution_patch: str, ground_truth: str) -> float:
        """Evaluate the success rate of the solution."""
        # Simple heuristic: compare number of changed files and lines
        solution_files = self._count_files_in_patch(solution_patch)
        ground_truth_files = self._count_files_in_patch(ground_truth)

        solution_lines = self._count_changed_lines(solution_patch)
        ground_truth_lines = self._count_changed_lines(ground_truth)

        # Calculate file similarity
        if ground_truth_files == 0:
            file_similarity = 1.0 if solution_files == 0 else 0.0
        else:
            file_similarity = max(0.0, 1.0 - abs(solution_files - ground_truth_files) / ground_truth_files)

        # Calculate line similarity
        if ground_truth_lines == 0:
            line_similarity = 1.0 if solution_lines == 0 else 0.0
        else:
            line_similarity = max(0.0, 1.0 - abs(solution_lines - ground_truth_lines) / ground_truth_lines)

        # Combine metrics
        return 0.5 * file_similarity + 0.5 * line_similarity

    def _evaluate_code_quality(self, solution_patch: str) -> float:
        """Evaluate the code quality of the solution."""
        # Simple heuristic based on patch characteristics
        score = 1.0

        # Check for potential issues

        # 1. Too many changes
        changed_lines = self._count_changed_lines(solution_patch)
        if changed_lines > 100:
            score -= 0.2

        # 2. Commented-out code
        if re.search(r'^\+\s*//.*\b(TODO|FIXME|XXX)\b', solution_patch, re.MULTILINE):
            score -= 0.1

        # 3. Print statements or debug code
        if re.search(r'^\+\s*(console\.log|print\(|System\.out|debugger|alert\()', solution_patch, re.MULTILINE):
            score -= 0.1

        # 4. Hardcoded values without explanation
        if re.search(r'^\+.*\b[0-9]{4,}\b', solution_patch, re.MULTILINE):
            score -= 0.05

        # Ensure score is within bounds
        return max(0.0, min(1.0, score))

    def _evaluate_patch_quality(self, solution_patch: str, ground_truth: str) -> float:
        """Evaluate the quality of the patch compared to ground truth."""
        # Simple heuristic based on patch similarity

        # Normalize patches
        solution_norm = self._normalize_patch(solution_patch)
        ground_truth_norm = self._normalize_patch(ground_truth)

        # Calculate similarity using difflib
        matcher = difflib.SequenceMatcher(None, solution_norm, ground_truth_norm)
        similarity = matcher.ratio()

        return similarity

    def _compute_overall_score(self, results: Dict[str, float]) -> float:
        """Compute an overall score from individual metrics."""
        if not results:
            return 0.0

        # Weights for each metric
        weights = {
            "success_rate": 0.3,
            "code_quality": 0.2,
            "patch_quality": 0.2,
            "test_success": 0.3  # Add test success with significant weight
        }

        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in results:
                weighted_sum += results[metric] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _count_files_in_patch(self, patch: str) -> int:
        """Count the number of files in a patch."""
        return len(re.findall(r'diff --git', patch))

    def _count_changed_lines(self, patch: str) -> int:
        """Count the number of changed lines in a patch."""
        additions = len(re.findall(r'^\+[^+]', patch, re.MULTILINE))
        deletions = len(re.findall(r'^-[^-]', patch, re.MULTILINE))
        return additions + deletions

    def _normalize_patch(self, patch: str) -> str:
        """Normalize a patch for comparison."""
        # Remove file headers and hunks headers
        lines = patch.split('\n')
        normalized_lines = []

        for line in lines:
            # Skip diff headers, file mode lines, and hunk headers
            if (line.startswith('diff --git') or
                    line.startswith('index ') or
                    line.startswith('--- ') or
                    line.startswith('+++ ') or
                    line.startswith('@@ ')):
                continue

            # Keep only content lines
            if line.startswith('+') or line.startswith('-') or line.startswith(' '):
                normalized_lines.append(line)

        return '\n'.join(normalized_lines)

    def _run_tests(self, issue: Dict[str, Any], solution_patch: str, fail_to_pass: List[str],
                   pass_to_pass: List[str]) -> Dict[str, Any]:
        """
        Run tests to verify if the solution patch fixes the issue.

        Args:
            issue: Issue dictionary with metadata
            solution_patch: The solution patch to apply
            fail_to_pass: List of tests that should go from failing to passing
            pass_to_pass: List of tests that should continue to pass

        Returns:
            Dictionary with test results
        """
        results = {
            "passing": 0,
            "failing": 0,
            "pass_to_pass_maintained": 0,
            "detailed_results": []
        }

        # Get repository information
        repo = issue.get("repo", "")
        repo_path = Path(self.config["data"]["repositories"]) / repo

        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {repo_path}")
            return results

        # Apply the patch to a temp branch
        try:
            import subprocess
            import tempfile

            # Create a temp file with the patch
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(solution_patch)
                patch_file = f.name

            # Create a temporary branch for testing
            branch_name = f"test-{int(time.time())}"
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Apply the patch
            apply_result = subprocess.run(
                ["git", "apply", patch_file],
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if apply_result.returncode != 0:
                logger.error(f"Failed to apply patch: {apply_result.stderr.decode('utf-8')}")
                # Clean up
                subprocess.run(["git", "checkout", "-", "-f"], cwd=repo_path)
                os.unlink(patch_file)
                return results

            # Run each test that should now pass
            for test in fail_to_pass:
                test_result = self._run_single_test(repo_path, test)
                passing = test_result.get("passing", False)

                if passing:
                    results["passing"] += 1
                else:
                    results["failing"] += 1

                results["detailed_results"].append({
                    "test": test,
                    "passing": passing,
                    "output": test_result.get("output", "")
                })

            # Run each test that should continue to pass
            for test in pass_to_pass:
                test_result = self._run_single_test(repo_path, test)
                passing = test_result.get("passing", False)

                if passing:
                    results["pass_to_pass_maintained"] += 1

                results["detailed_results"].append({
                    "test": test,
                    "passing": passing,
                    "output": test_result.get("output", "")
                })

            # Clean up
            subprocess.run(["git", "checkout", "-", "-f"], cwd=repo_path)
            os.unlink(patch_file)

            return results

        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return results

    def _run_single_test(self, repo_path: Path, test: str) -> Dict[str, Any]:
        """
        Run a single test and return the result.

        Args:
            repo_path: Path to the repository
            test: Test identifier to run

        Returns:
            Dictionary with test result information
        """
        result = {
            "passing": False,
            "output": ""
        }

        try:
            import subprocess

            # Determine test command based on repository type
            # This is very project-specific and might need customization
            if (repo_path / "pytest.ini").exists() or (repo_path / "conftest.py").exists():
                # Pytest project
                cmd = ["python", "-m", "pytest", test, "-v"]
            elif (repo_path / "setup.py").exists():
                # Standard Python package
                cmd = ["python", "-m", "unittest", test]
            else:
                # Try direct execution
                cmd = ["python", test]

            # Run the test
            test_run = subprocess.run(
                cmd,
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Capture output
            output = test_run.stdout + "\n" + test_run.stderr
            result["output"] = output

            # Check if test passed (return code 0 typically means success)
            result["passing"] = test_run.returncode == 0

            return result

        except Exception as e:
            logger.error(f"Error running test {test}: {str(e)}")
            result["output"] = f"Error: {str(e)}"
            return result
