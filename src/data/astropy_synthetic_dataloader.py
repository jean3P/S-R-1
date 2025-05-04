# src/data/astropy_synthetic_dataloader.py

import logging
import os
import re
import csv
from typing import Dict, List, Any, Optional
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


class AstropySyntheticDataLoader:
    """
    Loader for Astropy Synthetic Dataset.
    This loader is designed to work with the CSV dataset created by the synthetic test generator.
    """

    def __init__(self, config):
        """
        Initialize Astropy Synthetic data loader.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.data_path = Path(config["data"]["astropy_dataset_path"])
        self.cache_dir = Path(config["data"]["cache_dir"])

        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the Astropy Synthetic dataset.

        Returns:
            List of dictionaries containing issue information.
        """
        # Try to find the dataset file
        # First check if file_path is specified in config
        if "file_path" in self.config["data"]:
            file_path = Path(self.config["data"]["file_path"])
            if file_path.exists():
                logger.info(f"Loading dataset from specified file_path: {file_path}")
                return self._load_csv_dataset(file_path)

        # Then check the astropy_dataset_path
        dataset_path = self.data_path

        # Try different file extensions and formats
        possible_paths = [
            Path("/storage/homefs/jp22b083/SSI/S-R-1/src/data/astropy_implementation_bugs_dataset.csv"),
            ]

        for path in possible_paths:
            if path.exists() and path.is_file():
                logger.info(f"Loading dataset from: {path}")
                return self._load_csv_dataset(path)

        raise FileNotFoundError(
            f"Dataset file not found. Tried: {[str(p) for p in possible_paths]}. "
            f"Please run the synthetic dataset generator script first."
        )

    def _load_csv_dataset(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load dataset from a CSV file."""
        issues = []
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add an instance_id field for compatibility with SWE-bench loader
                row["instance_id"] = f"astropy-{row.get('branch_name', 'unknown')}"
                row["repo"] = os.path.dirname(row["Path_repo"])
                issues.append(row)

        logger.info(f"Loaded {len(issues)} issues from CSV dataset")
        return issues

    def load_issue(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific issue by ID.

        Args:
            issue_id: ID of the issue to load.

        Returns:
            Dictionary containing issue information.
        """
        issues = self.load_dataset()
        for issue in issues:
            # Check both instance_id and branch_name fields to be compatible
            if issue.get("instance_id") == issue_id or issue.get("branch_name") == issue_id:
                return issue

        logger.warning(f"Issue {issue_id} not found in dataset")
        return None

    def get_issue_description(self, issue: Dict[str, Any]) -> str:
        """
        Extract the description from an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the issue description.
        """
        # First check for problem_statement field (from our synthetic dataset)
        if "problem_statement" in issue and issue["problem_statement"]:
            logger.debug(f"Using problem_statement field ({len(issue['problem_statement'])} chars)")
            return issue["problem_statement"]

        # Absolute fallback: use ID information to create a minimal description
        repo = issue.get("repo", "unknown")
        issue_id = issue.get("instance_id", "unknown")
        branch_name = issue.get("branch_name", "unknown")
        logger.warning(f"No description found for issue {issue_id}. Using minimal description.")

        # Create a minimal description with whatever information we have
        fallback = f"Fix issue in repository {repo}, branch: {branch_name}. "
        fallback += "Examine the codebase to identify and fix the bug in the test. "

        return fallback

    def get_solution_patch(self, issue: Dict[str, Any]) -> str:
        """
        Get the solution patch for an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the solution patch.
        """
        # Get the solution patch from the GT_test_patch field
        if "GT_test_patch" in issue and issue["GT_test_patch"]:
            return issue["GT_test_patch"]

        logger.warning(f"No solution patch found for issue {issue.get('instance_id', 'unknown')}")
        return ""

    def get_hints(self, issue: Dict[str, Any]) -> Optional[str]:
        """
        Extract hints from an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing hints if available, None otherwise.
        """
        # Get hints from hint_text field
        if "hint_text" in issue:
            return issue.get("hint_text", "")

        return None

    def get_failed_code(self, issue: Dict[str, Any]) -> str:
        """
        Get the failing code for an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the failing code.
        """
        if "FAIL_TO_PASS" in issue and issue["FAIL_TO_PASS"]:
            return issue["FAIL_TO_PASS"]

        logger.warning(f"No failing code found for issue {issue.get('instance_id', 'unknown')}")
        return ""

    def get_passing_code(self, issue: Dict[str, Any]) -> str:
        """
        Get the passing code (ground truth) for an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the passing code.
        """
        if "PASS_TO_PASS" in issue and issue["PASS_TO_PASS"]:
            return issue["PASS_TO_PASS"]

        logger.warning(f"No passing code found for issue {issue.get('instance_id', 'unknown')}")
        return ""

    def get_complexity(self, issue: Dict[str, Any]) -> str:
        """
        Get the complexity level of an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the complexity level.
        """
        if "complexity" in issue:
            return issue.get("complexity", "unknown")

        return "unknown"

    def checkout_issue_branch(self, issue: Dict[str, Any]) -> bool:
        """
        Checkout the branch for an issue to prepare for testing.

        Args:
            issue: Issue dictionary.

        Returns:
            Boolean indicating success.
        """
        repo_path = issue.get("repo", "")
        branch_name = issue.get("branch_name", "")

        if not branch_name:
            logger.warning(f"No branch name specified for issue {issue.get('instance_id', '')}")
            return False

        if not repo_path or not os.path.exists(repo_path):
            logger.error(f"Repository path does not exist: {repo_path}")
            return False

        # Checkout the branch for this issue
        try:
            subprocess.run(
                ["git", "checkout", branch_name, "-f"],
                cwd=repo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Successfully checked out branch {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout branch {branch_name}: {e}")
            return False

    def get_python_env_path(self, issue: Dict[str, Any]) -> str:
        """
        Get the Python environment path for an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the Python environment path.
        """
        if "path_env" in issue and issue["path_env"]:
            return issue["path_env"]

        logger.warning(f"No Python environment path found for issue {issue.get('instance_id', 'unknown')}")
        return "python"  # Default to system Python

    def find_failing_test(self, issue: Dict[str, Any]) -> str:
        """
        Extract the test function name and file path from the failing code.

        Args:
            issue: Issue dictionary.

        Returns:
            String in the format "file_path::function_name"
        """
        failing_code = self.get_failed_code(issue)

        # Extract function name
        match = re.search(r"def\s+(\w+)\s*\(", failing_code)
        if not match:
            logger.warning(f"Could not extract function name from failing code")
            return ""

        function_name = match.group(1)

        # Get the repository path and branch information
        repo_path = issue.get("repo", "")
        branch_name = issue.get("branch_name", "")

        if not branch_name:
            logger.warning(f"No branch name specified for issue {issue.get('instance_id', '')}")
            return ""

        # Extract test file using git diff
        try:
            # Checkout the branch if not already on it
            self.checkout_issue_branch(issue)

            # Get diff to find the modified file
            result = subprocess.run(
                ["git", "show", "--name-only", "--pretty=format:", branch_name],
                cwd=repo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Parse output to find test file
            files = result.stdout.strip().split('\n')
            test_files = [f for f in files if f.endswith('.py') and 'test' in f]

            if not test_files:
                logger.warning(f"Could not find test file in branch {branch_name}")
                return ""

            # Use the first test file
            test_file = test_files[0]

            # Return file::function format
            return f"{test_file}::{function_name}"

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract test file from branch {branch_name}: {e}")
            return ""

    def run_failing_test(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the failing test for an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            Dictionary with test results.
        """
        # Get the Python environment path
        python_path = self.get_python_env_path(issue)

        # Find the failing test
        test_path = self.find_failing_test(issue)

        if not test_path:
            logger.warning(f"Could not determine test path for issue {issue.get('instance_id', 'unknown')}")
            return {"success": False, "output": "Could not determine test path"}

        # Get repository path
        repo_path = issue.get("repo", "")

        if not repo_path or not os.path.exists(repo_path):
            logger.error(f"Repository path does not exist: {repo_path}")
            return {"success": False, "output": f"Repository path does not exist: {repo_path}"}

        # Checkout the branch if not already on it
        self.checkout_issue_branch(issue)

        # Run the test
        try:
            result = subprocess.run(
                [python_path, "-m", "pytest", test_path, "-v"],
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            return {
                "success": result.returncode == 0,
                "output": result.stdout + "\n" + result.stderr,
                "returncode": result.returncode
            }

        except Exception as e:
            logger.error(f"Failed to run test {test_path}: {e}")
            return {"success": False, "output": f"Failed to run test: {str(e)}"}

    def prepare_issue_for_pipeline(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare an issue for processing in a pipeline.

        This method enriches the issue with additional information
        and formats it for processing in a code reflection pipeline.

        Args:
            issue: Issue dictionary.

        Returns:
            Enhanced issue dictionary.
        """
        # Create a new dictionary with the essential information
        pipeline_issue = {
            "id": issue.get("instance_id", ""),
            "branch_name": issue.get("branch_name", ""),
            "repo_path": issue.get("repo", ""),
            "env_path": issue.get("path_env", ""),
            "problem": self.get_issue_description(issue),
            "complexity": self.get_complexity(issue),
            "failing_code": self.get_failed_code(issue),
            "passing_code": self.get_passing_code(issue),
            "solution_patch": self.get_solution_patch(issue),
            "hint": self.get_hints(issue),
        }

        # Add information about the test
        test_path = self.find_failing_test(issue)
        if test_path:
            parts = test_path.split("::")
            pipeline_issue["test_file"] = parts[0] if len(parts) > 0 else ""
            pipeline_issue["test_function"] = parts[1] if len(parts) > 1 else ""

        # Run the failing test to confirm it fails
        test_result = self.run_failing_test(issue)
        pipeline_issue["test_fails"] = not test_result["success"]
        pipeline_issue["test_output"] = test_result["output"]

        return pipeline_issue
