# src/data/data_loader.py

import logging
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SWEBenchDataLoader:
    """
    Loader for SWE-bench-Verified dataset.
    """

    def __init__(self, config):
        """
        Initialize SWE-bench data loader.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.data_path = Path(config["data"]["swe_bench_path"])
        self.cache_dir = Path(config["data"]["cache_dir"])
        self.max_context_length = config["data"]["max_context_length"]

        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the SWE-bench-Verified dataset.

        Returns:
            List of dictionaries containing issue information.
        """
        # Try to find the dataset file
        # First check if file_path is specified in config
        if "file_path" in self.config["data"]:
            file_path = Path(self.config["data"]["file_path"])
            if file_path.exists():
                logger.info(f"Loading dataset from specified file_path: {file_path}")
                return self._load_json_dataset(file_path)
        
        # Then check the swe_bench_path
        dataset_path = self.data_path
        
        # Try different file extensions and formats
        possible_paths = [
            # Original paths
            dataset_path / "swe-bench-verified.jsonl",
            dataset_path / "swe-bench-verified.json",
            dataset_path / "swe_bench_verified.jsonl",
            dataset_path / "swe_bench_verified.json",
            dataset_path / "swe_bench_lite.json",
            dataset_path,
            # Additional paths to check
            Path("data/datasets/swe_bench_verified.json"),
            Path("data/datasets/swe_bench_lite.json"),
            Path("data/datasets/swe-bench-verified.json"),
            # Check for symlinked file
            Path("src/data/swe-bench-verified/swe_bench_verified.json")
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                logger.info(f"Loading dataset from: {path}")
                if path.suffix == '.jsonl':
                    return self._load_jsonl_dataset(path)
                else:
                    return self._load_json_dataset(path)
        
        # If we get here, we couldn't find the dataset
        raise FileNotFoundError(
            f"Dataset file not found. Tried: {[str(p) for p in possible_paths]}. "
            f"Please run the download script: python -m src.scripts.download_swe_bench"
        )
    
    def _load_jsonl_dataset(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load dataset from a JSONL file."""
        issues = []
        with open(file_path, 'r') as f:
            for line in f:
                issue_data = json.loads(line)
                issues.append(issue_data)
        
        logger.info(f"Loaded {len(issues)} issues from JSONL dataset")
        return issues
    
    def _load_json_dataset(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load dataset from a JSON file."""
        with open(file_path, 'r') as f:
            issues = json.load(f)
        
        logger.info(f"Loaded {len(issues)} issues from JSON dataset")
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
            # Check both instance_id and id fields to be compatible with different dataset formats
            if issue.get("instance_id") == issue_id or issue.get("id") == issue_id:
                return issue
        return None

    def get_issue_description(self, issue: Dict[str, Any]) -> str:
        """
        Extract the description from an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the issue description.
        """
        # First check for problem_statement field (from SWE-bench dataset)
        if "problem_statement" in issue:
            return issue["problem_statement"]

        # Fall back to traditional fields if problem_statement not found
        description = issue.get("description", "")
        title = issue.get("title", "")

        if title or description:
            return f"Title: {title}\n\nDescription:\n{description}"

        # Last resort - extract from the raw issue text if available
        raw_issue = issue.get("raw_issue", "")
        if raw_issue:
            return f"Raw Issue:\n{raw_issue}"

        return "No issue description available"

    def get_codebase_context(self, issue: Dict[str, Any]) -> str:
        """
        Get context from the codebase related to an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the codebase context.
        """
        # Extract repo and file paths
        repo = issue.get("repo", "")
        repo_path = self.data_path / "repos" / repo

        # Get file paths from the issue
        file_paths = []
        if "files_modified" in issue:
            file_paths.extend(issue["files_modified"])
        if "files_created" in issue:
            file_paths.extend(issue["files_created"])
        if "files_deleted" in issue:
            file_paths.extend(issue["files_deleted"])

        # Read file contents
        context = ""
        for file_path in file_paths:
            full_path = repo_path / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                context += f"File: {file_path}\n\n{content}\n\n"

        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "...[TRUNCATED]"

        return context

    def get_solution_patch(self, issue: Dict[str, Any]) -> str:
        """
        Get the solution patch for an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the solution patch.
        """
        # Handle SWE-bench dataset format which uses patch field
        if "patch" in issue:
            return issue["patch"]
        
        # Handle traditional format
        if "solution" in issue:
            return issue["solution"]

        return ""

    def get_hints(self, issue: Dict[str, Any]) -> Optional[str]:
        """
        Extract hints from an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing hints if available, None otherwise.
        """
        # Handle SWE-bench dataset format which might include hints_text
        if "hints_text" in issue:
            return issue.get("hints_text", "")

        # Handle hints that might be in comments field
        if "comments" in issue:
            return "Comments from the issue:\n" + issue.get("comments", "")

        return None
        
    def get_test_patch(self, issue: Dict[str, Any]) -> Optional[str]:
        """
        Extract test patch from an issue.
        
        Args:
            issue: Issue dictionary.
            
        Returns:
            String containing test patch if available, None otherwise.
        """
        # Handle SWE-bench dataset format which includes test_patch
        if "test_patch" in issue:
            return issue.get("test_patch", "")
            
        return None
        
    def get_fail_to_pass_tests(self, issue: Dict[str, Any]) -> List[str]:
        """
        Extract tests that should go from failing to passing.
        
        Args:
            issue: Issue dictionary.
            
        Returns:
            List of test identifiers.
        """
        # Handle SWE-bench dataset format which includes FAIL_TO_PASS
        if "FAIL_TO_PASS" in issue:
            fail_to_pass = issue.get("FAIL_TO_PASS", "[]")
            # It might be a JSON string or already parsed
            if isinstance(fail_to_pass, str):
                try:
                    return json.loads(fail_to_pass)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse FAIL_TO_PASS as JSON: {fail_to_pass}")
                    return []
            elif isinstance(fail_to_pass, list):
                return fail_to_pass
                
        return []
        
    def get_pass_to_pass_tests(self, issue: Dict[str, Any]) -> List[str]:
        """
        Extract tests that should continue to pass.
        
        Args:
            issue: Issue dictionary.
            
        Returns:
            List of test identifiers.
        """
        # Handle SWE-bench dataset format which includes PASS_TO_PASS
        if "PASS_TO_PASS" in issue:
            pass_to_pass = issue.get("PASS_TO_PASS", "[]")
            # It might be a JSON string or already parsed
            if isinstance(pass_to_pass, str):
                try:
                    return json.loads(pass_to_pass)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse PASS_TO_PASS as JSON: {pass_to_pass}")
                    return []
            elif isinstance(pass_to_pass, list):
                return pass_to_pass
                
        return []
