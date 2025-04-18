# data/data_loader.py

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
        dataset_path = self.data_path / "swe-bench-verified.jsonl"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

        issues = []
        with open(dataset_path, 'r') as f:
            for line in f:
                issue_data = json.loads(line)
                issues.append(issue_data)

        logger.info(f"Loaded {len(issues)} issues from SWE-bench-Verified dataset")
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
            if issue.get("id") == issue_id:
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
        description = issue.get("description", "")
        title = issue.get("title", "")
        return f"Title: {title}\n\nDescription:\n{description}"

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
        if "solution" in issue:
            return issue["solution"]

        # If solution is not directly available, try to find it in the patch
        if "patch" in issue:
            return issue["patch"]

        return ""
