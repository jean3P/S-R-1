# utils/git_utils.py
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class GitUtils:
    """
    Utilities for Git operations.
    """

    @staticmethod
    def apply_patch(repo_path: str, patch_path: str) -> bool:
        """
        Apply a patch to a repository.

        Args:
            repo_path: Path to the repository.
            patch_path: Path to the patch file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Run git apply
            result = subprocess.run(
                ["git", "apply", patch_path],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error applying patch: {e.stderr}")
            return False

    @staticmethod
    def create_branch(repo_path: str, branch_name: str) -> bool:
        """
        Create a new branch in a repository.

        Args:
            repo_path: Path to the repository.
            branch_name: Name of the branch to create.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Create and checkout branch
            result = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating branch: {e.stderr}")
            return False

    @staticmethod
    def commit_changes(repo_path: str, message: str) -> bool:
        """
        Commit changes in a repository.

        Args:
            repo_path: Path to the repository.
            message: Commit message.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Add all changes
            subprocess.run(
                ["git", "add", "."],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            # Commit changes
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error committing changes: {e.stderr}")
            return False

    @staticmethod
    def get_modified_files(repo_path: str) -> List[str]:
        """
        Get list of modified files in a repository.

        Args:
            repo_path: Path to the repository.

        Returns:
            List of modified file paths.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip().split("\n") if result.stdout.strip() else []
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting modified files: {e.stderr}")
            return []

    @staticmethod
    def reset_repository(repo_path: str) -> bool:
        """
        Reset repository to HEAD.

        Args:
            repo_path: Path to the repository.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Hard reset to HEAD
            result = subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error resetting repository: {e.stderr}")
            return False





