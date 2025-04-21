# src/utils/patch_validator.py

import os
import re
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class PatchValidator:
    """
    Validator for patches to ensure they apply correctly to the codebase.
    """

    def __init__(self, config):
        """
        Initialize the patch validator.

        Args:
            config: Configuration object containing paths.
        """
        self.config = config

    def validate_patch(self, patch: str, issue_id: str) -> Dict[str, Any]:
        """
        Validate if a patch applies cleanly to the repository.

        Args:
            patch: The patch content to validate.
            issue_id: ID of the issue for context.

        Returns:
            Dictionary with validation results.
        """
        logger.info(f"Validating patch for issue {issue_id}")

        if not patch or patch.strip() == "":
            return {
                "success": False,
                "feedback": "Empty patch provided. Please provide a valid Git-formatted patch."
            }

        # Check if patch contains a Git diff header
        if "diff --git" not in patch:
            return {
                "success": False,
                "feedback": "Patch does not contain a Git diff header. Make sure it starts with 'diff --git a/file b/file'."
            }

        # Extract repository information from issue ID
        repo = self._extract_repo_from_issue_id(issue_id)
        if not repo:
            return {
                "success": False,
                "feedback": f"Could not determine repository from issue ID: {issue_id}"
            }

        # Get repository path
        repo_path = Path(self.config["data"]["repositories"]) / repo
        if not repo_path.exists():
            return {
                "success": False,
                "feedback": f"Repository not found at {repo_path}"
            }

        # Save patch to temporary file
        patch_file = self._save_patch_to_temp_file(patch)
        if not patch_file:
            return {
                "success": False,
                "feedback": "Failed to create temporary patch file."
            }

        try:
            # Try to apply the patch in check mode
            return self._check_patch_application(patch_file, repo_path)
        finally:
            # Clean up
            try:
                if patch_file:
                    os.unlink(patch_file)
            except Exception as e:
                logger.error(f"Error removing temporary patch file: {e}")

    def _extract_repo_from_issue_id(self, issue_id: str) -> Optional[str]:
        """Extract repository name from issue ID."""
        # Assuming issue_id format: repo__issue_number
        # e.g. "astropy__astropy-12907" -> "astropy"
        match = re.match(r'^([^_]+)__', issue_id)
        if match:
            return match.group(1)
        return None

    def _save_patch_to_temp_file(self, patch: str) -> Optional[str]:
        """Save patch content to a temporary file."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch)
                return f.name
        except Exception as e:
            logger.error(f"Error creating temporary patch file: {e}")
            return None

    def _check_patch_application(self, patch_file: str, repo_path: Path) -> Dict[str, Any]:
        """Check if a patch can be applied to the repository."""
        try:
            # Run git apply with --check to see if the patch would apply cleanly
            process = subprocess.run(
                ["git", "apply", "--check", "--verbose", patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True
            )

            # Process output
            if process.returncode == 0:
                # Patch applies cleanly
                return {
                    "success": True,
                    "feedback": "Patch applies cleanly to the repository."
                }
            else:
                # Patch failed to apply
                error_info = self._parse_patch_error(process.stderr, process.stdout)
                return {
                    "success": False,
                    "feedback": self._generate_feedback(error_info)
                }
        except Exception as e:
            logger.error(f"Error validating patch: {e}")
            return {
                "success": False,
                "feedback": f"Error validating patch: {str(e)}"
            }

    def _parse_patch_error(self, stderr: str, stdout: str) -> Dict[str, Any]:
        """Parse error information from patch application output."""
        error_info = {
            "files": [],
            "line_numbers": [],
            "error_messages": []
        }

        # Extract file names
        file_matches = re.findall(r'error: patch failed: ([^:]+)', stderr)
        if file_matches:
            error_info["files"] = file_matches

        # Extract line numbers
        line_matches = re.findall(r'at line (\d+)', stderr)
        if line_matches:
            error_info["line_numbers"] = [int(line) for line in line_matches]

        # Extract error messages
        for line in stderr.split("\n"):
            if "error:" in line or "failed:" in line:
                error_info["error_messages"].append(line.strip())

        # Extract hunk information
        hunk_matches = re.findall(r'Hunk #(\d+) FAILED', stderr)
        if hunk_matches:
            error_info["failed_hunks"] = [int(hunk) for hunk in hunk_matches]

        return error_info

    def _generate_feedback(self, error_info: Dict[str, Any]) -> str:
        """Generate human-readable feedback from error information."""
        feedback = "Patch validation failed. Here are the details:\n\n"

        if error_info["files"]:
            feedback += "Files with problems:\n"
            for file in error_info["files"]:
                feedback += f"- {file}\n"
            feedback += "\n"

        if error_info["error_messages"]:
            feedback += "Error messages:\n"
            for msg in error_info["error_messages"]:
                feedback += f"- {msg}\n"
            feedback += "\n"

        if "failed_hunks" in error_info and error_info["failed_hunks"]:
            feedback += "Failed hunks:\n"
            for hunk in error_info["failed_hunks"]:
                feedback += f"- Hunk #{hunk}\n"
            feedback += "\n"

        if error_info["line_numbers"]:
            feedback += "Problematic line numbers:\n"
            for line in error_info["line_numbers"]:
                feedback += f"- Line {line}\n"
            feedback += "\n"

        feedback += "Suggestions for fixing the patch:\n"
        feedback += "1. Ensure the patch targets the correct file paths\n"
        feedback += "2. Check that line numbers and context lines match the current files\n"
        feedback += "3. Make sure the patch is formatted as a proper Git diff\n"
        feedback += "4. Verify that the modifications are applied to the correct functions/classes\n"

        return feedback
