# src/utils/patch_validator.py

import os
import re
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from ..data.data_loader import SWEBenchDataLoader
from ..utils.enhanced_patch_formatter import EnhancedPatchFormatter

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
        self.patch_formatter = EnhancedPatchFormatter(config)  # Initialize the formatter

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
        data_loader = SWEBenchDataLoader(self.config)
        issue = data_loader.load_issue(issue_id)
        logger.info(f"Issue: {issue}")

        if not issue:
            return {
                "success": False,
                "feedback": f"Could not load issue data for {issue_id}"
            }

        if not data_loader.prepare_repository_for_testing(issue):
            return {
                "success": False,
                "feedback": "Failed to prepare repository environment for testing"
            }

        if not patch or patch.strip() == "":
            return {
                "success": False,
                "feedback": "Empty patch provided. Please provide a valid Git-formatted patch."
            }

        # Extract repository information from issue ID
        repo = self._extract_repo_from_issue_id(issue_id)
        if not repo:
            return {
                "success": False,
                "feedback": f"Could not determine repository from issue ID: {issue_id}"
            }

        # Format the patch to fix common issues before validation
        formatted_patch = self.patch_formatter.format_patch(patch, repo)
        if formatted_patch != patch:
            logger.info("Patch was reformatted to fix common issues")

        # Get repository path
        repo_path = Path(self.config["data"]["repositories"]) / repo
        if not repo_path.exists():
            return {
                "success": False,
                "feedback": f"Repository not found at {repo_path}"
            }

        # Check if patch contains a Git diff header
        if "diff --git" not in formatted_patch:
            return {
                "success": False,
                "feedback": "Patch does not contain a Git diff header. Make sure it starts with 'diff --git a/file b/file'."
            }

        # Save patch to temporary file
        patch_file = self._save_patch_to_temp_file(formatted_patch)
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

        # For different formats, try to extract any repo-like part
        parts = issue_id.split('_')
        if parts and '/' in parts[0]:
            # Could be a GitHub-style identifier like "owner/repo"
            return parts[0]

        # For simple issue IDs without separation
        if '/' in issue_id:
            # Might be owner/repo format
            return issue_id.split('/')[0]

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

                # Try with more relaxed options if strict check fails
                relaxed_result = self._try_relaxed_application(patch_file, repo_path)
                if relaxed_result["success"]:
                    return relaxed_result

                return {
                    "success": False,
                    "feedback": self._generate_feedback(error_info),
                    "error_details": error_info
                }
        except Exception as e:
            logger.error(f"Error validating patch: {e}")
            return {
                "success": False,
                "feedback": f"Error validating patch: {str(e)}"
            }

    def _try_relaxed_application(self, patch_file: str, repo_path: Path) -> Dict[str, Any]:
        """Try applying the patch with more relaxed options."""
        try:
            # Run git apply with --ignore-whitespace and --reject options
            process = subprocess.run(
                ["git", "apply", "--check", "--ignore-whitespace", "--ignore-space-change", patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True
            )

            if process.returncode == 0:
                return {
                    "success": True,
                    "feedback": "Patch applies with relaxed whitespace options.",
                    "note": "Note: Used relaxed whitespace handling for validation."
                }

            # If normal application fails, try with fuzzy matching
            # This is more dangerous so we don't actually validate with this
            # But we can report if it might work with fuzzy
            fuzzy_process = subprocess.run(
                ["git", "apply", "--check", "--reject", "--ignore-whitespace", "--fuzz=3", patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True
            )

            if fuzzy_process.returncode == 0:
                return {
                    "success": False,  # Still report as false since it's not clean
                    "feedback": "Patch would apply with fuzzy matching, which may indicate line number issues. Please verify line numbers carefully.",
                    "might_apply_with_fuzzy": True
                }

            return {
                "success": False,
                "feedback": "Patch does not apply even with relaxed options."
            }
        except Exception as e:
            logger.error(f"Error checking relaxed patch application: {e}")
            return {
                "success": False,
                "feedback": f"Error checking relaxed application: {str(e)}"
            }

    def _parse_patch_error(self, stderr: str, stdout: str) -> Dict[str, Any]:
        """Parse error information from patch application output."""
        error_info = {
            "files": [],
            "line_numbers": [],
            "error_messages": [],
            "suggested_fixes": []
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

        # Attempt to suggest fixes based on common patterns
        if "context does not match" in stderr:
            error_info["suggested_fixes"].append(
                "The context lines don't match the file. Verify the file content and line numbers.")

        if "corrupt patch" in stderr:
            error_info["suggested_fixes"].append(
                "The patch format appears to be invalid. Check for missing newlines or improper formatting.")

        if "already exists in working directory" in stderr:
            error_info["suggested_fixes"].append(
                "The patch tries to create a file that already exists. Use modification instead of creation.")

        # Detect line ending issues
        if "trailing whitespace" in stderr or "space/tab" in stderr:
            error_info["suggested_fixes"].append(
                "There may be line ending or whitespace issues. Try applying with --ignore-whitespace.")

        # Detect offset issues
        offset_matches = re.findall(r'offset (\d+) lines', stderr)
        if offset_matches:
            offsets = [int(off) for off in offset_matches]
            error_info["offsets"] = offsets
            error_info["suggested_fixes"].append(
                f"Line numbers appear to be off by approximately {offsets[0]} lines. Adjust line numbers in hunk headers.")

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

        if "offsets" in error_info and error_info["offsets"]:
            feedback += f"Line number offset detected: approximately {error_info['offsets'][0]} lines\n\n"

        if "suggested_fixes" in error_info and error_info["suggested_fixes"]:
            feedback += "Specific suggestions:\n"
            for fix in error_info["suggested_fixes"]:
                feedback += f"- {fix}\n"
            feedback += "\n"

        feedback += "Suggestions for fixing the patch:\n"
        feedback += "1. Ensure the patch targets the correct file paths\n"
        feedback += "2. Check that line numbers and context lines match the current files\n"
        feedback += "3. Make sure the patch is formatted as a proper Git diff\n"
        feedback += "4. Verify that the modifications are applied to the correct functions/classes\n"
        feedback += "5. For context line mismatches, try to re-create the patch from the current file content\n"

        return feedback
