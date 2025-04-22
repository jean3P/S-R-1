# src/utils/patch_formatting_system.py

import logging
import re
from typing import Dict, Any, List
from .enhanced_patch_formatter import EnhancedPatchFormatter
from .multi_file_patch_helper import MultiFilePatchHelper
from .patch_validator import PatchValidator

logger = logging.getLogger(__name__)


class PatchFormattingSystem:
    """
    Integrated system for patch formatting, validation, and correction.
    Combines multiple utilities to provide comprehensive patch handling.
    """

    def __init__(self, config):
        """Initialize the patch formatting system."""
        self.config = config

        # Initialize components
        self.formatter = EnhancedPatchFormatter(config)
        self.multi_file_helper = MultiFilePatchHelper(config)
        self.validator = PatchValidator(config)

        # Debugging flag
        self.debug = config.get("patch_debug", False)

    def format_and_validate(self, patch: str, repo_name: str, issue_id: str) -> Dict[str, Any]:
        """
        Format and validate a patch in one operation.

        Args:
            patch: Original patch string
            repo_name: Repository name
            issue_id: Issue ID for validation

        Returns:
            Dictionary with results
        """
        if self.debug:
            logger.debug(f"ORIGINAL PATCH:\n{patch}\n{'=' * 40}")

        # Step 1: Basic formatting
        formatted_patch = self.formatter.format_patch(patch, repo_name)

        if self.debug:
            logger.debug(f"AFTER BASIC FORMATTING:\n{formatted_patch}\n{'=' * 40}")

        # Step 2: Handle multi-file patches
        is_multi_file = "diff --git" in formatted_patch and formatted_patch.count("diff --git") > 1

        if is_multi_file:
            is_valid_multi, fixed_multi_patch, validation_results = self.multi_file_helper.validate_multi_file_patch(
                formatted_patch, repo_name
            )

            if self.debug:
                logger.debug(f"AFTER MULTI-FILE HANDLING:\n{fixed_multi_patch}\n{'=' * 40}")

            formatted_patch = fixed_multi_patch

        # Step 3: Validate the patch
        validation_result = self.validator.validate_patch(formatted_patch, issue_id)

        # Step 4: Prepare the result
        result = {
            "original_patch": patch,
            "formatted_patch": formatted_patch,
            "is_multi_file": is_multi_file,
            "validation": validation_result,
            "success": validation_result.get("success", False)
        }

        if is_multi_file:
            result["multi_file_validation"] = validation_results

        return result

    def find_relevant_files(self, repo_name: str, search_text: str) -> List[str]:
        """
        Find files that might be relevant to a fix based on text search.

        Args:
            repo_name: Repository name
            search_text: Text to search for

        Returns:
            List of potentially relevant file paths
        """
        return self.multi_file_helper.find_files_in_repo(repo_name, search_text)

    def extract_patch_details(self, patch: str) -> Dict[str, Any]:
        """
        Extract key details from a patch for analysis.

        Args:
            patch: Patch string

        Returns:
            Dictionary with patch details
        """
        details = {
            "files": [],
            "hunks": [],
            "modifications": {
                "additions": 0,
                "deletions": 0
            }
        }

        # Extract files
        file_matches = re.findall(r'diff --git a/(.*?) b/(.*)', patch)
        for a_path, b_path in file_matches:
            file_info = {
                "a_path": a_path,
                "b_path": b_path,
                "is_new": "new file mode" in patch and b_path in patch,
                "is_deleted": "deleted file mode" in patch and a_path in patch
            }
            details["files"].append(file_info)

        # Extract hunks and count modifications
        hunk_matches = re.finditer(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*?)(?=@@ |\Z)',
                                   patch, re.DOTALL)

        for match in hunk_matches:
            hunk_text = match.group(0)
            old_start = int(match.group(1))
            old_count = int(match.group(2) or 1)
            new_start = int(match.group(3))
            new_count = int(match.group(4) or 1)

            # Count additions and deletions
            additions = hunk_text.count('\n+') - hunk_text.count('\n++')
            deletions = hunk_text.count('\n-') - hunk_text.count('\n--')

            details["modifications"]["additions"] += additions
            details["modifications"]["deletions"] += deletions

            hunk_info = {
                "old_start": old_start,
                "old_count": old_count,
                "new_start": new_start,
                "new_count": new_count,
                "additions": additions,
                "deletions": deletions
            }
            details["hunks"].append(hunk_info)

        return details

    def get_formatting_guide(self) -> str:
        """
        Get a detailed guide for proper patch formatting.

        Returns:
            String with formatting instructions
        """
        return """
        # Patch Formatting Guide

        A properly formatted Git patch must follow this structure:

        ```
        diff --git a/path/to/file.py b/path/to/file.py
        --- a/path/to/file.py
        +++ b/path/to/file.py
        @@ -start_line,number_of_lines +start_line,number_of_lines @@ optional section info
         context line (starts with a SPACE)
         context line
        -removed line (starts with a MINUS)
        +added line (starts with a PLUS)
         context line
        ```

        ## Key Requirements

        1. The first line must start with `diff --git a/` followed by the file path
        2. The second line must start with `--- a/` followed by the file path
        3. The third line must start with `+++ b/` followed by the file path
        4. Each hunk header must use the format `@@ -X,Y +X,Z @@` where:
           - X is the starting line number
           - Y is the number of lines in the original file
           - Z is the number of lines in the new file
        5. Context lines (unchanged) must start with a SPACE character
        6. Removed lines must start with a MINUS character
        7. Added lines must start with a PLUS character

        ## Common Mistakes to Avoid

        1. Missing diff header
        2. Inconsistent file paths between the different header lines
        3. Incorrect line numbers in hunk headers
        4. Missing or incorrect prefix characters for context/added/removed lines
        5. Not including enough context lines around the changes

        ## Tips for Successful Patches

        1. Generate patches directly from git using `git diff`
        2. Verify file paths and line numbers match the target repository
        3. Include at least 3 lines of context before and after changes
        4. Make minimal, focused changes to fix the specific issue
        5. Use tools to validate your patch before submission
        """

    def summarize_patch(self, patch: str) -> str:
        """
        Create a human-readable summary of a patch.

        Args:
            patch: Patch string

        Returns:
            Summary string
        """
        if not patch or len(patch.strip()) < 10:
            return "Empty or invalid patch"

        details = self.extract_patch_details(patch)

        summary = []
        summary.append(f"# Patch Summary")

        # Files summary
        file_count = len(details["files"])
        summary.append(f"\n## Files Modified: {file_count}")

        for file_info in details["files"]:
            if file_info["is_new"]:
                summary.append(f"- Created: {file_info['b_path']}")
            elif file_info["is_deleted"]:
                summary.append(f"- Deleted: {file_info['a_path']}")
            else:
                summary.append(f"- Modified: {file_info['a_path']}")

        # Modifications summary
        additions = details["modifications"]["additions"]
        deletions = details["modifications"]["deletions"]
        summary.append(f"\n## Changes: +{additions} -{deletions}")

        # Hunks summary
        hunk_count = len(details["hunks"])
        summary.append(f"\n## Hunks: {hunk_count}")

        # Get a sample of the changes
        changes_sample = self._extract_change_sample(patch)
        if changes_sample:
            summary.append(f"\n## Sample Changes:")
            summary.append(changes_sample)

        return "\n".join(summary)

    def _extract_change_sample(self, patch: str, max_lines: int = 10) -> str:
        """Extract a representative sample of changes from the patch."""
        # Look for the first hunk with actual changes
        hunk_pattern = r'@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*?\n((?:[ \-+].*\n)+)'
        hunk_matches = re.finditer(hunk_pattern, patch)

        for match in hunk_matches:
            hunk_content = match.group(1)

            # Only interested in hunks with actual changes
            if '+' in hunk_content or '-' in hunk_content:
                # Limit to max_lines
                lines = hunk_content.split('\n')[:max_lines]
                if len(lines) > max_lines:
                    lines.append("... (additional lines not shown)")

                return "\n".join(lines)

        return ""
