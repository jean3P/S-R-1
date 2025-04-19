# src/data/dataset_utils.py
import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class DatasetUtils:
    """
    Utility functions for working with the SWE-bench dataset.
    """

    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        """
        Extract code blocks from markdown text.

        Args:
            text: Markdown text with code blocks.

        Returns:
            List of code blocks.
        """
        # Match code blocks with language specification: ```python, ```javascript, etc.
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        return matches

    @staticmethod
    def filter_issues_by_repo(issues: List[Dict[str, Any]], repo_name: str) -> List[Dict[str, Any]]:
        """
        Filter issues by repository name.

        Args:
            issues: List of issue dictionaries.
            repo_name: Name of the repository to filter by.

        Returns:
            Filtered list of issues.
        """
        return [issue for issue in issues if issue.get("repo", "") == repo_name]

    @staticmethod
    def parse_diff(diff_text: str) -> Dict[str, Any]:
        """
        Parse a Git diff/patch text into structured data.

        Args:
            diff_text: Text of the Git diff/patch.

        Returns:
            Dictionary with parsed diff information.
        """
        result = {
            "files": []
        }

        current_file = None
        current_hunks = []

        lines = diff_text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for file header
            if line.startswith("diff --git"):
                # Save previous file if exists
                if current_file:
                    current_file["hunks"] = current_hunks
                    result["files"].append(current_file)

                # Extract file name
                file_pattern = r"diff --git a/(.*) b/(.*)"
                match = re.match(file_pattern, line)
                if match:
                    old_file, new_file = match.groups()
                    current_file = {
                        "old_file": old_file,
                        "new_file": new_file,
                        "is_new": False,
                        "is_deleted": False
                    }
                    current_hunks = []

            # Check for file mode changes
            elif line.startswith("new file mode"):
                if current_file:
                    current_file["is_new"] = True

            elif line.startswith("deleted file mode"):
                if current_file:
                    current_file["is_deleted"] = True

            # Check for hunk header
            elif line.startswith("@@"):
                hunk_pattern = r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)"
                match = re.match(hunk_pattern, line)
                if match:
                    old_start, old_count, new_start, new_count, desc = match.groups()
                    old_count = old_count or "1"
                    new_count = new_count or "1"

                    hunk = {
                        "old_start": int(old_start),
                        "old_count": int(old_count),
                        "new_start": int(new_start),
                        "new_count": int(new_count),
                        "description": desc.strip(),
                        "changes": []
                    }

                    # Process hunk changes
                    i += 1
                    while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("diff --git"):
                        change_line = lines[i]
                        if change_line:
                            if change_line[0] in [' ', '+', '-']:
                                change_type = {
                                    ' ': 'context',
                                    '+': 'addition',
                                    '-': 'deletion'
                                }[change_line[0]]

                                hunk["changes"].append({
                                    "type": change_type,
                                    "content": change_line[1:]
                                })
                        i += 1

                    current_hunks.append(hunk)
                    i -= 1  # Adjust index since we went one line too far

            i += 1

        # Add the last file if exists
        if current_file:
            current_file["hunks"] = current_hunks
            result["files"].append(current_file)

        return result

    @staticmethod
    def generate_patch_from_changes(changes: Dict[str, Any]) -> str:
        """
        Generate a Git patch from structured change data.

        Args:
            changes: Dictionary with changes information.

        Returns:
            String containing the Git patch.
        """
        patch = []

        for file_change in changes["files"]:
            old_file = file_change["old_file"]
            new_file = file_change["new_file"]

            # Add file header
            patch.append(f"diff --git a/{old_file} b/{new_file}")

            if file_change["is_new"]:
                patch.append("new file mode 100644")
            elif file_change["is_deleted"]:
                patch.append("deleted file mode 100644")

            patch.append(f"--- a/{old_file}")
            patch.append(f"+++ b/{new_file}")

            # Add hunks
            for hunk in file_change["hunks"]:
                old_start = hunk["old_start"]
                old_count = hunk["old_count"]
                new_start = hunk["new_start"]
                new_count = hunk["new_count"]
                desc = hunk["description"]

                patch.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@ {desc}")

                # Add changes
                for change in hunk["changes"]:
                    change_type = change["type"]
                    content = change["content"]

                    prefix = {
                        "context": " ",
                        "addition": "+",
                        "deletion": "-"
                    }[change_type]

                    patch.append(f"{prefix}{content}")

            patch.append("")  # Empty line between files

        return "\n".join(patch)
