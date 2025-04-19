# src/solution/patch_creator.py

from typing import Dict, Any
import logging
from pathlib import Path
import difflib
import re

logger = logging.getLogger(__name__)


class PatchCreator:
    """
    Create patches from generated code.
    """

    def __init__(self, config):
        """
        Initialize the patch creator.

        Args:
            config: Configuration object.
        """
        self.config = config
        logger.info("PatchCreator initialized")

    def create_patch(self, file_code_map: Dict[str, str], issue: Dict[str, Any]) -> str:
        """
        Create a patch from generated code.

        Args:
            file_code_map: Dictionary mapping file paths to code.
            issue: Issue dictionary.

        Returns:
            String containing the Git-formatted patch.
        """
        logger.info("Creating patch from generated code")

        if not file_code_map:
            logger.warning("Empty file code map provided")
            return "# No code changes to create patch from"

        # Get repo path
        repo = issue.get("repo", "")
        repo_path = Path(self.config["data"]["swe_bench_path"]) / "repos" / repo

        # Generate diff for each file
        patch_lines = []

        for file_path, new_code in file_code_map.items():
            # Get the original file content
            original_file_path = repo_path / file_path
            is_new_file = False

            if original_file_path.exists():
                with open(original_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    original_code = f.read()
            else:
                # New file
                original_code = ""
                is_new_file = True

            # Generate diff
            file_diff = self._create_file_diff(
                file_path=file_path,
                original=original_code,
                modified=new_code,
                is_new_file=is_new_file
            )

            if file_diff:
                patch_lines.append(file_diff)
                logger.debug(f"Created diff for file: {file_path}")
            else:
                logger.warning(f"No differences found for file: {file_path}")

        # Create an empty patch with a placeholder if no diffs were generated
        if not patch_lines:
            logger.warning("No diffs created, returning a placeholder patch")
            # Try to find relevant filenames from the issue
            if "files_modified" in issue and issue["files_modified"]:
                example_file = issue["files_modified"][0]
                patch_lines.append(f"# No changes detected in {example_file}")
            else:
                patch_lines.append("# No changes detected in any files")

        return "\n".join(patch_lines)

    def _create_file_diff(self, file_path: str, original: str, modified: str, is_new_file: bool = False) -> str:
        """Create a Git-formatted diff for a single file."""
        # Skip if there are no changes
        if original == modified and not is_new_file:
            return ""

        # Split into lines
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()

        # Generate unified diff
        diff_lines = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=""
        )

        diff_text = "\n".join(list(diff_lines))

        # If it's a new file, add the appropriate header
        if is_new_file and diff_text:
            diff_text = f"diff --git a/{file_path} b/{file_path}\nnew file mode 100644\n{diff_text}"
        elif diff_text and not diff_text.startswith("diff --git"):
            diff_text = f"diff --git a/{file_path} b/{file_path}\n{diff_text}"

        return diff_text

    def _cleanup_for_patch(self, code: str) -> str:
        """Clean up code for inclusion in a patch."""
        # Remove any markdown code block syntax
        code = re.sub(r'^```\w*\n', '', code)
        code = re.sub(r'\n```$', '', code)

        # Ensure there's a newline at the end
        if not code.endswith('\n'):
            code += '\n'

        return code
