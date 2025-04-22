# src/utils/enhanced_patch_formatter.py

import re
import os
import logging
import difflib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class EnhancedPatchFormatter:
    """
    Enhanced utility for fixing formatting issues in model-generated patches.
    Provides comprehensive repairs for common LLM patch generation errors.
    """

    def __init__(self, config):
        """Initialize the enhanced patch formatter."""
        self.config = config
        self.repo_base_path = Path(config["data"]["repositories"])
        self.patch_debug = config.get("patch_debug", False)

    def format_patch(self, patch: str, repo_name: str) -> str:
        """
        Format a model-generated patch to ensure it has correct syntax.

        Args:
            patch: The raw patch string from the model
            repo_name: The repository name for file validation

        Returns:
            A correctly formatted Git patch
        """
        if not patch or len(patch.strip()) < 10:
            logger.warning("Patch is too short or empty, cannot format.")
            return patch

        # Remove code block markers if present
        patch = self._remove_code_block_markers(patch)

        # Extract the repo path
        repo_path = self.repo_base_path / repo_name

        # Debug log
        if self.patch_debug:
            logger.debug(f"ORIGINAL PATCH:\n{patch}\n{'=' * 40}")

        # Step 1: Split patch into individual file changes
        file_patches = self._split_into_file_patches(patch)

        if not file_patches:
            # If no valid file patches found, try to create one from the entire content
            file_patches = [self._create_synthetic_patch(patch, repo_path)]

        # Step 2: Fix each file patch
        fixed_patches = []
        for file_patch in file_patches:
            if file_patch:
                # Fix patch header
                fixed_patch = self._fix_patch_header(file_patch, repo_name)

                # Fix file paths
                fixed_patch = self._fix_file_paths(fixed_patch, repo_path)

                # Fix hunk headers
                fixed_patch = self._fix_hunk_headers(fixed_patch)

                # Fix context lines
                fixed_patch = self._ensure_correct_line_markers(fixed_patch)

                # Validate against actual file content if possible
                fixed_patch = self._validate_against_file(fixed_patch, repo_path)

                fixed_patches.append(fixed_patch)

        # Step 3: Combine fixed patches
        formatted_patch = "\n".join(fixed_patches)

        # Debug log
        if self.patch_debug:
            logger.debug(f"FORMATTED PATCH:\n{formatted_patch}\n{'=' * 40}")

        return formatted_patch

    def _remove_code_block_markers(self, patch: str) -> str:
        """Remove markdown code block markers if present."""
        # Remove ```diff, ```patch, ```git or just ``` markers
        patch = re.sub(r'^```(?:diff|patch|git)?\n', '', patch, flags=re.MULTILINE)
        patch = re.sub(r'\n```$', '', patch, flags=re.MULTILINE)
        return patch

    def _split_into_file_patches(self, patch: str) -> List[str]:
        """Split a multi-file patch into individual file patches."""
        # Look for diff --git headers to split the patch
        diff_headers = re.finditer(r'^diff --git ', patch, re.MULTILINE)

        # Get indices of all diff headers
        indices = [match.start() for match in diff_headers]

        if not indices:
            # If no diff headers found, return the entire patch as a single entry
            return [patch]

        # Add the end of string as the final index
        indices.append(len(patch))

        # Split the patch using the indices
        file_patches = []
        for i in range(len(indices) - 1):
            file_patch = patch[indices[i]:indices[i + 1]].strip()
            # Only add non-empty patches
            if file_patch:
                file_patches.append(file_patch)

        return file_patches

    def _fix_patch_header(self, patch: str, repo_name: str) -> str:
        """Ensure the patch has correct diff, ---, and +++ headers."""
        lines = patch.split('\n')
        new_lines = []

        # Try to extract file path
        file_path = None

        # First check if it's a valid diff --git line
        if lines and lines[0].startswith('diff --git'):
            match = re.match(r'diff --git a/(.*?)\s+b/(.*)', lines[0])
            if match:
                a_path, b_path = match.groups()
                # If paths don't match, use the first one
                file_path = a_path
                # Fix the diff --git line
                new_lines.append(f"diff --git a/{file_path} b/{file_path}")
            else:
                # Malformed diff header, try to extract file path
                file_path_match = re.search(r'diff --git (?:a/)?([^\s]+)', lines[0])
                if file_path_match:
                    file_path = file_path_match.group(1)
                    new_lines.append(f"diff --git a/{file_path} b/{file_path}")
                else:
                    # No path found, keep original
                    new_lines.append(lines[0])
        else:
            # No diff header, try to find file path in the patch
            file_path = self._extract_file_path_from_content(patch)
            if file_path:
                new_lines.append(f"diff --git a/{file_path} b/{file_path}")
            else:
                # No file path found, use placeholder
                new_lines.append("diff --git a/unknown.py b/unknown.py")
                file_path = "unknown.py"

        # Process remaining lines and ensure --- and +++ headers
        i = 1 if lines and lines[0].startswith('diff --git') else 0
        minus_found = plus_found = False

        while i < len(lines):
            line = lines[i]

            if line.startswith('--- '):
                minus_found = True
                new_lines.append(f"--- a/{file_path}")
            elif line.startswith('+++ '):
                plus_found = True
                new_lines.append(f"+++ b/{file_path}")
            elif line.startswith('index '):
                # Skip index lines, they're optional
                pass
            else:
                # Add any other line
                new_lines.append(line)

            i += 1

        # If --- and +++ headers weren't found, add them after diff line
        if not (minus_found and plus_found):
            # Insert after diff header
            idx = 1  # Default position after diff line
            new_lines.insert(idx, f"--- a/{file_path}")
            new_lines.insert(idx + 1, f"+++ b/{file_path}")

        return '\n'.join(new_lines)

    def _extract_file_path_from_content(self, patch: str) -> Optional[str]:
        """Try to extract a file path from patch content."""
        # Try to find file path in @@ lines
        hunk_match = re.search(r'@@ .* @@ (?:function |class )?([^\n(]+?\.(?:py|java|js|c|cpp|h|rb))', patch)
        if hunk_match:
            return hunk_match.group(1).strip()

        # Try to find Python file references
        py_file = re.search(r'(?:\/|^)([a-zA-Z0-9_\/\-\.]+\.py)', patch)
        if py_file:
            return py_file.group(1)

        # Look for any file extension
        any_file = re.search(r'(?:\/|^)([a-zA-Z0-9_\/\-\.]+\.[a-zA-Z0-9]+)', patch)
        if any_file:
            return any_file.group(1)

        return None

    def _fix_file_paths(self, patch: str, repo_path: Path) -> str:
        """Fix file paths to ensure they exist in the repository."""
        # Extract current file path from patch
        match = re.search(r'diff --git a/(.*?) b/', patch)
        if not match:
            return patch

        current_path = match.group(1)

        # Check if file exists
        file_exists = (repo_path / current_path).exists()
        if file_exists:
            return patch

        # File doesn't exist, try to find a matching file
        # Start with filename only
        filename = os.path.basename(current_path)

        # Find all files with the same name
        matching_files = list(repo_path.glob(f"**/{filename}"))

        if matching_files:
            # Use the first match
            new_path = str(matching_files[0].relative_to(repo_path))
            logger.info(f"Fixing file path: {current_path} -> {new_path}")

            # Replace all occurrences of the file path
            patch = patch.replace(f"a/{current_path}", f"a/{new_path}")
            patch = patch.replace(f"b/{current_path}", f"b/{new_path}")

            return patch

        return patch

    def _fix_hunk_headers(self, patch: str) -> str:
        """Fix hunk headers to ensure they have correct format."""
        lines = patch.split('\n')
        new_lines = []

        for line in lines:
            if line.startswith('@@ '):
                # Try to parse the hunk header
                match = re.match(r'@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@(.*)', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2) or 1)
                    new_start = int(match.group(3))
                    new_count = int(match.group(4) or 1)
                    context = match.group(5) or ''

                    # Ensure counts are at least 1
                    old_count = max(1, old_count)
                    new_count = max(1, new_count)

                    # Reconstruct the header
                    new_lines.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{context}")
                else:
                    # Try to create a valid header from invalid format
                    numbers = re.findall(r'-?(\d+)', line)
                    if len(numbers) >= 2:
                        old_start = int(numbers[0])
                        new_start = int(numbers[1])
                        context = re.search(r'@@\s*.*?@@\s*(.*)', line)
                        context_str = f" {context.group(1)}" if context else ""
                        new_lines.append(f"@@ -{old_start},1 +{new_start},1 @@{context_str}")
                    else:
                        # Can't fix, keep as is
                        new_lines.append(line)
            else:
                new_lines.append(line)

        return '\n'.join(new_lines)

    def _ensure_correct_line_markers(self, patch: str) -> str:
        """Ensure all lines after a hunk header have correct +, -, or space prefixes."""
        lines = patch.split('\n')
        new_lines = []
        in_hunk = False

        for line in lines:
            if line.startswith('@@ '):
                in_hunk = True
                new_lines.append(line)
            elif in_hunk:
                if not line:
                    # Empty line, skip
                    new_lines.append(line)
                elif line[0] not in ['+', '-', ' ']:
                    # Line without proper prefix, assume it's a context line
                    new_lines.append(' ' + line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        return '\n'.join(new_lines)

    def _validate_against_file(self, patch: str, repo_path: Path) -> str:
        """Validate patch against actual file content and fix if needed."""
        # Extract file path
        match = re.search(r'diff --git a/(.*?) b/', patch)
        if not match:
            return patch

        file_path = match.group(1)
        full_path = repo_path / file_path

        if not full_path.exists():
            return patch

        try:
            # Read original file
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()

            # Get hunks
            hunks = []
            hunk_pattern = r'(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*?\n)((?: .*\n|\+.*\n|-.*\n)*)'
            hunk_matches = re.finditer(hunk_pattern, patch, re.DOTALL)

            for hunk_match in hunk_matches:
                header = hunk_match.group(1)
                content = hunk_match.group(2)
                hunks.append((header, content))

            if not hunks:
                return patch

            # Validate and fix each hunk
            fixed_hunks = []
            original_lines = original_content.splitlines()

            for header, content in hunks:
                header_match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)', header)
                if not header_match:
                    fixed_hunks.append(header + content)
                    continue

                old_start = int(header_match.group(1))
                old_count = int(header_match.group(2) or 1)
                context = header_match.group(5) or ''

                # Extract lines to be removed
                removed_lines = []
                for line in content.splitlines():
                    if line.startswith('-'):
                        removed_lines.append(line[1:])

                # Check if removed lines match original content
                file_context = original_lines[max(0, old_start - 1):max(0, old_start - 1 + old_count)]

                if not self._lines_match(removed_lines, file_context) and len(removed_lines) > 0:
                    # Try to find the correct location in the file
                    new_location = self._find_lines_in_file(removed_lines, original_lines)
                    if new_location is not None:
                        new_start, matched_lines = new_location
                        # Create a new header with corrected line numbers
                        new_header = f"@@ -{new_start},{len(matched_lines)} +{new_start},{len(matched_lines)} @@{context}\n"

                        # Create new content with corrected context
                        new_content = ""
                        for line in matched_lines:
                            if line in removed_lines:
                                new_content += f"-{line}\n"
                            else:
                                new_content += f" {line}\n"

                        # Add addition lines from original hunk
                        for line in content.splitlines():
                            if line.startswith('+'):
                                new_content += f"{line}\n"

                        fixed_hunks.append(new_header + new_content)
                    else:
                        # Can't find matching lines, keep as is
                        fixed_hunks.append(header + content)
                else:
                    fixed_hunks.append(header + content)

            # Replace hunks in original patch
            fixed_patch = patch
            for i, (original_hunk, fixed_hunk) in enumerate(zip(hunks, fixed_hunks)):
                original_text = original_hunk[0] + original_hunk[1]
                fixed_patch = fixed_patch.replace(original_text, fixed_hunk)

            return fixed_patch

        except Exception as e:
            logger.warning(f"Error validating patch against file: {e}")
            return patch

    def _lines_match(self, removed_lines: List[str], file_context: List[str]) -> bool:
        """Check if removed lines match file context."""
        if len(removed_lines) == 0:
            return True

        # Create normalized versions (strip whitespace)
        normalized_removed = [line.strip() for line in removed_lines]
        normalized_context = [line.strip() for line in file_context]

        # Check direct match
        if normalized_removed == normalized_context:
            return True

        # Check if removed lines are a subset of context
        if all(line in normalized_context for line in normalized_removed):
            return True

        # Check similarity ratio
        similarity = difflib.SequenceMatcher(None,
                                             "\n".join(normalized_removed),
                                             "\n".join(normalized_context)).ratio()
        return similarity > 0.8

    def _find_lines_in_file(self, lines: List[str], file_lines: List[str]) -> Optional[Tuple[int, List[str]]]:
        """Find the location of lines in the file."""
        if not lines:
            return None

        # Try to find the first removed line in the file
        first_line = lines[0].strip()

        # Find all occurrences of the first line
        occurrences = []
        for i, line in enumerate(file_lines):
            if line.strip() == first_line:
                occurrences.append(i)

        if not occurrences:
            # Try partial match
            best_match = -1
            best_score = 0
            for i, line in enumerate(file_lines):
                score = difflib.SequenceMatcher(None, line.strip(), first_line).ratio()
                if score > 0.8 and score > best_score:
                    best_score = score
                    best_match = i

            if best_match >= 0:
                occurrences.append(best_match)

        if not occurrences:
            return None

        # For each occurrence, check if subsequent lines match
        best_match = None
        best_matched_lines = []

        for start in occurrences:
            context_size = min(len(file_lines) - start, max(10, len(lines) * 2))
            context = file_lines[start:start + context_size]

            # Check how many lines match
            matched_lines = []
            line_idx = 0

            while line_idx < len(context) and len(matched_lines) < len(lines):
                context_line = context[line_idx].strip()

                # Try direct match with any remaining removed line
                for j, removed_line in enumerate(lines):
                    if removed_line.strip() == context_line:
                        matched_lines.append(context[line_idx])
                        break

                line_idx += 1

            if len(matched_lines) > 0:
                if best_match is None or len(matched_lines) > len(best_matched_lines):
                    best_match = start + 1  # 1-based line numbers
                    best_matched_lines = context[:len(matched_lines)]

        if best_match is not None:
            return best_match, best_matched_lines

        return None

    def _create_synthetic_patch(self, content: str, repo_path: Path) -> str:
        """
        Create a synthetic patch when no valid patch format is detected.
        This handles cases where the LLM outputs code changes without proper diff format.
        """
        # Try to find any Python file references
        file_path = self._extract_file_path_from_content(content)

        if not file_path:
            # Look for Python-like code and try to infer what file it might be for
            python_code = re.search(r'(def\s+\w+|class\s+\w+|import\s+\w+)', content)
            if python_code:
                # Just use a placeholder name
                file_path = "unknown.py"
            else:
                # No clear file type, use generic name
                file_path = "unknown.txt"

        # Create a basic patch structure
        patch = f"diff --git a/{file_path} b/{file_path}\n"
        patch += f"--- a/{file_path}\n"
        patch += f"+++ b/{file_path}\n"

        # Try to find code additions/changes
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', content, re.DOTALL)
        if code_blocks:
            # Use the first code block
            code = code_blocks[0]
            lines = code.split('\n')

            # Create a hunk
            patch += "@@ -1,1 +1,{} @@\n".format(len(lines))
            patch += "- # Original code\n"
            for line in lines:
                patch += f"+ {line}\n"
        else:
            # No code blocks, try to use text content
            # Find lines that look like code (indentation, special characters)
            code_lines = []
            for line in content.split('\n'):
                # Skip empty lines or plain English sentences
                if not line.strip() or (
                        line.strip().endswith('.') and ' ' in line and not line.strip().startswith('#')):
                    continue
                # Check for code indicators: indentation, operators, brackets
                if (line.startswith('    ') or
                        any(x in line for x in ['(', ')', '{', '}', '[', ']', '=', '+=', '-=', '*=', '/=']) or
                        line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'import '))):
                    code_lines.append(line)

            if code_lines:
                # Create a hunk with the extracted code lines
                patch += "@@ -1,1 +1,{} @@\n".format(len(code_lines))
                patch += "- # Original code\n"
                for line in code_lines:
                    patch += f"+ {line}\n"
            else:
                # Last resort: just use some content
                patch += "@@ -1,1 +1,3 @@\n"
                patch += "- # Original code\n"
                patch += "+ # Modified code\n"
                patch += "+ # No clear code changes detected\n"
                patch += "+ # Please verify this patch manually\n"

        return patch

