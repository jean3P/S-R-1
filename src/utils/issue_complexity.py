# utils/issue_complexity.py

import re
import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class IssueComplexityAnalyzer:
    """
    Utility class for analyzing and scoring the complexity of GitHub issues.
    """

    def __init__(self):
        """Initialize the issue complexity analyzer."""
        pass

    def calculate_complexity(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate complexity metrics for a GitHub issue.

        Args:
            issue: Issue dictionary containing metadata and patch.

        Returns:
            Dictionary with complexity metrics.
        """
        patch_complexity = self._analyze_patch_complexity(issue.get("patch", ""))
        description_complexity = self._analyze_description_complexity(issue.get("issue_description", ""))
        files_complexity = self._analyze_files_complexity(issue)

        # Combine metrics into an overall complexity score
        # Weight factors can be adjusted based on importance
        total_complexity = (
                0.4 * patch_complexity["overall"] +
                0.2 * description_complexity["overall"] +
                0.4 * files_complexity["overall"]
        )

        # Normalize to 0-10 scale
        normalized_complexity = min(10, max(0, total_complexity))

        return {
            "overall": normalized_complexity,
            "patch_complexity": patch_complexity,
            "description_complexity": description_complexity,
            "files_complexity": files_complexity,
            "complexity_level": self._get_complexity_level(normalized_complexity)
        }

    def _analyze_patch_complexity(self, patch: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a patch.

        Args:
            patch: Git patch string.

        Returns:
            Dictionary with patch complexity metrics.
        """
        # Count the number of files modified
        files_modified = len(re.findall(r'diff --git', patch))

        # Count the number of chunks modified
        chunks_modified = len(re.findall(r'@@ -\d+,\d+ \+\d+,\d+ @@', patch))

        # Count the number of lines added/removed
        lines_added = len(re.findall(r'^\+[^+]', patch, re.MULTILINE))
        lines_removed = len(re.findall(r'^-[^-]', patch, re.MULTILINE))
        total_changes = lines_added + lines_removed

        # Calculate change density (changes per file)
        change_density = total_changes / files_modified if files_modified > 0 else 0

        # Calculate complexity based on a weighted formula
        # Weight factors can be adjusted based on importance
        complexity = (
                0.2 * min(10, files_modified) +
                0.3 * min(10, chunks_modified / max(1, files_modified)) +
                0.5 * min(10, change_density / 10)
        )

        return {
            "overall": complexity,
            "files_modified": files_modified,
            "chunks_modified": chunks_modified,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "total_changes": total_changes,
            "change_density": change_density
        }

    def _analyze_description_complexity(self, description: str) -> Dict[str, Any]:
        """
        Analyze the complexity of an issue description.

        Args:
            description: Issue description text.

        Returns:
            Dictionary with description complexity metrics.
        """
        # Count the words in the description
        word_count = len(description.split())

        # Count the number of code blocks
        code_blocks = len(re.findall(r'```.*?```', description, re.DOTALL))

        # Look for technical terms (simple heuristic)
        technical_terms = len(re.findall(
            r'(error|bug|exception|function|method|class|object|parameter|return|variable|attribute|property|interface|framework|library|module|dependency|implementation|architecture)',
            description.lower()))

        # Count the number of steps or requirements
        steps = len(re.findall(r'^\d+\.', description, re.MULTILINE))

        # Calculate complexity based on a weighted formula
        complexity = (
                0.3 * min(10, word_count / 100) +
                0.3 * min(10, code_blocks * 2) +
                0.2 * min(10, technical_terms / 5) +
                0.2 * min(10, steps)
        )

        return {
            "overall": complexity,
            "word_count": word_count,
            "code_blocks": code_blocks,
            "technical_terms": technical_terms,
            "steps": steps
        }

    def _analyze_files_complexity(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the complexity based on the files involved.

        Args:
            issue: Issue dictionary.

        Returns:
            Dictionary with file complexity metrics.
        """
        # Get the list of files modified
        modified_files = []
        if "files_modified" in issue:
            modified_files.extend(issue["files_modified"])
        if "files_created" in issue:
            modified_files.extend(issue["files_created"])
        if "files_deleted" in issue:
            modified_files.extend(issue["files_deleted"])

        # Count files by type
        file_types = {}
        for file_path in modified_files:
            ext = file_path.split('.')[-1] if '.' in file_path else 'unknown'
            if ext in file_types:
                file_types[ext] += 1
            else:
                file_types[ext] = 1

        # Calculate type diversity (number of different file types)
        type_diversity = len(file_types)

        # Assign weights to different file types (more complex languages get higher weights)
        type_weights = {
            'py': 1.0,  # Python
            'c': 1.2,  # C
            'cpp': 1.3,  # C++
            'h': 1.1,  # Header
            'java': 1.2,  # Java
            'js': 1.0,  # JavaScript
            'ts': 1.1,  # TypeScript
            'go': 1.1,  # Go
            'rs': 1.2,  # Rust
            'rb': 1.0,  # Ruby
            'php': 1.0,  # PHP
            'html': 0.7,  # HTML
            'css': 0.7,  # CSS
            'json': 0.5,  # JSON
            'md': 0.3,  # Markdown
            'txt': 0.2,  # Text
            'unknown': 1.0  # Unknown
        }

        # Calculate weighted file count
        weighted_file_count = sum(count * type_weights.get(ext, 1.0) for ext, count in file_types.items())

        # Analyze directory depth
        dir_depths = [len(file_path.split('/')) - 1 for file_path in modified_files]
        avg_dir_depth = np.mean(dir_depths) if dir_depths else 0
        max_dir_depth = max(dir_depths) if dir_depths else 0

        # Calculate complexity
        complexity = (
                0.4 * min(10, len(modified_files)) +
                0.2 * min(10, type_diversity * 2) +
                0.2 * min(10, weighted_file_count / len(modified_files) if modified_files else 0) +
                0.2 * min(10, avg_dir_depth)
        )

        return {
            "overall": complexity,
            "files_count": len(modified_files),
            "type_diversity": type_diversity,
            "file_types": file_types,
            "weighted_file_count": weighted_file_count,
            "avg_dir_depth": avg_dir_depth,
            "max_dir_depth": max_dir_depth
        }

    def _get_complexity_level(self, complexity_score: float) -> str:
        """
        Convert numeric complexity to a text label.

        Args:
            complexity_score: Numeric complexity score.

        Returns:
            Text label representing complexity level.
        """
        if complexity_score < 2:
            return "Very Simple"
        elif complexity_score < 4:
            return "Simple"
        elif complexity_score < 6:
            return "Moderate"
        elif complexity_score < 8:
            return "Complex"
        else:
            return "Very Complex"
