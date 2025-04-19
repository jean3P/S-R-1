# src/evaluation/evaluator.py

import logging
import re
from typing import Dict, Any
import difflib

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for assessing the quality of solutions.
    """

    def __init__(self, config):
        """
        Initialize the evaluator.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.metrics = config["evaluation"]["metrics"]

    def evaluate_solution(self, issue: Dict[str, Any], solution_patch: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate a solution against ground truth.

        Args:
            issue: Issue dictionary.
            solution_patch: Generated solution patch.
            ground_truth: Ground truth patch.

        Returns:
            Dictionary containing evaluation metrics.
        """
        results = {}

        # Evaluate success rate
        if "success_rate" in self.metrics:
            results["success_rate"] = self._evaluate_success_rate(solution_patch, ground_truth)

        # Evaluate code quality
        if "code_quality" in self.metrics:
            results["code_quality"] = self._evaluate_code_quality(solution_patch)

        # Evaluate patch quality
        if "patch_quality" in self.metrics:
            results["patch_quality"] = self._evaluate_patch_quality(solution_patch, ground_truth)

        # Compute overall score
        results["overall_score"] = self._compute_overall_score(results)

        return results

    def _evaluate_success_rate(self, solution_patch: str, ground_truth: str) -> float:
        """Evaluate the success rate of the solution."""
        # Simple heuristic: compare number of changed files and lines
        solution_files = self._count_files_in_patch(solution_patch)
        ground_truth_files = self._count_files_in_patch(ground_truth)

        solution_lines = self._count_changed_lines(solution_patch)
        ground_truth_lines = self._count_changed_lines(ground_truth)

        # Calculate file similarity
        if ground_truth_files == 0:
            file_similarity = 1.0 if solution_files == 0 else 0.0
        else:
            file_similarity = max(0.0, 1.0 - abs(solution_files - ground_truth_files) / ground_truth_files)

        # Calculate line similarity
        if ground_truth_lines == 0:
            line_similarity = 1.0 if solution_lines == 0 else 0.0
        else:
            line_similarity = max(0.0, 1.0 - abs(solution_lines - ground_truth_lines) / ground_truth_lines)

        # Combine metrics
        return 0.5 * file_similarity + 0.5 * line_similarity

    def _evaluate_code_quality(self, solution_patch: str) -> float:
        """Evaluate the code quality of the solution."""
        # Simple heuristic based on patch characteristics
        score = 1.0

        # Check for potential issues

        # 1. Too many changes
        changed_lines = self._count_changed_lines(solution_patch)
        if changed_lines > 100:
            score -= 0.2

        # 2. Commented-out code
        if re.search(r'^\+\s*//.*\b(TODO|FIXME|XXX)\b', solution_patch, re.MULTILINE):
            score -= 0.1

        # 3. Print statements or debug code
        if re.search(r'^\+\s*(console\.log|print\(|System\.out|debugger|alert\()', solution_patch, re.MULTILINE):
            score -= 0.1

        # 4. Hardcoded values without explanation
        if re.search(r'^\+.*\b[0-9]{4,}\b', solution_patch, re.MULTILINE):
            score -= 0.05

        # Ensure score is within bounds
        return max(0.0, min(1.0, score))

    def _evaluate_patch_quality(self, solution_patch: str, ground_truth: str) -> float:
        """Evaluate the quality of the patch compared to ground truth."""
        # Simple heuristic based on patch similarity

        # Normalize patches
        solution_norm = self._normalize_patch(solution_patch)
        ground_truth_norm = self._normalize_patch(ground_truth)

        # Calculate similarity using difflib
        matcher = difflib.SequenceMatcher(None, solution_norm, ground_truth_norm)
        similarity = matcher.ratio()

        return similarity

    def _compute_overall_score(self, results: Dict[str, float]) -> float:
        """Compute an overall score from individual metrics."""
        if not results:
            return 0.0

        # Weights for each metric
        weights = {
            "success_rate": 0.5,
            "code_quality": 0.3,
            "patch_quality": 0.2
        }

        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in results:
                weighted_sum += results[metric] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _count_files_in_patch(self, patch: str) -> int:
        """Count the number of files in a patch."""
        return len(re.findall(r'diff --git', patch))

    def _count_changed_lines(self, patch: str) -> int:
        """Count the number of changed lines in a patch."""
        additions = len(re.findall(r'^\+[^+]', patch, re.MULTILINE))
        deletions = len(re.findall(r'^-[^-]', patch, re.MULTILINE))
        return additions + deletions

    def _normalize_patch(self, patch: str) -> str:
        """Normalize a patch for comparison."""
        # Remove file headers and hunks headers
        lines = patch.split('\n')
        normalized_lines = []

        for line in lines:
            # Skip diff headers, file mode lines, and hunk headers
            if (line.startswith('diff --git') or
                    line.startswith('index ') or
                    line.startswith('--- ') or
                    line.startswith('+++ ') or
                    line.startswith('@@ ')):
                continue

            # Keep only content lines
            if line.startswith('+') or line.startswith('-') or line.startswith(' '):
                normalized_lines.append(line)

        return '\n'.join(normalized_lines)

