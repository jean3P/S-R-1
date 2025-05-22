"""
Solution diversity analyzer for measuring functional and structural
similarity across multiple code solutions.
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple

logger = logging.getLogger(__name__)


class SolutionDiversityAnalyzer:
    """
    Analyzes a set of code solutions to measure their diversity
    using structural and functional similarity metrics.
    """

    def __init__(self):
        """Initialize the solution diversity analyzer."""
        pass

    def analyze_diversity(self, solutions: List[str], solution_hashes: List[str] = None) -> Dict[str, Any]:
        """
        Analyze the diversity of a set of solutions.

        Args:
            solutions: List of solution code strings
            solution_hashes: Optional list of precomputed solution hashes

        Returns:
            Dictionary with diversity metrics
        """
        if not solutions:
            return {
                "unique_solutions": 0,
                "similarity_score": 0.0,
                "solution_lengths": {"min": 0, "max": 0, "avg": 0.0},
                "algorithm_approaches": {},
                "feature_diversity": 0.0
            }

        # Calculate solution length statistics
        solution_lengths = [len(s) for s in solutions]
        min_length = min(solution_lengths) if solution_lengths else 0
        max_length = max(solution_lengths) if solution_lengths else 0
        avg_length = sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0

        # Count unique solutions if hashes provided
        unique_count = 0
        unique_ratio = 0.0
        if solution_hashes:
            unique_hashes = set(solution_hashes)
            unique_count = len(unique_hashes)
            unique_ratio = unique_count / len(solution_hashes) if solution_hashes else 0.0

        # Determine AST-like features for each solution to measure structural similarity
        solution_features = []
        for solution in solutions:
            # Extract key features from the code
            features = self._extract_code_features(solution)
            solution_features.append(features)

        # Calculate feature-based similarity
        similarity_score = self._calculate_overall_similarity(solution_features)

        # Calculate algorithmic approach diversity
        algo_approaches = self._identify_algorithm_approaches(solutions)

        # Calculate feature diversity ratio
        feature_set_tuples = [tuple(sorted(f.items())) for f in solution_features]
        unique_feature_sets = set(feature_set_tuples)
        feature_diversity = len(unique_feature_sets) / len(solution_features) if solution_features else 0.0

        result = {
            "unique_solutions": unique_count,
            "unique_ratio": unique_ratio,
            "similarity_score": similarity_score,
            "solution_lengths": {
                "min": min_length,
                "max": max_length,
                "avg": avg_length
            },
            "algorithm_approaches": algo_approaches,
            "feature_diversity": feature_diversity,
            # Additional metrics:
            "features_analysis": self._analyze_features(solution_features),
            "complexity_diversity": self._analyze_complexity(solutions)
        }

        return result

    def _normalize_code(self, code: str) -> str:
        """
        Normalize code for consistent feature extraction:
        - Remove comments
        - Standardize whitespace
        - Remove string literals

        Args:
            code: Original code string

        Returns:
            Normalized code string
        """
        import re

        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)

        # Replace string literals with empty strings
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)

        # Standardize whitespace
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        return ' '.join(lines)

    def _extract_code_features(self, code: str) -> Dict[str, Any]:
        """
        Extract key features from code to represent its structural characteristics.
        This is a simplified AST-like analysis without external dependencies.

        Args:
            code: Python code string

        Returns:
            Dictionary of code features
        """
        # Normalize code first
        normalized = self._normalize_code(code)

        features = {
            # Control flow structures
            "has_recursion": bool(re.search(r'def\s+\w+[^}]*?\1\s*\(', normalized)),
            "loop_count": len(re.findall(r'\bfor\b|\bwhile\b', normalized)),
            "if_count": len(re.findall(r'\bif\b', normalized)),
            "else_count": len(re.findall(r'\belse\b', normalized)),

            # Data structures used
            "uses_list": bool(re.search(r'\[\]|\blist\b', normalized)),
            "uses_dict": bool(re.search(r'\{\}|\bdict\b', normalized)),
            "uses_set": bool(re.search(r'set\(', normalized)),
            "uses_heap": bool(re.search(r'heapq\.', normalized) or re.search(r'heappush|heappop', normalized)),
            "uses_queue": bool(re.search(r'collections\.deque|Queue', normalized)),
            "uses_stack": bool(re.search(r'\.append\(.*\.pop\(', normalized)),

            # Algorithm indicators
            "uses_dp": bool(re.search(r'dp\s*=|memo\s*=', normalized)),
            "uses_bfs": bool(re.search(r'\bqueue\b|\bdeque\b.*popleft', normalized)),
            "uses_dfs": bool(re.search(r'dfs\(|def\s+dfs', normalized)),
            "uses_binary_search": bool(re.search(r'mid\s*=|binary_search|left\s*<=\s*right', normalized)),

            # Code complexity indicators
            "return_count": len(re.findall(r'\breturn\b', normalized)),
            "assignment_count": len(re.findall(r'=(?!=)', normalized)),
            "class_method_count": len(re.findall(r'def\s+\w+', normalized)),
            "nested_loops": bool(re.search(r'for.*for|while.*while|for.*while|while.*for', normalized)),

            # Other structural features
            "line_count": len(code.splitlines()),
            "char_count": len(code),
            "max_indent": max([len(line) - len(line.lstrip()) for line in code.splitlines() if line.strip()], default=0)
        }

        return features

    def _calculate_feature_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two feature sets.

        Args:
            features1: First feature dictionary
            features2: Second feature dictionary

        Returns:
            Similarity score between 0 and 1
        """
        # Get all boolean and numeric features
        bool_features = [k for k, v in features1.items() if isinstance(v, bool)]
        numeric_features = [k for k, v in features1.items() if isinstance(v, (int, float)) and k not in bool_features]

        # Calculate Jaccard similarity for boolean features
        bool_similarity = 0
        if bool_features:
            matches = sum(1 for f in bool_features if features1[f] == features2[f])
            bool_similarity = matches / len(bool_features)

        # Calculate normalized difference for numeric features
        numeric_similarity = 0
        if numeric_features:
            differences = []
            for feat in numeric_features:
                # Skip features with zero values in both to avoid division issues
                if features1[feat] == 0 and features2[feat] == 0:
                    differences.append(1.0)  # Perfect match
                elif features1[feat] == 0 or features2[feat] == 0:
                    differences.append(0.0)  # No match
                else:
                    # Calculate normalized difference
                    max_val = max(features1[feat], features2[feat])
                    min_val = min(features1[feat], features2[feat])
                    differences.append(min_val / max_val)

            numeric_similarity = sum(differences) / len(differences) if differences else 0

        # Combine similarities (weight can be adjusted)
        combined_similarity = (bool_similarity * 0.6) + (numeric_similarity * 0.4)
        return combined_similarity

    def _calculate_overall_similarity(self, feature_sets: List[Dict[str, Any]]) -> float:
        """
        Calculate average pairwise similarity across all feature sets.

        Args:
            feature_sets: List of feature dictionaries

        Returns:
            Average similarity score
        """
        if len(feature_sets) <= 1:
            return 0.0

        # Sample solutions if there are too many (to avoid O(n²) complexity)
        sample_features = feature_sets
        if len(feature_sets) > 20:
            import random
            sample_indices = random.sample(range(len(feature_sets)), 20)
            sample_features = [feature_sets[i] for i in sample_indices]

        # Calculate pairwise feature similarities
        total_similarity = 0.0
        comparison_count = 0

        for i in range(len(sample_features)):
            for j in range(i + 1, len(sample_features)):
                sim = self._calculate_feature_similarity(sample_features[i], sample_features[j])
                total_similarity += sim
                comparison_count += 1

        # Calculate average similarity
        similarity_score = total_similarity / comparison_count if comparison_count > 0 else 0.0
        return similarity_score

    def _identify_algorithm_approaches(self, solutions: List[str]) -> Dict[str, int]:
        """
        Identify and categorize algorithmic approaches used across solutions.

        Args:
            solutions: List of solution code strings

        Returns:
            Dictionary mapping approach names to occurrence counts
        """
        approach_counts = {
            "dynamic_programming": 0,
            "greedy": 0,
            "divide_and_conquer": 0,
            "breadth_first_search": 0,
            "depth_first_search": 0,
            "binary_search": 0,
            "two_pointers": 0,
            "sliding_window": 0,
            "backtracking": 0,
            "hash_table": 0,
            "math_based": 0,
            "simulation": 0,
            "other": 0
        }

        # Patterns to match various algorithmic approaches
        patterns = {
            "dynamic_programming": [r'dp\s*=', r'memo\s*=', r'cache\s*=', r'@lru_cache', r'tabulation', r'memoization'],
            "greedy": [r'greedy', r'maximal', r'optimal', r'sort.*then'],
            "divide_and_conquer": [r'merge\s*\(', r'partition', r'quick_sort', r'divide', r'conquer'],
            "breadth_first_search": [r'queue\s*=', r'deque\s*\(', r'\.popleft\(', r'bfs', r'level by level'],
            "depth_first_search": [r'dfs\s*\(', r'def\s+dfs', r'visit\s*\(', r'stack\s*='],
            "binary_search": [r'mid\s*=', r'binary_search', r'left\s*<=\s*right', r'bisect'],
            "two_pointers": [r'left\s*=.*right\s*=', r'start\s*=.*end\s*=', r'two\s*pointers', r'pointer'],
            "sliding_window": [r'window', r'sliding', r'substr', r'subarray.*current'],
            "backtracking": [r'backtrack', r'def\s+bt', r'combination', r'permutation'],
            "hash_table": [r'dict\s*\(', r'map\s*=', r'counter\s*=', r'set\s*\(', r'defaultdict'],
            "math_based": [r'math\.', r'factorial', r'combinations', r'gcd', r'lcm', r'prime'],
            "simulation": [r'simulate', r'simulation', r'process', r'step by step']
        }

        for solution in solutions:
            # Classify each solution based on patterns
            solution_lower = solution.lower()
            found_approach = False

            for approach, approach_patterns in patterns.items():
                if any(re.search(pattern, solution_lower) for pattern in approach_patterns):
                    approach_counts[approach] += 1
                    found_approach = True
                    break

            if not found_approach:
                approach_counts["other"] += 1

        # Filter out approaches with zero occurrences
        return {k: v for k, v in approach_counts.items() if v > 0}

    def _analyze_features(self, feature_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze feature distributions across solutions.

        Args:
            feature_sets: List of feature dictionaries

        Returns:
            Dictionary with feature analysis metrics
        """
        if not feature_sets:
            return {}

        # Analyze boolean features
        bool_features = {k: v for k, v in feature_sets[0].items() if isinstance(v, bool)}
        bool_feature_stats = {}

        for feature in bool_features:
            true_count = sum(1 for fs in feature_sets if fs.get(feature, False))
            bool_feature_stats[feature] = {
                "true_ratio": true_count / len(feature_sets),
                "count": true_count
            }

        # Analyze numeric features
        numeric_features = {k: v for k, v in feature_sets[0].items()
                            if isinstance(v, (int, float)) and k not in bool_features}
        numeric_feature_stats = {}

        for feature in numeric_features:
            values = [fs.get(feature, 0) for fs in feature_sets]
            numeric_feature_stats[feature] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "variance": sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)
            }

        return {
            "boolean_features": bool_feature_stats,
            "numeric_features": numeric_feature_stats
        }

    def _analyze_complexity(self, solutions: List[str]) -> Dict[str, Any]:
        """
        Analyze code complexity metrics across solutions.

        Args:
            solutions: List of solution code strings

        Returns:
            Dictionary with complexity analysis metrics
        """
        # Extract complexity indicators
        complexities = []
        for solution in solutions:
            normalized = self._normalize_code(solution)
            loop_nesting = max([len(re.findall(r'\bfor\b|\bwhile\b', line)) for line in solution.splitlines()],
                               default=0)

            complexity = {
                "line_count": len(solution.splitlines()),
                "loop_count": len(re.findall(r'\bfor\b|\bwhile\b', normalized)),
                "condition_count": len(re.findall(r'\bif\b|\belif\b|\belse\b', normalized)),
                "method_count": len(re.findall(r'def\s+\w+', normalized)),
                "assignment_count": len(re.findall(r'=(?!=)', normalized)),
                "max_loop_nesting": loop_nesting,
                # Estimate time complexity category
                "complexity_category": self._estimate_complexity_category(solution, loop_nesting)
            }
            complexities.append(complexity)

        # Count solutions in each complexity category
        complexity_distribution = {}
        for c in complexities:
            category = c["complexity_category"]
            complexity_distribution[category] = complexity_distribution.get(category, 0) + 1

        # Calculate averages
        avg_stats = {}
        if complexities:
            for key in complexities[0]:
                if key != "complexity_category":
                    avg_stats[f"avg_{key}"] = sum(c[key] for c in complexities) / len(complexities)

        return {
            "complexity_distribution": complexity_distribution,
            "avg_complexity_metrics": avg_stats,
            "complexity_diversity_ratio": len(complexity_distribution) / min(len(solutions), 6) if solutions else 0
        }

    def _estimate_complexity_category(self, code: str, loop_nesting: int) -> str:
        """
        Estimate the time complexity category of the solution.

        Args:
            code: Solution code string
            loop_nesting: Maximum nesting level of loops

        Returns:
            Estimated complexity category as a string
        """
        normalized = self._normalize_code(code)

        # Check for recursion or deep nested loops (exponential patterns)
        has_recursion = False
        # Look for function definitions
        function_defs = re.findall(r'def\s+(\w+)', normalized)
        for func_name in function_defs:
            # Check if the function calls itself
            if re.search(r'\b' + re.escape(func_name) + r'\s*\(', normalized):
                has_recursion = True
                break

        if (has_recursion and loop_nesting >= 1) or loop_nesting >= 3:
            return "exponential"

        # Check for polynomial patterns
        elif loop_nesting == 2:
            return "quadratic"

        # Check for linearithmic patterns
        elif (re.search(r'sort\(', normalized) or
              re.search(r'merge', normalized) or
              re.search(r'partition', normalized)):
            return "linearithmic"

        # Check for linear patterns
        elif loop_nesting == 1:
            return "linear"

        # Check for logarithmic patterns
        elif re.search(r'mid\s*=|binary_search|left\s*<=\s*right', normalized):
            return "logarithmic"

        # Default to constant if no loops or recursion detected
        else:
            return "constant"
