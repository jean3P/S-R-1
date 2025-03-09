# src/utils/metrics.py
import re
import ast
from typing import Dict, Any


def calculate_code_metrics(code: str) -> Dict[str, Any]:
    """
    Calculate metrics for the code.

    Args:
        code: Code to analyze

    Returns:
        Metrics dictionary
    """
    metrics = {
        "line_count": len(code.splitlines()),
        "char_count": len(code),
        "complexity": _estimate_complexity(code)
    }

    # Add line type counts
    line_counts = _count_line_types(code)
    metrics.update(line_counts)

    # Add imports count
    try:
        tree = ast.parse(code)
        metrics["imports_count"] = _count_imports(tree)

        # Add function count
        metrics["function_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))

        # Add class count
        metrics["class_count"] = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

        # Calculate docstring coverage
        metrics["docstring_coverage"] = _calculate_docstring_coverage(tree)

        # Calculate complexity per function
        metrics["function_complexity"] = _get_function_complexity(tree)
    except SyntaxError:
        # If code has syntax errors, we can't parse it with ast
        metrics["imports_count"] = len(re.findall(r"^\s*(?:import|from)\s+", code, re.MULTILINE))
        metrics["function_count"] = len(re.findall(r"^\s*def\s+", code, re.MULTILINE))
        metrics["class_count"] = len(re.findall(r"^\s*class\s+", code, re.MULTILINE))
        metrics["has_syntax_errors"] = True

    return metrics


def compare_solutions(solution1: str, solution2: str) -> Dict[str, Any]:
    """
    Compare two solutions.

    Args:
        solution1: First solution
        solution2: Second solution

    Returns:
        Comparison metrics
    """
    metrics1 = calculate_code_metrics(solution1)
    metrics2 = calculate_code_metrics(solution2)

    comparison = {
        "line_count_diff": metrics2["line_count"] - metrics1["line_count"],
        "char_count_diff": metrics2["char_count"] - metrics1["char_count"],
        "complexity_diff": metrics2["complexity"] - metrics1["complexity"]
    }

    # Calculate code similarity
    similarity = _calculate_similarity(solution1, solution2)
    comparison["similarity"] = similarity

    # Check for improvements in key metrics
    improvements = []

    # Check if complexity decreased
    if metrics2["complexity"] < metrics1["complexity"]:
        improvements.append("reduced_complexity")

    # Check if docstring coverage improved
    if metrics2.get("docstring_coverage", 0) > metrics1.get("docstring_coverage", 0):
        improvements.append("improved_documentation")

    # Check if code is more concise
    if metrics2["code_lines"] < metrics1["code_lines"] and metrics2.get("function_count", 0) >= metrics1.get(
            "function_count", 0):
        improvements.append("more_concise")

    # Check if comment ratio improved
    comment_ratio1 = metrics1.get("comment_lines", 0) / max(1, metrics1.get("code_lines", 1))
    comment_ratio2 = metrics2.get("comment_lines", 0) / max(1, metrics2.get("code_lines", 1))

    if comment_ratio2 > comment_ratio1:
        improvements.append("better_commented")

    comparison["improvements"] = improvements

    return comparison


def _estimate_complexity(code: str) -> float:
    """
    Estimate code complexity using a simplified metric.

    Args:
        code: Code to analyze

    Returns:
        Complexity score
    """
    try:
        tree = ast.parse(code)
        analyzer = ComplexityAnalyzer()
        analyzer.visit(tree)
        return analyzer.complexity
    except SyntaxError:
        # Fallback to simple heuristic if code has syntax errors
        # Count control structures as a rough estimate
        control_structures = [
            "if ", "else:", "elif ",
            "for ", "while ",
            "try:", "except:", "finally:"
        ]

        # Count occurrences of control structures
        score = 0
        for structure in control_structures:
            score += code.count(structure)

        return score


def _count_line_types(code: str) -> Dict[str, int]:
    """
    Count different types of lines in the code.

    Args:
        code: Code to analyze

    Returns:
        Dictionary with line type counts
    """
    lines = code.splitlines()
    blank_lines = 0
    comment_lines = 0
    code_lines = 0
    docstring_lines = 0

    # Detect if we're inside a docstring
    in_docstring = False
    docstring_delimiters = 0

    for line in lines:
        stripped = line.strip()

        # Count docstring delimiters (''' or """)
        if '"""' in stripped or "'''" in stripped:
            docstring_delimiters += stripped.count('"""') + stripped.count("'''")
            in_docstring = docstring_delimiters % 2 == 1  # Toggle docstring state
            docstring_lines += 1
            continue

        # If we're inside a docstring
        if in_docstring:
            docstring_lines += 1
        # If it's a blank line
        elif not stripped:
            blank_lines += 1
        # If it's a comment line
        elif stripped.startswith("#"):
            comment_lines += 1
        # Otherwise it's code
        else:
            code_lines += 1

    return {
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "docstring_lines": docstring_lines,
        "code_lines": code_lines
    }


def _count_imports(tree: ast.AST) -> int:
    """
    Count the number of imports in the AST.

    Args:
        tree: AST to analyze

    Returns:
        Number of imports
    """
    count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            count += len(node.names)
        elif isinstance(node, ast.ImportFrom):
            count += len(node.names)

    return count


def _calculate_docstring_coverage(tree: ast.AST) -> float:
    """
    Calculate docstring coverage for functions and classes.

    Args:
        tree: AST to analyze

    Returns:
        Docstring coverage ratio (0.0 to 1.0)
    """
    def_count = 0
    with_docstring = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            def_count += 1
            if ast.get_docstring(node):
                with_docstring += 1

    if def_count == 0:
        return 0.0

    return with_docstring / def_count


def _get_function_complexity(tree: ast.AST) -> Dict[str, int]:
    """
    Calculate complexity for each function.

    Args:
        tree: AST to analyze

    Returns:
        Dictionary mapping function names to complexity scores
    """
    complexities = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            analyzer = ComplexityAnalyzer()
            analyzer.visit(node)
            complexities[node.name] = analyzer.complexity

    return complexities


def _calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using Jaccard similarity on tokens.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0.0 to 1.0)
    """

    # Clean and tokenize the texts
    def tokenize(text):
        # Remove comments
        text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)

        # Remove docstrings
        text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", '', text, flags=re.DOTALL)

        # Split into tokens (words and symbols)
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return set(tokens)

    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    # Calculate Jaccard similarity
    if not tokens1 and not tokens2:
        return 1.0  # Both are empty, consider them identical

    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union


class ComplexityAnalyzer(ast.NodeVisitor):
    """AST visitor to calculate code complexity."""

    def __init__(self):
        self.complexity = 1  # Start with 1 for the base complexity

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # Add 1 for each boolean operator (and, or)
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_Try(self, node):
        # Add 1 for try and 1 for each except handler
        self.complexity += 1 + len(node.handlers)
        self.generic_visit(node)