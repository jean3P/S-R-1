# src/utils/parsing.py

import re
import ast
from typing import Dict, Any, Optional, List


def extract_code_blocks(text: str) -> List[str]:
    """
    Extract code blocks from text.

    Args:
        text: Text that may contain code blocks

    Returns:
        List of code blocks
    """
    # Match code blocks surrounded by triple backticks
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)

    # If no code blocks with backticks found, try to extract indented blocks
    if not code_blocks:
        # Look for indented blocks of code (4 spaces or 1 tab)
        indented_blocks = re.findall(r"(?:^(?:    |\t).*?$)+", text, re.MULTILINE)
        if indented_blocks:
            # Remove the indentation for each line
            code_blocks = [re.sub(r"^(?:    |\t)", "", block, flags=re.MULTILINE) for block in indented_blocks]

    # If still no code blocks found, try to find Python-like code without markers
    if not code_blocks:
        # Look for potential function definitions
        function_blocks = re.findall(r"(def\s+\w+\s*\(.*?\).*?(?:return|pass).*?)(?:\n\n|$)", text, re.DOTALL)
        if function_blocks:
            code_blocks = function_blocks

    return code_blocks


def extract_python_function(text: str, function_name: Optional[str] = None) -> Optional[str]:
    """
    Extract a Python function from text.

    Args:
        text: Text that may contain a Python function
        function_name: Name of the function to extract (if specified)

    Returns:
        Extracted function or None if not found
    """
    # If function name is specified, look for that specific function
    if function_name:
        pattern = rf"def\s+{function_name}\s*\(.*?\).*?(?:def|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # Clean up the function (remove trailing def if captured)
            func = match.group(0)
            if func.endswith("def"):
                func = func[:-3].strip()
            return func

    # Otherwise, extract the first function definition
    pattern = r"def\s+\w+\s*\(.*?\).*?(?:def|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Clean up the function (remove trailing def if captured)
        func = match.group(0)
        if func.endswith("def"):
            func = func[:-3].strip()
        return func

    return None


def parse_execution_result(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Parse execution results.

    Args:
        stdout: Standard output
        stderr: Standard error

    Returns:
        Parsed results
    """
    result = {
        "success": len(stderr.strip()) == 0,
        "stdout": stdout,
        "stderr": stderr,
        "has_output": len(stdout.strip()) > 0,
        "has_errors": len(stderr.strip()) > 0
    }

    # Try to extract error type if there are errors
    if result["has_errors"]:
        error_match = re.search(r"^(\w+Error):", stderr, re.MULTILINE)
        if error_match:
            result["error_type"] = error_match.group(1)

        # Try to extract line number of the error
        line_match = re.search(r"line (\d+)", stderr)
        if line_match:
            result["error_line"] = int(line_match.group(1))

    # Try to identify test results in stdout
    if result["has_output"]:
        # Look for test passed/failed markers
        passed_tests = len(re.findall(r"test.*?passed|passed.*?test", stdout, re.IGNORECASE))
        failed_tests = len(re.findall(r"test.*?failed|failed.*?test", stdout, re.IGNORECASE))

        if passed_tests > 0 or failed_tests > 0:
            result["test_results"] = {
                "passed": passed_tests,
                "failed": failed_tests,
                "total": passed_tests + failed_tests
            }

    return result


def extract_imports(code: str) -> List[str]:
    """
    Extract import statements from Python code.

    Args:
        code: Python code

    Returns:
        List of imported modules/names
    """
    imports = []

    # Match import statements
    import_matches = re.findall(r"^\s*import\s+([\w\s,]+)", code, re.MULTILINE)
    for match in import_matches:
        # Split multiple imports on the same line
        modules = [m.strip() for m in match.split(",")]
        imports.extend(modules)

    # Match from ... import statements
    from_import_matches = re.findall(r"^\s*from\s+([\w\.]+)\s+import\s+([\w\s,\*]+)", code, re.MULTILINE)
    for module, names in from_import_matches:
        imported_names = [n.strip() for n in names.split(",")]
        imports.extend([f"{module}.{name}" for name in imported_names if name != "*"])
        if "*" in imported_names:
            imports.append(f"{module}.*")

    return imports


def count_lines_of_code(code: str) -> Dict[str, int]:
    """
    Count lines of code, ignoring comments and blank lines.

    Args:
        code: Python code

    Returns:
        Dictionary with line counts
    """
    lines = code.splitlines()
    total_lines = len(lines)
    blank_lines = 0
    comment_lines = 0
    code_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith("#"):
            comment_lines += 1
        else:
            code_lines += 1

    return {
        "total_lines": total_lines,
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "code_lines": code_lines
    }


def extract_comments(code: str) -> List[str]:
    """
    Extract comments from Python code.

    Args:
        code: Python code

    Returns:
        List of comments
    """
    # Line comments
    line_comments = re.findall(r"^\s*#\s*(.*?)$", code, re.MULTILINE)

    # Docstrings
    try:
        tree = ast.parse(code)
        docstrings = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if ast.get_docstring(node):
                    docstrings.append(ast.get_docstring(node))
    except SyntaxError:
        # If code has syntax errors, return only line comments
        return line_comments

    # Combine all comments
    return line_comments + docstrings
