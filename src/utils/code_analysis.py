# src/utils/code_analysis.py

import ast
from typing import Dict, Any
from src.utils.parsing import extract_docstring


def extract_function_info(node: ast.FunctionDef) -> Dict[str, Any]:
    """
    Extract information from a function AST node.

    Args:
        node: Function definition AST node

    Returns:
        Dictionary with function information
    """
    # Extract arguments
    args = []
    for arg in node.args.args:
        args.append(arg.arg)

    # Extract docstring
    docstring = extract_docstring(node)

    # Count statements to gauge complexity
    statement_count = sum(1 for _ in ast.walk(node) if isinstance(_, (
        ast.Assign, ast.AugAssign, ast.Return, ast.Expr, ast.If, ast.For, ast.While
    )))

    # Extract return annotation if available
    returns = None
    if node.returns:
        if isinstance(node.returns, ast.Name):
            returns = node.returns.id
        elif isinstance(node.returns, ast.Subscript):
            # Handle cases like List[str]
            returns = ast.unparse(node.returns)
        else:
            try:
                returns = ast.unparse(node.returns)
            except:
                # Fallback for older Python versions
                returns = str(node.returns)

    return {
        "name": node.name,
        "args": args,
        "returns": returns,
        "docstring": docstring,
        "complexity": statement_count,
        "line_count": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else None,
        "type": "function"
    }


def extract_class_info(node: ast.ClassDef) -> Dict[str, Any]:
    """
    Extract information from a class AST node.

    Args:
        node: Class definition AST node

    Returns:
        Dictionary with class information
    """
    # Extract base classes
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            try:
                bases.append(f"{base.value.id}.{base.attr}")
            except AttributeError:
                bases.append(str(base))
        else:
            try:
                bases.append(ast.unparse(base))
            except:
                # Fallback for older Python versions
                bases.append(str(base))

    # Extract docstring
    docstring = extract_docstring(node)

    # Extract methods
    methods = []
    attributes = []

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            methods.append(extract_function_info(item))
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    attributes.append({
                        "name": target.id,
                        "type": "attribute"
                    })

    return {
        "name": node.name,
        "bases": bases,
        "docstring": docstring,
        "methods": methods,
        "attributes": attributes,
        "method_count": len(methods),
        "line_count": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else None,
        "type": "class"
    }


def analyze_code_file(code: str) -> Dict[str, Any]:
    """
    Analyze a Python code file.

    Args:
        code: Python code as a string

    Returns:
        Dictionary with file analysis
    """
    try:
        tree = ast.parse(code)

        # Extract imports
        imports = []
        import_froms = []

        # Extract top-level functions and classes
        functions = []
        classes = []

        # Extract global variables
        variables = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    import_froms.append(f"{module}.{name.name}")
            elif isinstance(node, ast.FunctionDef):
                functions.append(extract_function_info(node))
            elif isinstance(node, ast.ClassDef):
                classes.append(extract_class_info(node))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append({
                            "name": target.id,
                            "type": "variable"
                        })

        # Calculate complexity metrics
        complexity = len(functions) + len(classes) * 2 + sum(f.get("complexity", 0) for f in functions)

        return {
            "imports": imports,
            "import_froms": import_froms,
            "functions": functions,
            "classes": classes,
            "variables": variables,
            "complexity": complexity
        }

    except Exception as e:
        return {
            "error": str(e),
            "imports": [],
            "functions": [],
            "classes": []
        }
