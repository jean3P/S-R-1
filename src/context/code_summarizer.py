# src/context/code_summarizer.py

import ast
import os
from typing import Dict, List, Any, Optional

from src.utils.logging import get_logger
from src.utils.parsing import extract_docstring


class CodeSummarizer:
    """
    Creates compact representations of code files, functions, and classes.
    Used to efficiently represent code for LLM context without wasting tokens.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}
        self.max_summary_length = self.config.get("max_summary_length", 500)
        self.include_docstrings = self.config.get("include_docstrings", True)
        self.include_imports = self.config.get("include_imports", True)

    def summarize_file(self, file_path: str) -> Dict[str, Any]:
        """
        Summarize a Python file to its essential components.

        Args:
            file_path: Path to the file to summarize

        Returns:
            Dictionary containing file summary information
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the Python code
            tree = ast.parse(content)

            # Extract imports
            imports = []
            if self.include_imports:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(f"import {name.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for name in node.names:
                            imports.append(f"from {module} import {name.name}")

            # Extract functions and classes
            functions = []
            classes = []

            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    func_info = self.summarize_function_node(node)
                    functions.append(func_info)
                elif isinstance(node, ast.ClassDef):
                    class_info = self.summarize_class_node(node)
                    classes.append(class_info)

            # Construct file summary
            file_summary = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "imports": imports,
                "functions": functions,
                "classes": classes,
                "summary_type": "file"
            }

            return file_summary

        except Exception as e:
            self.logger.error(f"Error summarizing file {file_path}: {e}")
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "error": str(e)
            }

    def summarize_function_node(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Summarize a function AST node.

        Args:
            node: AST node for the function

        Returns:
            Dictionary with function summary
        """
        # Extract signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        # Extract docstring
        docstring = extract_docstring(node) if self.include_docstrings else None

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

        return {
            "name": node.name,
            "args": args,
            "returns": returns,
            "docstring": docstring,
            "complexity": statement_count,
            "line_count": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else None,
            "type": "function"
        }

    def summarize_class_node(self, node: ast.ClassDef) -> Dict[str, Any]:
        """
        Summarize a class AST node.

        Args:
            node: AST node for the class

        Returns:
            Dictionary with class summary
        """
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{base.value.id}.{base.attr}")

        # Extract docstring
        docstring = extract_docstring(node) if self.include_docstrings else None

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self.summarize_function_node(item))

        return {
            "name": node.name,
            "bases": bases,
            "docstring": docstring,
            "methods": methods,
            "method_count": len(methods),
            "line_count": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else None,
            "type": "class"
        }

    def summarize_function(self, function_code: str) -> Dict[str, Any]:
        """
        Summarize a function from its code string.

        Args:
            function_code: String containing function code

        Returns:
            Dictionary with function summary
        """
        try:
            tree = ast.parse(function_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return self.summarize_function_node(node)

            return {"error": "No function found in the provided code"}
        except Exception as e:
            self.logger.error(f"Error summarizing function: {e}")
            return {"error": str(e)}

    def generate_file_index(self, project_files: List[str]) -> Dict[str, Any]:
        """
        Generate an index of all files in the project.

        Args:
            project_files: List of file paths in the project

        Returns:
            Dictionary with project file index
        """
        index = {
            "files": {},
            "classes": {},
            "functions": {}
        }

        for file_path in project_files:
            if not file_path.endswith('.py'):
                continue

            file_summary = self.summarize_file(file_path)
            index["files"][file_path] = {
                "name": file_summary.get("file_name"),
                "classes": [c.get("name") for c in file_summary.get("classes", [])],
                "functions": [f.get("name") for f in file_summary.get("functions", [])]
            }

            # Index classes
            for cls in file_summary.get("classes", []):
                class_name = cls.get("name")
                index["classes"][class_name] = {
                    "file": file_path,
                    "methods": [m.get("name") for m in cls.get("methods", [])]
                }

            # Index functions
            for func in file_summary.get("functions", []):
                func_name = func.get("name")
                index["functions"][func_name] = {
                    "file": file_path
                }

        return index
    