# src/utils/code_analyzer.py

import ast
import re
from typing import List, Set, Optional


class CodeAnalyzer:
    """
    A utility class for analyzing Python code structure.
    Used by the progressive disclosure system to extract focused information.
    """

    @staticmethod
    def find_component_in_ast(tree: ast.AST, target_name: str, component_type: Optional[str] = None) -> Optional[
        ast.AST]:
        """
        Find a component (class or function) in an AST.

        Args:
            tree: The AST to search
            target_name: Name of component to find
            component_type: Type of component ("class", "function", or None for both)

        Returns:
            The AST node if found, None otherwise
        """
        # Handle namespaced references (Class.method)
        if '.' in target_name:
            parts = target_name.split('.')
            if len(parts) == 2:
                class_name, method_name = parts

                # Find the class first
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        # Then find the method in the class
                        for child in node.body:
                            if isinstance(child, ast.FunctionDef) and child.name == method_name:
                                return child
                return None

        # Direct component search
        for node in ast.walk(tree):
            if component_type == "class" and isinstance(node, ast.ClassDef) and node.name == target_name:
                return node
            elif component_type == "function" and isinstance(node, ast.FunctionDef) and node.name == target_name:
                return node
            elif component_type is None and (
                    (isinstance(node, ast.ClassDef) and node.name == target_name) or
                    (isinstance(node, ast.FunctionDef) and node.name == target_name)
            ):
                return node

        return None

    @staticmethod
    def extract_function_calls(tree: ast.AST) -> Set[str]:
        """
        Extract all function calls from an AST.

        Args:
            tree: The AST to analyze

        Returns:
            Set of function/method names that are called
        """
        function_calls = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Direct function call: function()
                    function_calls.add(node.func.id)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    # Method call: obj.method()
                    function_calls.add(node.func.attr)

        return function_calls

    @staticmethod
    def extract_imports(tree: ast.AST) -> List[str]:
        """
        Extract import statements from an AST.

        Args:
            tree: The AST to analyze

        Returns:
            List of import statements
        """
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(f"import {name.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append(f"from {module} import {name.name}")

        return imports

    @staticmethod
    def find_inheritance_relationships(tree: ast.AST, class_name: str) -> List[str]:
        """
        Find subclasses that inherit from a given class in an AST.

        Args:
            tree: The AST to analyze
            class_name: Name of the parent class

        Returns:
            List of subclass names
        """
        subclasses = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == class_name:
                        subclasses.append(node.name)
                    elif isinstance(base, ast.Attribute) and base.attr == class_name:
                        subclasses.append(node.name)

        return subclasses

    @staticmethod
    def extract_component_source(source_code: str, node: ast.AST) -> str:
        """
        Extract the source code for a component from the original source and an AST node.

        Args:
            source_code: Original source code
            node: AST node for the component

        Returns:
            Source code of the component
        """
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return ""

        lines = source_code.split('\n')
        start_line = node.lineno - 1  # AST line numbers are 1-based
        end_line = node.end_lineno

        return '\n'.join(lines[start_line:end_line])

    @staticmethod
    def extract_signature(node: ast.AST) -> Optional[str]:
        """
        Extract a signature from an AST node (function or class).

        Args:
            node: AST node

        Returns:
            Signature as a string, or None if not applicable
        """
        if isinstance(node, ast.FunctionDef):
            # Function signature
            args = []
            for arg in node.args.args:
                if hasattr(arg, 'annotation') and arg.annotation is not None:
                    # Try to get type annotation
                    try:
                        annotation = ast.unparse(arg.annotation)
                        args.append(f"{arg.arg}: {annotation}")
                    except:
                        args.append(arg.arg)
                else:
                    args.append(arg.arg)

            # Return annotation
            returns = ""
            if hasattr(node, 'returns') and node.returns is not None:
                try:
                    returns = f" -> {ast.unparse(node.returns)}"
                except:
                    pass

            return f"def {node.name}({', '.join(args)}){returns}"

        elif isinstance(node, ast.ClassDef):
            # Class signature
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(f"{base.value.id}.{base.attr}")

            return f"class {node.name}({', '.join(bases)})"

        return None

    @staticmethod
    def truncate_implementation(implementation: str, max_tokens: int, model_name: str,
                                token_counter_func) -> str:
        """
        Truncate an implementation to fit within token limits.

        Args:
            implementation: The implementation code
            max_tokens: Maximum tokens allowed
            model_name: Name of the model for token counting
            token_counter_func: Function to count tokens

        Returns:
            Truncated implementation
        """
        if not implementation:
            return ""

        tokens = token_counter_func(implementation, model_name)

        if tokens <= max_tokens:
            return implementation

        lines = implementation.split('\n')

        # If just a few lines, return as is
        if len(lines) <= 5:
            return implementation

        # Keep header and an important part of the implementation
        header_lines = 3
        footer_lines = 3

        # Check for docstring and include it in header
        docstring_end = header_lines
        for i, line in enumerate(lines[header_lines:header_lines + 5]):
            if '"""' in line or "'''" in line:
                docstring_end = header_lines + i + 2

        header = lines[:max(header_lines, docstring_end)]
        footer = lines[-footer_lines:]

        # Check if there are any important structural elements in the middle
        important_patterns = [
            r'^\s*def\s+\w+',  # Method definition
            r'^\s*class\s+\w+',  # Class definition
            r'^\s*if\s+__name__\s*==\s*',  # Main block
            r'^\s*return\s+'  # Return statement
        ]

        # Find important lines in the middle
        important_lines = []
        for i, line in enumerate(lines):
            if i < len(header) or i >= len(lines) - len(footer):
                continue

            for pattern in important_patterns:
                if re.match(pattern, line):
                    important_lines.append((i, line))
                    break

        # Take a sample of important lines if there are many
        if len(important_lines) > 5:
            sampled_indices = [i for i, _ in important_lines]
            sampled_indices.sort()
            step = len(sampled_indices) // 5
            sampled_indices = sampled_indices[::step][:5]
            important_lines = [(i, lines[i]) for i in sampled_indices]

        # Construct the truncated implementation
        result = '\n'.join(header)
        result += "\n\n# ... [implementation truncated] ...\n"

        if important_lines:
            result += "\n# Important snippets:\n"
            for _, line in important_lines:
                result += line + "\n"

        result += "\n# ... [implementation truncated] ...\n\n"
        result += '\n'.join(footer)

        return result
