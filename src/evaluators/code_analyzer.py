# src/evaluators/code_analyzer.py

import os
import tempfile
import subprocess
import time
import re
import ast
from typing import Dict, Any, Tuple

from src.evaluators.base_evaluator import BaseEvaluator


class CodeAnalyzer(BaseEvaluator):
    """Evaluator that performs static analysis on Python code."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the code analyzer.

        Args:
            config: Analyzer configuration
        """
        super().__init__(config)

        # Extract configuration
        self.timeout = config.get("timeout", 30)
        self.python_path = config.get("python_path", "python")
        self.use_pylint = config.get("use_pylint", False)
        self.use_flake8 = config.get("use_flake8", True)
        self.use_mypy = config.get("use_mypy", False)
        self.check_complexity = config.get("check_complexity", True)
        self.complexity_threshold = config.get("complexity_threshold", 10)
        self.min_comments_ratio = config.get("min_comments_ratio", 0.1)

        # Pylint configuration
        self.pylint_options = config.get("pylint_options", ["--disable=C0111"])

        # Flake8 configuration
        self.flake8_options = config.get("flake8_options", ["--max-complexity=10", "--max-line-length=100"])

        # MyPy configuration
        self.mypy_options = config.get("mypy_options", ["--strict"])

    def evaluate(self, code: str) -> Tuple[str, str]:
        """
        Evaluate Python code by performing static analysis.

        Args:
            code: Python code to analyze

        Returns:
            Tuple of (output, errors)
        """
        self.logger.info("Performing static analysis on Python code")

        start_time = time.time()
        success = True

        try:
            # Save the code to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_filename = tmp_file.name

            self.logger.debug(f"Code saved to temporary file: {tmp_filename}")

            # Perform static analysis
            results = []
            errors_list = []

            # 1. AST Analysis
            ast_results, ast_success = self._analyze_ast(code)
            results.append(f"# AST Analysis:\n{ast_results}")
            if not ast_success:
                success = False
                errors_list.append("AST Analysis failed")

            # 2. Run linters if configured
            if self.use_pylint:
                pylint_output, pylint_success = self._run_pylint(tmp_filename)
                results.append(f"# Pylint Analysis:\n{pylint_output}")
                if not pylint_success:
                    success = False
                    errors_list.append("Pylint check failed")

            if self.use_flake8:
                flake8_output, flake8_success = self._run_flake8(tmp_filename)
                results.append(f"# Flake8 Analysis:\n{flake8_output}")
                if not flake8_success:
                    success = False
                    errors_list.append("Flake8 check failed")

            if self.use_mypy:
                mypy_output, mypy_success = self._run_mypy(tmp_filename)
                results.append(f"# MyPy Analysis:\n{mypy_output}")
                if not mypy_success:
                    success = False
                    errors_list.append("MyPy check failed")

            # Combine results
            output = "\n\n".join(results)
            errors = "\n".join(errors_list)

            execution_time = time.time() - start_time

            self.logger.info(f"Static analysis completed in {execution_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Error during static analysis: {str(e)}")
            output = ""
            errors = f"Analyzer Error: {str(e)}"
            execution_time = time.time() - start_time
            success = False

        finally:
            # Clean up the temporary file
            if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
                os.remove(tmp_filename)

        # Record metrics
        self._record_evaluation(success, execution_time)

        return output, errors

    def _analyze_ast(self, code: str) -> Tuple[str, bool]:
        """
        Analyze code using Python's AST.

        Args:
            code: Python code to analyze

        Returns:
            Tuple of (analysis_result, success)
        """
        try:
            tree = ast.parse(code)

            # Count various elements
            class CodeVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.function_count = 0
                    self.class_count = 0
                    self.import_count = 0
                    self.comment_lines = 0
                    self.code_lines = 0
                    self.complexity = {}
                    self.function_args = {}
                    self.used_variables = set()
                    self.defined_variables = set()
                    self.function_names = []
                    self.class_names = []
                    self.import_names = []

                def visit_FunctionDef(self, node):
                    self.function_count += 1
                    self.function_names.append(node.name)
                    self.function_args[node.name] = len(node.args.args)

                    # Calculate cyclomatic complexity
                    visitor = CyclomaticComplexityVisitor()
                    visitor.visit(node)
                    self.complexity[node.name] = visitor.complexity

                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    self.class_count += 1
                    self.class_names.append(node.name)
                    self.generic_visit(node)

                def visit_Import(self, node):
                    self.import_count += len(node.names)
                    for name in node.names:
                        self.import_names.append(name.name)
                    self.generic_visit(node)

                def visit_ImportFrom(self, node):
                    self.import_count += len(node.names)
                    if node.module:
                        for name in node.names:
                            self.import_names.append(f"{node.module}.{name.name}")
                    else:
                        for name in node.names:
                            self.import_names.append(name.name)
                    self.generic_visit(node)

                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):
                        self.defined_variables.add(node.id)
                    elif isinstance(node.ctx, ast.Load):
                        self.used_variables.add(node.id)
                    self.generic_visit(node)

            class CyclomaticComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.complexity = 1  # Start with 1 for the function itself

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

            # Count lines of code and comments
            lines = code.splitlines()
            code_lines = 0
            comment_lines = 0

            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    code_lines += 1
                elif stripped.startswith('#'):
                    comment_lines += 1

            # Run the visitor
            visitor = CodeVisitor()
            visitor.visit(tree)
            visitor.code_lines = code_lines
            visitor.comment_lines = comment_lines

            # Calculate metrics
            comment_ratio = comment_lines / max(1, code_lines)
            unused_vars = visitor.defined_variables - visitor.used_variables
            undefined_vars = visitor.used_variables - visitor.defined_variables - set(dir(__builtins__))

            # Filter out function and class names from undefined variables
            undefined_vars -= set(visitor.function_names)
            undefined_vars -= set(visitor.class_names)

            # Find complex functions
            complex_functions = {
                fname: complexity
                for fname, complexity in visitor.complexity.items()
                if complexity > self.complexity_threshold
            }

            # Prepare the analysis result
            result = [
                f"Lines of code: {code_lines}",
                f"Comment lines: {comment_lines}",
                f"Comment ratio: {comment_ratio:.2f}",
                f"Number of functions: {visitor.function_count}",
                f"Number of classes: {visitor.class_count}",
                f"Number of imports: {visitor.import_count}",
                f"Imports: {', '.join(visitor.import_names)}",
                f"Function complexity:",
            ]

            for fname, complexity in visitor.complexity.items():
                result.append(f"  - {fname}: {complexity}")

            if complex_functions:
                result.append("\nWarning: Complex functions detected:")
                for fname, complexity in complex_functions.items():
                    result.append(f"  - {fname}: {complexity} > {self.complexity_threshold}")

            if unused_vars:
                result.append("\nWarning: Unused variables:")
                for var in sorted(unused_vars):
                    result.append(f"  - {var}")

            if undefined_vars:
                result.append("\nWarning: Potentially undefined variables:")
                for var in sorted(undefined_vars):
                    result.append(f"  - {var}")

            # Check if comment ratio is too low
            if comment_ratio < self.min_comments_ratio:
                result.append(
                    f"\nWarning: Comment ratio ({comment_ratio:.2f}) is below recommended level ({self.min_comments_ratio})")

            # Determine success
            success = not (
                    complex_functions or
                    undefined_vars or
                    (comment_ratio < self.min_comments_ratio)
            )

            return "\n".join(result), success

        except SyntaxError as e:
            self.logger.error(f"Syntax error in AST analysis: {str(e)}")
            return f"Syntax error: {str(e)}", False

        except Exception as e:
            self.logger.error(f"Error in AST analysis: {str(e)}")
            return f"Analysis error: {str(e)}", False

    def _run_pylint(self, filename: str) -> Tuple[str, bool]:
        """
        Run pylint on the code.

        Args:
            filename: Path to the Python file

        Returns:
            Tuple of (pylint_output, success)
        """
        try:
            args = [self.python_path, '-m', 'pylint', filename]
            args.extend(self.pylint_options)

            self.logger.debug(f"Running pylint: {' '.join(args)}")

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Pylint returns non-zero exit code for warnings and errors
            # We'll consider it successful if the score is high enough
            success = True
            if result.returncode != 0:
                # Extract the score from the output
                score_match = re.search(r'Your code has been rated at (-?\d+\.\d+)/10', result.stdout)
                if score_match:
                    score = float(score_match.group(1))
                    success = score >= 7.0  # Consider a score >= 7.0 as success
                else:
                    success = False

            return result.stdout, success

        except subprocess.TimeoutExpired:
            self.logger.warning(f"pylint execution timed out after {self.timeout} seconds")
            return f"pylint execution timed out after {self.timeout} seconds", False

        except Exception as e:
            self.logger.error(f"Error running pylint: {str(e)}")
            return f"pylint error: {str(e)}", False

    def _run_flake8(self, filename: str) -> Tuple[str, bool]:
        """
        Run flake8 on the code.

        Args:
            filename: Path to the Python file

        Returns:
            Tuple of (flake8_output, success)
        """
        try:
            args = [self.python_path, '-m', 'flake8', filename]
            args.extend(self.flake8_options)

            self.logger.debug(f"Running flake8: {' '.join(args)}")

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # flake8 returns 0 if no issues found, non-zero otherwise
            success = result.returncode == 0

            # If no output and return code is 0, it passed
            if success and not result.stdout:
                return "No issues found", True

            return result.stdout, success

        except subprocess.TimeoutExpired:
            self.logger.warning(f"flake8 execution timed out after {self.timeout} seconds")
            return f"flake8 execution timed out after {self.timeout} seconds", False

        except Exception as e:
            self.logger.error(f"Error running flake8: {str(e)}")
            return f"flake8 error: {str(e)}", False

    def _run_mypy(self, filename: str) -> Tuple[str, bool]:
        """
        Run mypy on the code.

        Args:
            filename: Path to the Python file

        Returns:
            Tuple of (mypy_output, success)
        """
        try:
            args = [self.python_path, '-m', 'mypy', filename]
            args.extend(self.mypy_options)

            self.logger.debug(f"Running mypy: {' '.join(args)}")

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # mypy returns 0 if no issues found, non-zero otherwise
            success = result.returncode == 0

            # If no output and return code is 0, it passed
            if success and not result.stdout:
                return "No type issues found", True

            return result.stdout, success

        except subprocess.TimeoutExpired:
            self.logger.warning(f"mypy execution timed out after {self.timeout} seconds")
            return f"mypy execution timed out after {self.timeout} seconds", False

        except Exception as e:
            self.logger.error(f"Error running mypy: {str(e)}")
            return f"mypy error: {str(e)}", False
