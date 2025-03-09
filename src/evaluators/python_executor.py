# src/evaluators/python_executor.py
import os
import tempfile
import subprocess
import time
import re
from typing import Dict, Any, Optional, List, Tuple, Union

from src.evaluators.base_evaluator import BaseEvaluator


class PythonExecutor(BaseEvaluator):
    """Evaluator that executes Python code."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Python executor.

        Args:
            config: Executor configuration
        """
        super().__init__(config)

        # Extract configuration
        self.timeout = config.get("timeout", 30)
        self.use_venv = config.get("use_venv", False)
        self.venv_path = config.get("venv_path")
        self.python_path = config.get("python_path", "python")
        self.install_dependencies = config.get("install_dependencies", False)
        self.allowed_imports = config.get("allowed_imports", None)  # None means all imports allowed
        self.forbidden_modules = config.get("forbidden_modules", [
            "os.system", "subprocess", "shutil.rmtree", "shutil.remove"
        ])

        # Extra testing options
        self.include_test_cases = config.get("include_test_cases", True)
        self.include_assertions = config.get("include_assertions", True)

    def evaluate(self, code: str) -> Tuple[str, str]:
        """
        Evaluate Python code by executing it.

        Args:
            code: Python code to execute

        Returns:
            Tuple of (stdout, stderr)
        """
        self.logger.info("Evaluating Python code")

        # Check for security issues
        security_check = self._check_code_security(code)
        if not security_check["safe"]:
            self.logger.warning(f"Code security check failed: {security_check['reason']}")
            return "", f"Security Error: {security_check['reason']}"

        # Create a temporary file for the code
        start_time = time.time()
        success = False

        try:
            # Save the code to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_filename = tmp_file.name

            self.logger.debug(f"Code saved to temporary file: {tmp_filename}")

            # Add test cases if configured
            if self.include_test_cases and self.config.get("test_cases"):
                test_code = self._generate_test_code(code)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as test_file:
                    test_file.write(test_code)
                    test_filename = test_file.name

                self.logger.debug(f"Test code saved to temporary file: {test_filename}")
            else:
                test_filename = None

            # Execute the code
            output, errors = self._execute_file(tmp_filename)

            # Execute tests if available
            if test_filename:
                test_output, test_errors = self._execute_file(test_filename)

                # Append test results to output
                if test_output:
                    output += f"\n\n# Test Results:\n{test_output}"
                if test_errors:
                    errors += f"\n\n# Test Errors:\n{test_errors}"

                # Clean up test file
                if os.path.exists(test_filename):
                    os.remove(test_filename)

            execution_time = time.time() - start_time
            success = len(errors.strip()) == 0

            self.logger.info(f"Code execution completed in {execution_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            output = ""
            errors = f"Evaluator Error: {str(e)}"
            execution_time = time.time() - start_time
            success = False

        finally:
            # Clean up the temporary file
            if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
                os.remove(tmp_filename)

        # Record metrics
        self._record_evaluation(success, execution_time)

        return output, errors

    def _execute_file(self, filename: str) -> Tuple[str, str]:
        """
        Execute a Python file.

        Args:
            filename: Path to the Python file

        Returns:
            Tuple of (stdout, stderr)
        """
        try:
            # Determine Python executable
            python_exe = self.python_path
            if self.use_venv and self.venv_path:
                if os.name == 'nt':  # Windows
                    python_exe = os.path.join(self.venv_path, 'Scripts', 'python.exe')
                else:  # Unix/Linux/Mac
                    python_exe = os.path.join(self.venv_path, 'bin', 'python')

            # Execute the file
            self.logger.debug(f"Executing: {python_exe} {filename}")

            result = subprocess.run(
                [python_exe, filename],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            output = result.stdout
            errors = result.stderr

            self.logger.debug(f"Execution return code: {result.returncode}")

            return output, errors

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Code execution timed out after {self.timeout} seconds")
            return "", f"Execution timed out after {self.timeout} seconds"

        except Exception as e:
            self.logger.error(f"Error executing file: {str(e)}")
            return "", f"Execution error: {str(e)}"

    def _check_code_security(self, code: str) -> Dict[str, Any]:
        """
        Check code for security issues.

        Args:
            code: Python code to check

        Returns:
            Dictionary with 'safe' boolean and 'reason' string if unsafe
        """
        # Check for forbidden modules/functions
        for forbidden in self.forbidden_modules:
            if forbidden in code:
                return {
                    "safe": False,
                    "reason": f"Use of forbidden module/function: {forbidden}"
                }

        # Check for specific imports if allowed_imports is specified
        if self.allowed_imports is not None:
            import_pattern = re.compile(r'^\s*import\s+(\w+)|^\s*from\s+(\w+)', re.MULTILINE)
            matches = import_pattern.findall(code)

            for match in matches:
                module = match[0] or match[1]  # Either 'import x' or 'from x'
                if module not in self.allowed_imports:
                    return {
                        "safe": False,
                        "reason": f"Import of non-allowed module: {module}"
                    }

        # Code is considered safe
        return {"safe": True}

    def _generate_test_code(self, code: str) -> str:
        """
        Generate test code for the given code.

        Args:
            code: Python code to test

        Returns:
            Test code
        """
        test_cases = self.config.get("test_cases", [])

        if not test_cases:
            return ""

        # Extract function name from the main code
        function_match = re.search(r'def\s+(\w+)\s*\(', code)
        if not function_match:
            self.logger.warning("Could not find a function definition in the code")
            return ""

        function_name = function_match.group(1)

        # Create the test code
        test_code = [
            "# Auto-generated test code",
            code,
            "",
            "# Test cases",
            "import sys",
            "",
            "def run_tests():",
            "    passed = 0",
            "    failed = 0",
            "    print('Running tests for function: " + function_name + "')",
            ""
        ]

        # Add test cases
        for i, test_case in enumerate(test_cases, 1):
            input_val = test_case.get("input")
            expected = test_case.get("expected")

            if input_val is None or expected is None:
                continue

            # Format the input (handle both single values and lists)
            if isinstance(input_val, (list, tuple)):
                input_str = ", ".join(repr(x) for x in input_val)
            else:
                input_str = repr(input_val)

            test_code.extend([
                f"    # Test case {i}",
                f"    try:",
                f"        result = {function_name}({input_str})",
                f"        expected = {repr(expected)}",
                f"        if result == expected:",
                f"            print(f\"✓ Test {i} passed: {function_name}({input_str}) == {repr(expected)}\")",
                f"            passed += 1",
                f"        else:",
                f"            print(f\"✗ Test {i} failed: {function_name}({input_str}) returned {{result}} instead of {repr(expected)}\")",
                f"            failed += 1",
                f"    except Exception as e:",
                f"        print(f\"✗ Test {i} error: {{str(e)}} when calling {function_name}({input_str})\")",
                f"        failed += 1",
                f""
            ])

        # Add summary
        test_code.extend([
            "    # Summary",
            "    print(f'Tests complete: {passed} passed, {failed} failed')",
            "    return passed, failed",
            "",
            "if __name__ == '__main__':",
            "    passed, failed = run_tests()",
            "    sys.exit(1 if failed > 0 else 0)"
        ])

        return "\n".join(test_code)

    def run_test_cases(self, code: str) -> Dict[str, Any]:
        """
        Run test cases on the code.

        Args:
            code: Code to test

        Returns:
            Test results
        """
        test_cases = self.config.get("test_cases", [])
        if not test_cases:
            return {"tested": False, "message": "No test cases defined"}

        # Generate test code
        test_code = self._generate_test_code(code)

        # Execute test code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as test_file:
            test_file.write(test_code)
            test_filename = test_file.name

        try:
            output, errors = self._execute_file(test_filename)

            # Parse test results
            passed = 0
            failed = 0
            details = []

            # Extract test results using regex
            pass_pattern = re.compile(r'✓ Test (\d+) passed')
            fail_pattern = re.compile(r'✗ Test (\d+) (failed|error)')

            for line in output.split('\n'):
                if '✓ Test' in line:
                    passed += 1
                    match = pass_pattern.search(line)
                    if match:
                        test_num = int(match.group(1))
                        details.append({
                            "test_case": test_num,
                            "passed": True,
                            "message": line
                        })
                elif '✗ Test' in line:
                    failed += 1
                    match = fail_pattern.search(line)
                    if match:
                        test_num = int(match.group(1))
                        details.append({
                            "test_case": test_num,
                            "passed": False,
                            "message": line
                        })

            results = {
                "tested": True,
                "total": len(test_cases),
                "passed": passed,
                "failed": failed,
                "output": output,
                "errors": errors,
                "details": details
            }

            return results

        except Exception as e:
            self.logger.error(f"Error running test cases: {str(e)}")
            return {
                "tested": True,
                "success": False,
                "error": str(e)
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(test_filename):
                os.remove(test_filename)
