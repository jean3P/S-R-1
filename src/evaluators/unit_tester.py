# src/evaluators/unit_tester.py
import os
import tempfile
import subprocess
import time
import re
from typing import Dict, Any, Tuple

from src.evaluators.base_evaluator import BaseEvaluator


class UnitTester(BaseEvaluator):
    """Evaluator that runs unit tests on Python code."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the unit tester.

        Args:
            config: Tester configuration
        """
        super().__init__(config)

        # Extract configuration
        self.timeout = config.get("timeout", 30)
        self.python_path = config.get("python_path", "python")
        self.use_pytest = config.get("use_pytest", False)
        self.verbose = config.get("verbose", True)

        # Test parametrization
        self.test_template = config.get("test_template")
        self.test_cases = config.get("test_cases", [])
        self.assertions = config.get("assertions", [])

    def evaluate(self, code: str) -> Tuple[str, str]:
        """
        Evaluate Python code by running unit tests.

        Args:
            code: Python code to test

        Returns:
            Tuple of (stdout, stderr)
        """
        self.logger.info("Running unit tests on Python code")

        start_time = time.time()
        success = False

        try:
            # Extract function name and code
            function_match = re.search(r'def\s+(\w+)\s*\(', code)
            if not function_match:
                self.logger.warning("Could not find a function definition in the code")
                return "", "Error: No function definition found in the code"

            function_name = function_match.group(1)

            # Generate test code
            test_code = self._generate_test_code(code, function_name)

            # Save the code and test code to temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
                code_file.write(code)
                code_filename = code_file.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as test_file:
                test_file.write(test_code)
                test_filename = test_file.name

            self.logger.debug(f"Code saved to: {code_filename}")
            self.logger.debug(f"Test code saved to: {test_filename}")

            # Run the tests
            if self.use_pytest:
                output, errors = self._run_pytest(test_filename)
            else:
                output, errors = self._run_unittest(test_filename)

            execution_time = time.time() - start_time
            success = "FAILED" not in output and len(errors.strip()) == 0

            self.logger.info(f"Unit tests completed in {execution_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Error during testing: {str(e)}")
            output = ""
            errors = f"Testing Error: {str(e)}"
            execution_time = time.time() - start_time
            success = False

        finally:
            # Clean up the temporary files
            if 'code_filename' in locals() and os.path.exists(code_filename):
                os.remove(code_filename)
            if 'test_filename' in locals() and os.path.exists(test_filename):
                os.remove(test_filename)

        # Record metrics
        self._record_evaluation(success, execution_time)

        return output, errors

    def _generate_test_code(self, code: str, function_name: str) -> str:
        """
        Generate unit test code.

        Args:
            code: Python code containing the function to test
            function_name: Name of the function to test

        Returns:
            Unit test code
        """
        if self.test_template:
            # Use the provided template
            test_code = self.test_template.format(
                function_name=function_name,
                code=code
            )
        else:
            # Generate a default unittest template
            module_name = "solution"  # We'll name the module 'solution'

            test_code = [
                "import unittest",
                "import sys",
                "import os",
                "import importlib.util",
                "",
                f"# Load the solution code as a module",
                f"def load_solution(filename):",
                f"    module_name = '{module_name}'",
                f"    spec = importlib.util.spec_from_file_location(module_name, filename)",
                f"    module = importlib.util.module_from_spec(spec)",
                f"    spec.loader.exec_module(module)",
                f"    return module",
                "",
                f"# Find the solution file (it's in the same directory as this test file)",
                f"current_dir = os.path.dirname(os.path.abspath(__file__))",
                f"solution_file = os.path.join(current_dir, '{os.path.basename(locals().get('code_filename', 'solution.py'))}')",
                f"solution = load_solution(solution_file)",
                "",
                f"class Test{function_name.capitalize()}(unittest.TestCase):"
            ]

            # Add test methods for each test case
            for i, test_case in enumerate(self.test_cases, 1):
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
                    f"    def test_{i}_{function_name}(self):",
                    f"        \"\"\"Test case {i}: {function_name}({input_str}) should return {repr(expected)}\"\"\"",
                    f"        result = solution.{function_name}({input_str})",
                    f"        self.assertEqual(result, {repr(expected)})"
                ])

            # Add custom assertions if provided
            for i, assertion in enumerate(self.assertions, 1):
                test_code.extend([
                    f"    def test_assertion_{i}(self):",
                    f"        \"\"\"Custom assertion {i}\"\"\"",
                    f"        {assertion}"
                ])

            # Add main block
            test_code.extend([
                "",
                "if __name__ == '__main__':",
                "    unittest.main()"
            ])

            test_code = "\n".join(test_code)

        return test_code

    def _run_unittest(self, test_filename: str) -> Tuple[str, str]:
        """
        Run tests using unittest.

        Args:
            test_filename: Path to the test file

        Returns:
            Tuple of (stdout, stderr)
        """
        try:
            args = [self.python_path, '-m', 'unittest', test_filename]
            if self.verbose:
                args.insert(3, '-v')

            self.logger.debug(f"Running unittest: {' '.join(args)}")

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.warning(f"unittest execution timed out after {self.timeout} seconds")
            return "", f"unittest execution timed out after {self.timeout} seconds"

        except Exception as e:
            self.logger.error(f"Error running unittest: {str(e)}")
            return "", f"unittest error: {str(e)}"

    def _run_pytest(self, test_filename: str) -> Tuple[str, str]:
        """
        Run tests using pytest.

        Args:
            test_filename: Path to the test file

        Returns:
            Tuple of (stdout, stderr)
        """
        try:
            args = [self.python_path, '-m', 'pytest', test_filename]
            if self.verbose:
                args.append('-v')

            self.logger.debug(f"Running pytest: {' '.join(args)}")

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.warning(f"pytest execution timed out after {self.timeout} seconds")
            return "", f"pytest execution timed out after {self.timeout} seconds"

        except Exception as e:
            self.logger.error(f"Error running pytest: {str(e)}")
            return "", f"pytest error: {str(e)}"

    def run_test_cases(self, code: str) -> Dict[str, Any]:
        """
        Run test cases on the code.

        Args:
            code: Code to test

        Returns:
            Test results
        """
        output, errors = self.evaluate(code)

        # Parse unittest/pytest output to extract results
        if self.use_pytest:
            return self._parse_pytest_output(output, errors)
        else:
            return self._parse_unittest_output(output, errors)

    def _parse_unittest_output(self, output: str, errors: str) -> Dict[str, Any]:
        """
        Parse unittest output.

        Args:
            output: stdout from unittest
            errors: stderr from unittest

        Returns:
            Parsed test results
        """
        if not output and not errors:
            return {"tested": False, "message": "No test output"}

        # Check for test failures
        if "FAIL:" in output or "ERROR:" in output:
            success = False
        else:
            success = True

        # Extract test stats
        total_pattern = re.search(r'Ran (\d+) test', output)
        total = int(total_pattern.group(1)) if total_pattern else 0

        # Count failures
        failures = output.count("FAIL:")
        errors_count = output.count("ERROR:")

        # Calculate passed tests
        passed = total - (failures + errors_count)

        # Extract test details
        details = []
        test_pattern = re.compile(r'(test_\w+) \(([\w\.]+)\) \.\.\. (ok|FAIL|ERROR)')
        for match in test_pattern.finditer(output):
            test_name = match.group(1)
            test_result = match.group(3)
            details.append({
                "test_name": test_name,
                "passed": test_result == "ok",
                "result": test_result
            })

        return {
            "tested": True,
            "success": success,
            "total": total,
            "passed": passed,
            "failed": failures,
            "errors": errors_count,
            "output": output,
            "error_output": errors,
            "details": details
        }

    def _parse_pytest_output(self, output: str, errors: str) -> Dict[str, Any]:
        """
        Parse pytest output.

        Args:
            output: stdout from pytest
            errors: stderr from pytest

        Returns:
            Parsed test results
        """
        if not output and not errors:
            return {"tested": False, "message": "No test output"}

        # Extract test summary
        summary_pattern = re.search(r'(\d+) passed, (\d+) failed(, (\d+) error)?', output)
        if summary_pattern:
            passed = int(summary_pattern.group(1))
            failed = int(summary_pattern.group(2))
            errors_count = int(summary_pattern.group(4)) if summary_pattern.group(4) else 0
            total = passed + failed + errors_count
            success = failed == 0 and errors_count == 0
        else:
            # Fallback if summary not found
            passed = output.count("PASSED")
            failed = output.count("FAILED")
            errors_count = 0
            total = passed + failed
            success = failed == 0

        # Extract test details
        details = []
        test_pattern = re.compile(r'(test_\w+)(?:\[\w+\])? (PASSED|FAILED|ERROR|XFAILED|XPASSED)')
        for match in test_pattern.finditer(output):
            test_name = match.group(1)
            test_result = match.group(2)
            details.append({
                "test_name": test_name,
                "passed": test_result in ["PASSED", "XPASSED"],
                "result": test_result
            })

        return {
            "tested": True,
            "success": success,
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors_count,
            "output": output,
            "error_output": errors,
            "details": details
        }
