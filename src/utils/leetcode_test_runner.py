# src/utils/leetcode_test_runner.py

import logging
import subprocess
import time
import traceback
import sys
import os
import tempfile
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class LeetCodeTestRunner:
    """
    Utility for running tests on LeetCode solutions.
    """

    def __init__(self, config):
        """
        Initialize the test runner.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.timeout = config.get("leetcode", {}).get("test_timeout", 10)  # Timeout in seconds
        self.max_retries = config.get("leetcode", {}).get("max_test_retries", 2)

    def run_tests(self, problem_data: Dict[str, Any], solution_code: str) -> Dict[str, Any]:
        """
        Run tests for a solution.
        """
        try:
            # Log individual components
            logger.debug(f"IMPORTS:\n{problem_data.get('prompt', 'No imports found')}")
            logger.debug(f"TEST CODE:\n{problem_data.get('test', 'No test found')}")
            logger.debug(f"ENTRY POINT: {problem_data.get('entry_point', 'No entry point found')}")
            logger.debug(f"SOLUTION CODE:\n{solution_code}")

            # Prepare and log the combined code
            combined_code = self.prepare_test_code(problem_data, solution_code)
            logger.debug(f"EXECUTING TEST WITH COMBINED CODE:\n{combined_code}")

            # Execute the test
            result = self.execute_test(combined_code)
            return result
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}", exc_info=True)
            return {"status": "error", "error_message": str(e)}

    def _create_test_file(self, solution: str, test_code: str, input_output: List[Dict[str, Any]],
                          entry_point: str) -> str:
        """
        Create a complete test file with solution and tests.

        Args:
            solution: Solution code string.
            test_code: Test code string.
            input_output: List of input-output test cases.
            entry_point: The function entry point.

        Returns:
            Complete test file content.
        """
        # Extract method name from entry point
        method_name = ""
        if "." in entry_point:
            method_name = entry_point.split(".")[-1]

        # Create a test file that combines the solution and test code
        test_file = f"""
import time
import json
import traceback
import sys

# Solution code
{solution}

# Test result tracking
test_results = {{
    "status": "pass",
    "error_message": "",
    "output": "",
    "failed_tests": [],
    "execution_time": 0,
    "total_tests": 0,
    "passed_tests": 0
}}

# Test execution
try:
    start_time = time.time()

    # Execute test code
    {test_code}

    # Additional tests from input-output pairs
    additional_tests = {json.dumps(input_output)}

    for i, test_case in enumerate(additional_tests):
        try:
            test_results["total_tests"] += 1

            # Parse input and expected output
            inputs = test_case.get("input", [])
            expected = test_case.get("output")

            # Execute the solution
            solution = Solution()
            actual = None

            # Call the method dynamically
            if isinstance(inputs, list):
                actual = getattr(solution, "{method_name}", lambda *args: None)(
                    *inputs
                )
            elif isinstance(inputs, dict):
                actual = getattr(solution, "{method_name}", lambda **kwargs: None)(
                    **inputs
                )

            # Compare results
            if actual == expected:
                test_results["passed_tests"] += 1
            else:
                test_results["status"] = "fail"
                test_results["failed_tests"].append({{
                    "test_id": f"io_{{i+1}}",
                    "input": inputs,
                    "expected": expected,
                    "actual": actual
                }})

        except Exception as e:
            test_results["status"] = "fail"
            test_results["failed_tests"].append({{
                "test_id": f"io_{{i+1}}",
                "input": test_case.get("input", ""),
                "expected": test_case.get("output", ""),
                "error": str(e)
            }})

    test_results["execution_time"] = time.time() - start_time

except Exception as e:
    test_results["status"] = "error"
    test_results["error_message"] = str(e)
    test_results["output"] = traceback.format_exc()

# Print results as JSON
print(json.dumps(test_results))
"""
        return test_file

    def execute_test(self, test_code: str) -> Dict[str, Any]:
        """
        Execute a test script.
        """
        # Create a temporary file to store the test
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
                test_file = f.name
                f.write(test_code.encode('utf-8'))

            logger.debug(f"Test code written to temporary file: {test_file}")

            # Run the test with a timeout
            start_time = time.time()
            process = subprocess.Popen(
                [sys.executable, test_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            logger.debug(f"Test process started with PID: {process.pid}")

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                return_code = process.returncode

                # Convert bytes to string
                stdout_str = stdout.decode('utf-8', errors='replace')
                stderr_str = stderr.decode('utf-8', errors='replace')

                logger.debug(f"TEST STDOUT:\n{stdout_str}")
                logger.debug(f"TEST STDERR:\n{stderr_str}")
                logger.debug(f"TEST RETURN CODE: {return_code}")

                # Process the result
                if return_code == 0:
                    return {
                        "status": "pass",
                        "execution_time": execution_time
                    }
                else:
                    return {
                        "status": "fail",
                        "error_message": f"Test execution failed with return code {return_code}",
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                        "execution_time": execution_time
                    }
            except subprocess.TimeoutExpired:
                process.kill()
                logger.debug("TEST TIMEOUT: Process killed after exceeding timeout")
                return {
                    "status": "timeout",
                    "error_message": f"Test execution timed out after {self.timeout} seconds"
                }
        except Exception as e:
            logger.error(f"Error in test execution: {str(e)}", exc_info=True)
            return {"status": "error", "error_message": str(e)}
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.unlink(test_file)

    def prepare_test_code(self, problem_data: Dict[str, Any], solution_code: str) -> str:
        """
        Build a syntactically-clean test script:

          • std-lib / helper imports   (problem_data["prompt"])
          • the user’s solution code   (solution_code)
          • the author’s tests         (problem_data["test"])
          • a one-liner that calls     check(<entry_point>)
        """
        import textwrap

        imports = textwrap.dedent(problem_data.get("prompt", ""))
        tests = textwrap.dedent(problem_data.get("test", ""))
        entry_point = problem_data.get("entry_point", "").strip()

        # Use a list to guarantee every line starts at column 0
        lines: list[str] = [
            "# Imports ------------------------------------------------------------",
            imports,
            "",
            "# User solution -------------------------------------------------------",
            textwrap.dedent(solution_code),
            "",
            "# Author tests --------------------------------------------------------",
            tests,
            "",
            "# Kick off the tests --------------------------------------------------",
            'if __name__ == "__main__":',
            f"    check({entry_point})",
            "",
        ]

        combined_code = "\n".join(lines)

        logger.debug("PREPARED TEST CODE:\n%s", combined_code)
        return combined_code


