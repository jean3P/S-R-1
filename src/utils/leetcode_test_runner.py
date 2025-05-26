# src/utils/leetcode_test_runner.py

import logging
import subprocess
import time
import sys
import os
import tempfile
from typing import Dict, Any, List
import json
import re
import uuid

logger = logging.getLogger(__name__)


class LeetCodeEnvironmentManager:
    """Manages testing in a shared conda environment for LeetCode."""

    def __init__(self, config=None):
        """Initialize with a persistent conda environment."""
        # Use the fixed environment path
        self.env_name = "env_leet_code_issues"
        self.env_path = "/storage/homefs/jp22b083/.conda/envs/env_leet_code_issues"

        # Track installed packages
        self.installed_packages = set()
        self._load_installed_packages()

        # Maximum retry attempts for package installation
        self.max_install_retries = 2

        # Store import failures to avoid repetitive installation attempts
        self.failed_packages = set()

        logger.info(f"Using persistent conda environment: {self.env_name}")

    def _load_installed_packages(self):
        """Load the list of already installed packages in the environment."""
        try:
            # Run pip list in the environment to get installed packages
            result = subprocess.run(
                f"conda run -n {self.env_name} pip list --format=json",
                shell=True, capture_output=True, text=True, check=True
            )

            packages_json = json.loads(result.stdout)
            for pkg in packages_json:
                self.installed_packages.add(pkg["name"].lower())

            logger.info(f"Loaded {len(self.installed_packages)} pre-installed packages")
        except Exception as e:
            logger.error(f"Error loading installed packages: {str(e)}")
            # Continue with empty set if we can't load the list

    def detect_imports(self, code):
        """Extract required package names from code that need installation."""
        # Pattern to match import statements
        import_pattern = re.compile(r'from\s+([a-zA-Z0-9_\.]+)[\.\s]+import|import\s+([a-zA-Z0-9_\.]+)')

        packages_to_check = set()
        for match in import_pattern.finditer(code):
            pkg_full = match.group(1) or match.group(2)
            if not pkg_full:
                continue

            # Extract base package name (before any dots)
            pkg_base = pkg_full.split('.')[0].lower()
            packages_to_check.add(pkg_base)

        # Now, check which packages need to be installed
        packages_to_install = []

        for pkg in packages_to_check:
            # Skip if already in installed or failed sets
            if pkg.lower() in self.installed_packages or pkg.lower() in self.failed_packages:
                continue

            # Try to import it in the conda environment to see if it's available
            check_cmd = f"conda run -n {self.env_name} python -c \"import {pkg}\" 2>/dev/null"
            result = subprocess.run(check_cmd, shell=True)

            # If import fails, it needs installation
            if result.returncode != 0:
                packages_to_install.append(pkg)

        return packages_to_install

    def install_required_packages(self, required_packages):
        """Install packages that aren't already installed."""
        packages_to_install = []

        # Check which packages need to be installed
        for pkg in required_packages:
            if pkg.lower() not in self.installed_packages and pkg.lower() not in self.failed_packages:
                packages_to_install.append(pkg)

        if not packages_to_install:
            return True, "All required packages already installed"

        logger.info(f"Installing packages: {', '.join(packages_to_install)}")

        # Try to install the packages
        try:
            pkg_str = " ".join(packages_to_install)
            result = subprocess.run(
                f"conda run -n {self.env_name} pip install {pkg_str} -q",
                shell=True, capture_output=True, text=True, timeout=60
            )

            # Check if installation was successful
            if result.returncode == 0:
                # Update the list of installed packages
                for pkg in packages_to_install:
                    self.installed_packages.add(pkg.lower())
                return True, "Successfully installed packages"
            else:
                # Mark packages as failed
                for pkg in packages_to_install:
                    self.failed_packages.add(pkg.lower())
                return False, f"Failed to install packages: {result.stderr}"
        except Exception as e:
            # Mark packages as failed
            for pkg in packages_to_install:
                self.failed_packages.add(pkg.lower())
            return False, f"Error installing packages: {str(e)}"

    def run_test(self, test_code, timeout=30):
        """Run test in the shared environment."""
        # Create a temporary file with the combined code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
            f.write(test_code.encode())
            f.flush()
            temp_filename = f.name

        start_time = time.time()
        try:
            # Detect required packages
            required_packages = self.detect_imports(test_code)

            # Try to install required packages
            install_success, install_message = self.install_required_packages(required_packages)

            # If package installation failed, return failure
            if not install_success:
                logger.warning(f"Cannot run test due to package installation failure: {install_message}")
                return {
                    "status": "import_error",
                    "error_message": f"Test environment setup failed: {install_message}",
                    "execution_time": 0,
                    "failed_tests": [],
                    "import_failures": required_packages
                }

            # Run the test in the conda environment
            result = subprocess.run(
                f"conda run -n {self.env_name} python {temp_filename}",
                shell=True, capture_output=True, text=True, timeout=timeout
            )

            execution_time = time.time() - start_time

            # Check for import errors in the output
            if "ModuleNotFoundError:" in result.stderr or "ImportError:" in result.stderr:
                # Find the failed import
                import_fail_match = re.search(r'No module named \'([^\']+)\'', result.stderr)
                failed_module = import_fail_match.group(1) if import_fail_match else "unknown"

                # Add to failed packages
                self.failed_packages.add(failed_module.lower())

                logger.warning(f"Test failed due to missing module: {failed_module}")
                return {
                    "status": "import_error",
                    "error_message": f"Missing required module: {failed_module}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": execution_time,
                    "import_failures": [failed_module]
                }

            # Determine if there are failed test cases
            failed_tests = []
            std_pattern = re.compile(r'assert\s+candidate\((.*?)\)\s*==\s*(.*?)(?:\n|$)')
            for match in std_pattern.finditer(result.stderr):  # or stderr_str in execute_test
                args = match.group(1).strip()
                expected = match.group(2).strip()

                # We don't have the actual value in the error message, so leave it as unknown
                failed_tests.append({
                    "input": args,
                    "expected": expected,
                    "actual": "unknown"  # We don't have this information in the error message
                })

            # Parse and return results
            return {
                "status": "pass" if result.returncode == 0 else "fail",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time,
                "failed_tests": failed_tests,
                "error_message": result.stderr.strip() if result.returncode != 0 else ""
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error_message": f"Test execution timed out after {timeout}s",
                "execution_time": timeout
            }
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except Exception:
                pass


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

        # Initialize environment manager if isolated testing is enabled
        self.use_isolated_env = config.get("leetcode", {}).get("use_isolated_env", True)
        if self.use_isolated_env:
            self.env_manager = LeetCodeEnvironmentManager(config)
            logger.info("Initialized LeetCode environment manager for testing")
        else:
            self.env_manager = None
            logger.info("Using direct testing without environment manager")

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
            if self.use_isolated_env:
                result = self.env_manager.run_test(combined_code, self.timeout)
            else:
                result = self.execute_test(combined_code)

            return result
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}", exc_info=True)
            return {"status": "error", "error_message": str(e)}

    def execute_test(self, test_code: str) -> Dict[str, Any]:
        """
        Execute a test script directly (without environment manager).
        """
        # Create a temporary file to store the test
        test_file = None
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

                # Check for import errors
                if "ModuleNotFoundError:" in stderr_str or "ImportError:" in stderr_str:
                    # Find the failed import
                    import_fail_match = re.search(r'No module named \'([^\']+)\'', stderr_str)
                    failed_module = import_fail_match.group(1) if import_fail_match else "unknown"

                    logger.warning(f"Test failed due to missing module: {failed_module}")
                    return {
                        "status": "import_error",
                        "error_message": f"Missing required module: {failed_module}",
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                        "execution_time": execution_time,
                        "import_failures": [failed_module]
                    }

                # Process the result
                if return_code == 0:
                    return {
                        "status": "pass",
                        "execution_time": execution_time
                    }
                else:
                    # Try to extract failed test info
                    failed_tests = []
                    std_pattern = re.compile(r'assert\s+candidate\((.*?)\)\s*==\s*(.*?)(?:\n|$)')
                    for match in std_pattern.finditer(stderr_str):  # or stderr_str in execute_test
                        args = match.group(1).strip()
                        expected = match.group(2).strip()

                        # We don't have the actual value in the error message, so leave it as unknown
                        failed_tests.append({
                            "input": args,
                            "expected": expected,
                            "actual": "unknown"  # We don't have this information in the error message
                        })

                    return {
                        "status": "fail",
                        "error_message": f"Test execution failed with return code {return_code}",
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                        "execution_time": execution_time,
                        "failed_tests": failed_tests
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
            if test_file and os.path.exists(test_file):
                os.unlink(test_file)

    def prepare_test_code(self, problem_data: Dict[str, Any], solution_code: str) -> str:
        """
        Build a syntactically-clean test script.
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
