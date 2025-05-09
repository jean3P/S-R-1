"""
Code evaluation module using HuggingFace's code_eval metric.
Provides standardized pass@k evaluation for code generation.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import tempfile
import traceback

logger = logging.getLogger(__name__)


class CodeEvaluator:
    """
    Evaluator for code solutions using Hugging Face's code_eval metric.
    """

    def __init__(self, config):
        """
        Initialize the code evaluator.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.evaluation_config = config.get("evaluation", {})

        # Get code_eval specific configuration
        self.code_eval_config = self.evaluation_config.get("code_eval", {})
        self.k_values = self.code_eval_config.get("k_values", [1, 3, 5, 10])
        self.num_workers = self.code_eval_config.get("num_workers", 4)
        self.timeout = self.code_eval_config.get("timeout", 3.0)

        # Set environment variable to allow code execution (with warning)
        if not os.environ.get("HF_ALLOW_CODE_EVAL"):
            logger.warning(
                "Setting HF_ALLOW_CODE_EVAL=1. This permits execution of untrusted model-generated code. "
                "Make sure this is running in a sandbox environment."
            )
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        # Load the code_eval metric
        try:
            from evaluate import load
            self.code_eval = load("code_eval")
            logger.info("Successfully loaded code_eval metric")
        except Exception as e:
            logger.error(f"Failed to load code_eval metric: {str(e)}")
            self.code_eval = None

    def evaluate_solutions(self, problem_data: Dict[str, Any], solutions: List[str]) -> Dict[str, Any]:
        """
        Evaluate a list of solutions against the problem's test cases.

        Args:
            problem_data: Problem data dictionary
            solutions: List of solution code strings

        Returns:
            Dictionary with evaluation metrics
        """
        if self.code_eval is None:
            logger.error("Cannot evaluate solutions: code_eval metric not loaded")
            return {"error": "code_eval metric not loaded"}

        if not solutions:
            logger.warning("No solutions to evaluate")
            return {"error": "no solutions provided"}

        # Create test cases from problem data
        test_cases = self._create_test_cases(problem_data)
        if not test_cases:
            logger.error("Failed to create test cases")
            return {"error": "failed to create test cases"}

        # Format solutions for code_eval
        formatted_solutions = [solutions]  # code_eval expects a list of lists

        try:
            # Compute pass@k using code_eval
            logger.info(f"Evaluating {len(solutions)} solutions with pass@{self.k_values}")
            pass_at_k, results = self.code_eval.compute(
                references=test_cases,
                predictions=formatted_solutions,
                k=self.k_values,
                num_workers=self.num_workers,
                timeout=self.timeout
            )

            # Format and return results
            evaluation = {
                "pass_at_k": pass_at_k,
                "detailed_results": results,
                "solutions_evaluated": len(solutions),
                "test_cases": len(test_cases)
            }

            return evaluation

        except Exception as e:
            logger.error(f"Error during code_eval evaluation: {str(e)}")
            logger.debug(traceback.format_exc())
            return {"error": str(e)}

    def _create_test_cases(self, problem_data: Dict[str, Any]) -> List[str]:
        """
        Create HuggingFace code_eval compatible test cases from problem data.

        Args:
            problem_data: Problem data dictionary

        Returns:
            List of test case strings
        """
        test_cases = []
        entry_point = problem_data.get("entry_point", "")

        # Extract method name from entry point
        method_name = ""
        if "." in entry_point:
            method_name = entry_point.split(".")[-1]

        # If we have test code, create a test wrapper
        if problem_data.get("test"):
            test_wrapper = f"""
def check_solution(candidate):
    import unittest
    import sys
    from io import StringIO

    # Store original stdout
    original_stdout = sys.stdout

    # Create a string buffer to capture output
    test_output = StringIO()
    sys.stdout = test_output

    try:
        # Execute test code
        {problem_data.get("test", "")}

        # Execute extra assertions
        {self._create_assertions_from_io(problem_data, method_name)}

        # If we reach here, all tests passed
        return True
    except AssertionError as e:
        # Test failure
        return False
    except Exception as e:
        # Other error
        return False
    finally:
        # Restore stdout
        sys.stdout = original_stdout

# Test the solution
result = check_solution({method_name})
assert result == True
"""
            test_cases.append(test_wrapper)

        # Create additional test cases from input-output pairs
        if problem_data.get("input_output"):
            for io_pair in problem_data.get("input_output", []):
                try:
                    input_data = io_pair.get("input", [])
                    expected_output = io_pair.get("output")

                    # Format inputs for assertion
                    if isinstance(input_data, list):
                        inputs_str = ", ".join(str(i) for i in input_data)
                    elif isinstance(input_data, dict):
                        inputs_str = ", ".join(f"{k}={v}" for k, v in input_data.items())
                    else:
                        inputs_str = str(input_data)

                    assertion = f"assert {method_name}({inputs_str}) == {expected_output}"
                    test_cases.append(assertion)
                except Exception as e:
                    logger.warning(f"Failed to create test case from I/O pair: {str(e)}")

        return test_cases

    def _create_assertions_from_io(self, problem_data: Dict[str, Any], method_name: str) -> str:
        """
        Create assertion statements from input-output pairs.

        Args:
            problem_data: Problem data dictionary
            method_name: Method name to test

        Returns:
            String with assertion statements
        """
        assertions = []

        for io_pair in problem_data.get("input_output", []):
            try:
                input_data = io_pair.get("input", [])
                expected_output = io_pair.get("output")

                # Format inputs for assertion
                if isinstance(input_data, list):
                    inputs_str = ", ".join(str(i) for i in input_data)
                elif isinstance(input_data, dict):
                    inputs_str = ", ".join(f"{k}={v}" for k, v in input_data.items())
                else:
                    inputs_str = str(input_data)

                assertion = f"assert {method_name}({inputs_str}) == {expected_output}"
                assertions.append(assertion)
            except Exception as e:
                logger.warning(f"Failed to create assertion from I/O pair: {str(e)}")

        return "\n".join(assertions)
