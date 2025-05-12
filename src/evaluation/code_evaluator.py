"""
Code evaluation module using HuggingFace's code_eval metric.
Provides standardized pass@k evaluation for code generation.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import tempfile
import traceback
import re
import textwrap

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

    def evaluate_solutions(
            self,
            problem_data: Dict[str, Any],
            solutions: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate candidate solutions with Hugging-Face `code_eval`.
        """
        if self.code_eval is None:
            logger.error("Cannot evaluate solutions: code_eval metric not loaded")
            return {"error": "code_eval metric not loaded"}

        if not solutions:
            logger.warning("No solutions to evaluate")
            return {"error": "no solutions provided"}

        # Create a test case for HumanEval format
        test_case = self._create_test_case(problem_data)
        if not test_case:
            logger.error("Failed to create test case")
            return {"error": "failed to create test case"}

        # Prepare solutions for evaluation
        prepared_solutions = []
        for solution in solutions:
            prepared_solution = self._prepare_complete_solution(solution, problem_data)
            prepared_solutions.append(prepared_solution)

        # For code_eval, we need one reference and one list of predictions
        references = [test_case]  # Single test case
        predictions = [prepared_solutions]  # Single list of solutions

        try:
            logger.info(
                f"Evaluating {len(prepared_solutions)} solution(s) with pass@{self.k_values}"
            )

            # Execute code evaluation
            pass_at_k, results = self.code_eval.compute(
                references=references,
                predictions=predictions,
                k=self.k_values,
                num_workers=self.num_workers,
                timeout=self.timeout,
            )

            return {
                "pass_at_k": pass_at_k,
                "detailed_results": results,
                "solutions_evaluated": len(prepared_solutions),
                "test_cases": 1,
            }

        except Exception as e:
            logger.error(f"Error during code_eval evaluation: {str(e)}")
            logger.debug(traceback.format_exc())
            return {"error": str(e)}

    def _create_test_case(self, problem_data: Dict[str, Any]) -> str:
        """
        Create a test case compatible with HuggingFace code_eval.
        The test case should execute and verify the solution.
        """
        # Extract necessary information
        entry_point = problem_data.get("entry_point", "")
        test_code = problem_data.get("test", "")

        # Parse entry point
        class_name, method_name = self._parse_entry_point(entry_point)

        # Extract the check function from the test code
        # The test code typically has a check function that takes a candidate
        test_case = f"""
# Extract test logic and execute it
{test_code}

# The check function expects a bound method, so we need to create one
solution_instance = Solution()
candidate_method = solution_instance.{method_name}

# Run the check
check(candidate_method)
"""

        return test_case.strip()

    def _prepare_complete_solution(self, solution: str, problem_data: Dict[str, Any]) -> str:
        """
        Prepare a complete solution including all necessary imports and test setup.
        """
        # Get imports from problem data
        prompt = problem_data.get("prompt", "")

        # Build complete solution
        complete_solution = []

        # Add standard imports
        complete_solution.append("""# Standard imports
import sys
from typing import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import random
import functools
import datetime
""")

        # Add any additional imports from the prompt
        if prompt:
            complete_solution.append("# Problem-specific imports")
            complete_solution.append(prompt)
            complete_solution.append("")

        # Add the solution
        complete_solution.append("# Solution")
        complete_solution.append(self._ensure_imports(solution))

        return "\n".join(complete_solution)

    def _parse_entry_point(self, entry_point: str) -> Tuple[str, str]:
        """
        Parse entry point into class name and method name.

        Args:
            entry_point: Entry point string like "Solution().twoSum" or "twoSum"

        Returns:
            Tuple of (class_name, method_name)
        """
        # Remove trailing parentheses if present
        entry_point = entry_point.replace("()", "")

        # Split on dot
        if "." in entry_point:
            parts = entry_point.split(".", 1)
            return parts[0], parts[1]
        else:
            # Default to Solution class
            return "Solution", entry_point

    def _ensure_imports(self, solution: str) -> str:
        """Add missing imports to a solution."""
        lines = solution.splitlines()
        imports_to_add = []

        # Track what's already imported
        existing_imports = "\n".join(lines)

        # Check for common imports that might be missing
        needs_typing = False
        typing_imports = []

        if "List[" in solution and "from typing import" not in existing_imports and "import typing" not in existing_imports:
            typing_imports.append("List")
            needs_typing = True

        if "Dict[" in solution and "from typing import" not in existing_imports and "import typing" not in existing_imports:
            typing_imports.append("Dict")
            needs_typing = True

        if "Optional[" in solution and "from typing import" not in existing_imports and "import typing" not in existing_imports:
            typing_imports.append("Optional")
            needs_typing = True

        if "Set[" in solution and "from typing import" not in existing_imports and "import typing" not in existing_imports:
            typing_imports.append("Set")
            needs_typing = True

        if "Tuple[" in solution and "from typing import" not in existing_imports and "import typing" not in existing_imports:
            typing_imports.append("Tuple")
            needs_typing = True

        if needs_typing and typing_imports:
            imports_to_add.append(f"from typing import {', '.join(typing_imports)}")

        # Collections imports
        if "deque(" in solution and "from collections import deque" not in existing_imports and "import collections" not in existing_imports:
            imports_to_add.append("from collections import deque")

        if "defaultdict(" in solution and "from collections import defaultdict" not in existing_imports and "import collections" not in existing_imports:
            imports_to_add.append("from collections import defaultdict")

        if "Counter(" in solution and "from collections import Counter" not in existing_imports and "import collections" not in existing_imports:
            imports_to_add.append("from collections import Counter")

        # Heapq imports
        if any(x in solution for x in ["heapq.", "heappush(", "heappop(", "heapify("]):
            if "import heapq" not in existing_imports and "from heapq import" not in existing_imports:
                imports_to_add.append("import heapq")

        # Math imports
        if "math." in solution or "sqrt(" in solution or "ceil(" in solution or "floor(" in solution:
            if "import math" not in existing_imports and "from math import" not in existing_imports:
                imports_to_add.append("import math")

        # Add imports at the beginning
        if imports_to_add:
            lines = imports_to_add + [""] + lines

        return "\n".join(lines)

    def _fix_indentation(self, code: str) -> str:
        """
        Fix indentation issues in Python code.
        """
        if not code or code.isspace():
            return code

        lines = code.split('\n')

        # Find minimum indentation for non-empty lines
        min_indent = float('inf')
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        if min_indent == float('inf'):
            min_indent = 0

        # Remove minimum indentation from all lines
        fixed_lines = []
        for line in lines:
            if line.strip():
                if len(line) >= min_indent:
                    fixed_lines.append(line[min_indent:])
                else:
                    fixed_lines.append(line.lstrip())
            else:
                fixed_lines.append('')

        return '\n'.join(fixed_lines)
