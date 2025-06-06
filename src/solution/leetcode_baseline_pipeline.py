# src/solution/leetcode_baseline_pipeline.py

import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
import gc

from ..models import create_model
from ..utils.leetcode_test_runner import LeetCodeTestRunner
from ..evaluation.code_evaluator import CodeEvaluator

logger = logging.getLogger(__name__)


class LeetCodeBaselinePipeline:
    """
    True baseline pipeline for LeetCode solutions - generates only 1 solution attempt.
    No tree search, no self-reflection, no multiple attempts.
    """

    def __init__(self, config, model_name=None):
        """
        Initialize the baseline LeetCode solution pipeline.

        Args:
            config: Configuration object.
            model_name: Name of the model to use, or None to use default.
        """
        self.config = config
        self.model_name = model_name or config.get("default_model", "deepseek-r1-distill")
        self.model = None  # Lazy-load model on demand

        # Initialize test runner
        self.test_runner = LeetCodeTestRunner(config)

        # Override to always generate only 1 solution for true baseline
        self.num_solutions = 1  # TRUE BASELINE: Only 1 attempt

        # Directory for storing results
        self.results_dir = Path(config.get("evaluation", {}).get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Memory optimization
        self.use_memory_optimization = config.get("memory_efficient", True)

        # Initialize code evaluator if enabled
        self.use_code_eval = config.get("evaluation", {}).get("use_code_eval", False)
        if self.use_code_eval:
            self.code_evaluator = CodeEvaluator(config)
            logger.info("Initialized code evaluator with HuggingFace code_eval")
        else:
            self.code_evaluator = None

        logger.info(f"Initialized LeetCodeBaselinePipeline with model {self.model_name}")
        logger.info(f"TRUE BASELINE MODE: Generating only 1 solution per problem")

    def _get_model(self):
        """Lazy-load model when needed."""
        if self.model is None:
            logger.info(f"Initializing model: {self.model_name}")
            self._run_memory_cleanup()
            self.model = create_model(self.model_name, self.config)
        return self.model

    def _run_memory_cleanup(self):
        """Run memory cleanup operations."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def solve_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a LeetCode problem by generating a single solution attempt.
        """
        logger.info(f"Starting baseline solution generation for problem {problem_data['problem_id']}")
        start_time = time.time()

        # Initialize tracking
        stats = {
            "num_generated": 0,
            "num_passed": 0,
            "generation_errors": 0,
            "test_errors": 0,
            "execution_time": 0.0
        }

        try:
            # Generate only 1 solution
            logger.info(f"Generating single baseline solution")

            # Generate the solution
            solution = self._generate_single_solution(problem_data)

            if solution:
                stats["num_generated"] = 1

                # Test the solution
                test_result = self._test_solution(problem_data, solution)

                # Track execution time
                if "execution_time" in test_result:
                    stats["execution_time"] = test_result["execution_time"]

                # Determine if passed
                passed = test_result.get("status") == "pass"
                if passed:
                    stats["num_passed"] = 1
                    logger.info(f"✓ Solution PASSED")
                else:
                    error_msg = test_result.get("error_message", "Unknown error")
                    logger.info(f"✗ Solution FAILED: {error_msg[:100]}...")

                # Prepare result
                result = {
                    "problem_id": problem_data["problem_id"],
                    "problem_title": problem_data["problem_title"],
                    "difficulty": problem_data.get("difficulty", ""),
                    "status": "solved" if passed else "unsolved",
                    "solution": solution,
                    "test_result": test_result,
                    "passed": passed,
                    "error_type": self._categorize_error(test_result.get("error_message", "")),
                    "stats": stats,
                    "processing_time": time.time() - start_time,
                    "baseline": True
                }

                # Perform code_eval for Pass@1 metric
                if self.code_evaluator:
                    logger.info(f"Evaluating solution with code_eval")
                    code_eval_results = self.code_evaluator.evaluate_solutions(
                        problem_data,
                        [solution]  # Single solution
                    )
                    result["code_eval_results"] = code_eval_results

                    # Log Pass@1 result
                    if "pass_at_k" in code_eval_results and "1" in code_eval_results["pass_at_k"]:
                        pass_at_1 = code_eval_results["pass_at_k"]["1"]
                        logger.info(f"Pass@1: {pass_at_1 * 100:.1f}%")

            else:
                stats["generation_errors"] = 1
                logger.warning(f"Failed to generate solution")

                result = {
                    "problem_id": problem_data["problem_id"],
                    "problem_title": problem_data["problem_title"],
                    "difficulty": problem_data.get("difficulty", ""),
                    "status": "error",
                    "error_message": "Failed to generate solution",
                    "stats": stats,
                    "processing_time": time.time() - start_time,
                    "baseline": True
                }

        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}", exc_info=True)
            result = {
                "problem_id": problem_data["problem_id"],
                "problem_title": problem_data["problem_title"],
                "difficulty": problem_data.get("difficulty", ""),
                "status": "error",
                "error_message": str(e),
                "stats": stats,
                "processing_time": time.time() - start_time,
                "baseline": True
            }

        # Save result to file
        self._save_results(problem_data["problem_id"], result)

        return result

    def _generate_single_solution(self, problem_data: Dict[str, Any]) -> str:
        """Generate a single solution for the problem."""
        model = self._get_model()

        self._run_memory_cleanup()

        # Create prompt - simple and direct for baseline
        prompt = self._create_baseline_prompt(problem_data)

        # Use low temperature for consistent baseline
        temperature = 0.1

        # Generate solution
        response = model.generate(prompt, temperature=temperature)

        # Extract solution code
        solution_code = self._extract_solution_code(response)

        return solution_code

    def _create_baseline_prompt(self, problem_data: Dict[str, Any]) -> str:
        """
        Create a simple, direct prompt for baseline solution generation.
        """
        imports_desc = problem_data['prompt']
        problem_desc = problem_data['query']

        prompt = f"""You are an expert Python programmer. Solve this LeetCode problem.

# Problem
Problem ID: {problem_data['problem_id']}  
Title: {problem_data['problem_title']}
Difficulty: {problem_data['difficulty']}

{problem_desc}

# Available Imports
{imports_desc}

# Required Function Signature
Entry Point: {problem_data['entry_point']}

# Instructions
1. Analyze the problem carefully
2. Write a correct Python solution
3. Use ONLY the imports provided above
4. Your solution must work for all test cases

# Output Format
Provide your solution in a Python code block:
    
        ```python
        class Solution:
            def method(self, params):
                # Your solution here
                return result
        ```
    
        Generate your solution now:"""

        return prompt

    def _extract_solution_code(self, response: str) -> str:
        """Extract solution code from model response."""
        import re

        # Try to find code block first
        code_block_pattern = r'```(?:python)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, response, re.IGNORECASE)

        if matches:
            # Take the last code block (usually the most complete)
            return matches[-1].strip()

        # Fallback: try to find class Solution
        class_pattern = r'(class\s+Solution[\s\S]+?)(?=\n\n|\Z)'
        class_match = re.search(class_pattern, response)

        if class_match:
            return class_match.group(1).strip()

        logger.warning("Could not extract solution code from response")
        return ""

    def _test_solution(self, problem_data: Dict[str, Any], solution: str) -> Dict[str, Any]:
        """Test a solution against the test cases."""
        # Fix indentation
        solution = self._fix_indentation(solution)

        # Run tests
        result = self.test_runner.run_tests(problem_data, solution)

        return result

    def _fix_indentation(self, code: str) -> str:
        """Fix indentation issues in Python code."""
        if not code or code.isspace():
            return code

        lines = code.split('\n')

        # Find minimum indentation
        min_indent = float('inf')
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                indent = len(line) - len(line.lstrip())
                if line.lstrip().startswith(('import ', 'from ', 'class ', 'def ')):
                    min_indent = min(min_indent, indent)

        if min_indent == float('inf'):
            min_indent = 0

        # Fix indentation
        fixed_lines = []
        for line in lines:
            if not line.strip():
                fixed_lines.append('')
            elif len(line) >= min_indent:
                fixed_lines.append(line[min_indent:])
            else:
                fixed_lines.append(line.lstrip())

        return '\n'.join(fixed_lines)

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message."""
        if not error_message:
            return "unknown"

        # Reuse categorization logic from main pipeline
        if "ModuleNotFoundError" in error_message or "ImportError" in error_message:
            return "import_error"
        elif "AssertionError" in error_message:
            return "assertion_failure"
        elif "IndexError" in error_message:
            return "index_error"
        elif "TypeError" in error_message:
            return "type_error"
        elif "ValueError" in error_message:
            return "value_error"
        elif "KeyError" in error_message:
            return "key_error"
        elif "RecursionError" in error_message:
            return "recursion_error"
        elif "TimeoutExpired" in error_message:
            return "timeout"
        else:
            return "other_error"

    def _save_results(self, problem_id: str, result: Dict[str, Any]) -> None:
        """Save the results to a file."""
        result_dir = self.results_dir / "baseline_results"
        result_dir.mkdir(parents=True, exist_ok=True)

        result_file = result_dir / f"{problem_id}.json"

        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

            logger.info(f"Saved baseline results to {result_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


