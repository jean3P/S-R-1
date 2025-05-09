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


class LeetCodeSolutionPipeline:
    """
    Pipeline for generating LeetCode solutions with multi-round self-reflection.
    Generates multiple candidate solutions and iteratively improves them.
    """

    def __init__(self, config, model_name=None):
        """
        Initialize the LeetCode solution pipeline.

        Args:
            config: Configuration object.
            model_name: Name of the model to use, or None to use default.
        """
        self.config = config
        self.model_name = model_name or config.get("default_model", "deepseek-r1-distill")
        self.model = None  # Lazy-load model on demand

        # Initialize test runner
        self.test_runner = LeetCodeTestRunner(config)

        # Configure parameters
        self.num_candidates = config.get("leetcode", {}).get("num_candidates", 3)
        self.num_reflection_rounds = config.get("leetcode", {}).get("reflection_rounds", 3)
        self.max_test_retries = config.get("leetcode", {}).get("max_test_retries", 2)

        # Cache for tested solutions
        self.solution_cache = {}

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

        logger.info(f"Initialized LeetCodeSolutionPipeline with model {self.model_name}")

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
        Solve a LeetCode problem using multi-round self-reflection.

        Args:
            problem_data: Problem data dictionary.

        Returns:
            Dictionary with solution results.
        """
        logger.info(f"Starting solution generation for problem {problem_data['problem_id']}")
        start_time = time.time()

        # Initialize tracking
        reflection_logs = []
        candidates = []
        final_solutions = []
        best_solution = None
        solution_found = False

        # Track statistics
        stats = {
            "rounds_completed": 0,
            "candidates_generated": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_errors": 0,
            "execution_times": []
        }

        # Round 1: Initial solutions using Chain of Thought
        try:
            round_start_time = time.time()
            initial_candidates = self._generate_initial_candidates(problem_data)

            round_log = {
                "round": 1,
                "phase": "Initial Generation",
                "candidates": [],
                "execution_time": time.time() - round_start_time
            }

            # Test each candidate
            for i, candidate in enumerate(initial_candidates):
                candidate_hash = self._calculate_solution_hash(candidate)
                test_result = self._test_solution(problem_data, candidate)

                candidate_log = {
                    "candidate_id": f"1_{i + 1}",
                    "solution": candidate,
                    "solution_hash": candidate_hash,
                    "test_result": test_result,
                    "passed": test_result.get("status") == "pass"
                }

                round_log["candidates"].append(candidate_log)
                candidates.append(candidate_log)

                # Update statistics
                stats["candidates_generated"] += 1
                if test_result.get("status") == "pass":
                    stats["tests_passed"] += 1
                    solution_found = True
                    if best_solution is None:
                        best_solution = candidate_log
                elif test_result.get("status") == "fail":
                    stats["tests_failed"] += 1
                else:
                    stats["test_errors"] += 1

            reflection_logs.append(round_log)
            stats["rounds_completed"] += 1

            # If we already found a solution, we might skip further rounds
            if solution_found and self.config.get("leetcode", {}).get("early_stopping", False):
                logger.info(f"Solution found in round 1, skipping further rounds due to early stopping")
                final_solutions.append(best_solution)
            else:
                # Continue with reflection rounds
                for round_num in range(2, self.num_reflection_rounds + 1):
                    round_start_time = time.time()

                    # Generate improved solutions based on previous rounds
                    improved_candidates = self._generate_improved_candidates(
                        problem_data,
                        candidates,
                        reflection_logs
                    )

                    round_log = {
                        "round": round_num,
                        "phase": f"Reflection Round {round_num - 1}",
                        "candidates": [],
                        "execution_time": time.time() - round_start_time
                    }

                    # Test each improved candidate
                    for i, candidate in enumerate(improved_candidates):
                        candidate_hash = self._calculate_solution_hash(candidate)

                        # Check if we've already tested this solution
                        if candidate_hash in self.solution_cache:
                            test_result = self.solution_cache[candidate_hash]
                            logger.info(f"Using cached test result for solution hash {candidate_hash[:8]}")
                        else:
                            test_result = self._test_solution(problem_data, candidate)
                            self.solution_cache[candidate_hash] = test_result

                        candidate_log = {
                            "candidate_id": f"{round_num}_{i + 1}",
                            "solution": candidate,
                            "solution_hash": candidate_hash,
                            "test_result": test_result,
                            "passed": test_result.get("status") == "pass"
                        }

                        round_log["candidates"].append(candidate_log)
                        candidates.append(candidate_log)

                        # Update statistics
                        stats["candidates_generated"] += 1
                        if test_result.get("status") == "pass":
                            stats["tests_passed"] += 1
                            final_solutions.append(candidate_log)
                            solution_found = True

                            # Update best solution if needed
                            if best_solution is None or (
                                    test_result.get("execution_time", float("inf"))
                                    < best_solution["test_result"].get("execution_time", float("inf"))
                            ):
                                best_solution = candidate_log
                        elif test_result.get("status") == "fail":
                            stats["tests_failed"] += 1
                        else:
                            stats["test_errors"] += 1

                    reflection_logs.append(round_log)
                    stats["rounds_completed"] += 1

                    # Check if we should stop early
                    if solution_found and self.config.get("leetcode", {}).get("early_stopping", False):
                        logger.info(
                            f"Solution found in round {round_num}, skipping further rounds due to early stopping")
                        break

        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}", exc_info=True)
            return {
                "problem_id": problem_data["problem_id"],
                "status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time
            }

        # Perform additional evaluation with code_eval if enabled
        code_eval_results = None
        if self.code_evaluator is not None and solution_found:
            logger.info("Performing additional evaluation with code_eval")

            # Collect all solutions that passed our internal tests
            passing_solutions = [s["solution"] for s in final_solutions]

            # Add the reference solution if available
            if problem_data.get("reference_solution"):
                passing_solutions.append(problem_data["reference_solution"])

            # Run code_eval evaluation
            code_eval_results = self.code_evaluator.evaluate_solutions(
                problem_data,
                passing_solutions
            )

        # Prepare final result
        total_time = time.time() - start_time

        result = {
            "problem_id": problem_data["problem_id"],
            "problem_title": problem_data["problem_title"],
            "difficulty": problem_data.get("difficulty", ""),
            "status": "solved" if solution_found else "unsolved",
            "best_solution": best_solution["solution"] if best_solution else None,
            "passed_solutions": [s["solution"] for s in final_solutions],
            "total_candidates": stats["candidates_generated"],
            "rounds": stats["rounds_completed"],
            "reflection_logs": reflection_logs,
            "stats": stats,
            "processing_time": total_time
        }

        # Add code_eval results if available
        if code_eval_results:
            result["code_eval_results"] = code_eval_results

        # Save result to file
        self._save_results(problem_data["problem_id"], result)

        return result

    def _generate_initial_candidates(self, problem_data: Dict[str, Any]) -> List[str]:
        """
        Generate initial solution candidates using Chain of Thought reasoning.

        Args:
            problem_data: Problem data dictionary.

        Returns:
            List of solution code candidates.
        """
        logger.info(f"Generating {self.num_candidates} initial solution candidates")
        model = self._get_model()
        candidates = []

        for i in range(self.num_candidates):
            self._run_memory_cleanup()

            # Create a prompt with Chain of Thought reasoning
            prompt = self._create_cot_prompt(problem_data, i + 1)

            # Log the prompt for debugging (truncate if too long)
            logger.debug(f"Initial candidate {i + 1} prompt: {prompt}")
            # Generate solution
            response = model.generate(prompt)

            # Log the full response for debugging
            logger.debug(f"Initial candidate {i + 1} full response: {response}")

            # Extract solution code
            solution_code = self._extract_solution_code(response)

            if solution_code:
                candidates.append(solution_code)
                logger.info(f"Generated initial candidate {i + 1}/{self.num_candidates} ({len(solution_code)} chars)")
            else:
                # Log more details about the failure
                logger.warning(f"Failed to extract solution code for candidate {i + 1}")
                logger.warning(f"Response excerpt: {response[:500]}...{response[-500:] if len(response) > 500 else ''}")

        return candidates

    def _generate_improved_candidates(
            self,
            problem_data: Dict[str, Any],
            previous_candidates: List[Dict[str, Any]],
            reflection_logs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate improved candidates through self-reflection.

        Args:
            problem_data: Problem data dictionary.
            previous_candidates: Previous solution candidates with test results.
            reflection_logs: Logs from previous reflection rounds.

        Returns:
            List of improved solution code candidates.
        """
        logger.info(f"Generating {self.num_candidates} improved solution candidates")
        model = self._get_model()
        candidates = []

        # Filter recent candidates (from last round)
        last_round = reflection_logs[-1]["round"]
        recent_candidates = [c for c in previous_candidates if c["candidate_id"].startswith(f"{last_round}_")]

        for i in range(self.num_candidates):
            self._run_memory_cleanup()

            # Create a prompt with self-reflection
            prompt = self._create_reflection_prompt(problem_data, recent_candidates, reflection_logs, i + 1)

            # Log the prompt for debugging (truncate if too long)
            logger.debug(f"Improved candidate {i + 1} prompt: {prompt}")

            # Generate improved solution
            response = model.generate(prompt)

            # Log the full response for debugging
            logger.debug(f"Improved candidate {i + 1} full response: {response}")

            # Extract solution code
            solution_code = self._extract_solution_code(response)

            if solution_code:
                candidates.append(solution_code)
                logger.info(f"Generated improved candidate {i + 1}/{self.num_candidates} ({len(solution_code)} chars)")
            else:
                # Log more details about the failure
                logger.warning(f"Failed to extract solution code for improved candidate {i + 1}")
                logger.warning(f"Response excerpt: {response[:500]}...{response[-500:] if len(response) > 500 else ''}")

        return candidates

    def _create_cot_prompt(self, problem_data: Dict[str, Any], candidate_num: int) -> str:
        """
        Create a Chain of Thought prompt for initial solution generation with strict formatting.
        """
        # Clean up the problem description if needed
        imports_desc = problem_data['prompt']
        problem_desc = problem_data['query']

        prompt = f"""
        You are an expert Python programmer solving a LeetCode problem. Follow my instructions precisely.

        # Problem Statement
        Problem ID: {problem_data['problem_id']}  
        Title: {problem_data['problem_title']}
        Difficulty: {problem_data['difficulty']}
        Tags: {', '.join(problem_data['tags'])}

        # Problem Description
        {problem_desc}

        # Imports
        {imports_desc}

        # Required Function Signature
        Entry Point: {problem_data['entry_point']}

        # Task
        You are generating candidate solution #{candidate_num}. Your solution will be tested against the test cases.

        # REQUIRED RESPONSE FORMAT
        You must follow this exact format:

        1. Start with "## Problem Analysis"
        2. Briefly analyze the problem (2-3 sentences)
        3. Then write "## Solution Approach" 
        4. Briefly describe your approach (2-3 sentences)
        5. Then write EXACTLY "## Code Solution"
        6. Provide your Python solution in a code block that starts with ```python and ends with ```
        7. The *very last* characters in your reply **must** be the closing ``` fence. Anything after that will be 
        ignored and may cause automatic failure.

        EXAMPLE OUTPUT FORMAT:
        ## Problem Analysis
        [Brief analysis]

        ## Solution Approach
        [Brief approach description]

        ## Code Solution
        ```python
        class Solution:
            def method(self, params):
                # Your solution here
                return result
        ```

        IMPORTANT: 
        - **Strict grader**: Replies not in the format above will be discarded without scoring.
        - Wrap code in a fence (```python` … ```), nothing else.  
        - Follow signature *exactly*: `{problem_data['entry_point']}`  
        - Handle edge cases.
        - Avoid indentation problems.
        """

        return prompt

    def _create_reflection_prompt(
            self,
            problem_data: Dict[str, Any],
            previous_candidates: List[Dict[str, Any]],
            reflection_logs: List[Dict[str, Any]],
            candidate_num: int
    ) -> str:
        """
        Create a self-reflection prompt with strict output formatting.
        """
        # Extract the test results from previous candidates
        test_feedback = []

        for candidate in previous_candidates:
            result = candidate["test_result"]
            if result.get("status") == "pass":
                test_feedback.append(f"Candidate {candidate['candidate_id']} PASSED all tests.")
            else:
                error_msg = result.get("error_message", "Unknown error")
                test_feedback.append(f"Candidate {candidate['candidate_id']} FAILED tests: {error_msg}")
                # Include failed test cases if available
                if "failed_tests" in result:
                    for i, test in enumerate(result["failed_tests"]):
                        test_feedback.append(
                            f"  Failed Test {i + 1}: Input={test.get('input')}, Expected={test.get('expected')}, Got={test.get('actual')}")

        # Include code examples from previous candidates
        code_examples = []
        passing_examples = [c for c in previous_candidates if c["test_result"].get("status") == "pass"]
        failing_examples = [c for c in previous_candidates if c["test_result"].get("status") != "pass"]

        # Add up to 1 passing example if available
        if passing_examples:
            example = passing_examples[0]
            code_examples.append(
                f"### Passing Example (Candidate {example['candidate_id']})\n```python\n{example['solution']}\n```")

        # Add up to 2 failing examples if available
        for i, example in enumerate(failing_examples[:2]):
            code_examples.append(
                f"### Failing Example (Candidate {example['candidate_id']})\n```python\n{example['solution']}\n```")

        # Construct the prompt
        problem_desc = problem_data['query']
        prompt = f"""
        You are an expert Python programmer solving a LeetCode problem. Follow my instructions precisely.

        # Problem Statement
        Problem ID: {problem_data['problem_id']}  
        Title: {problem_data['problem_title']}
        Difficulty: {problem_data['difficulty']}
        Tags: {', '.join(problem_data['tags'])}

        # Problem Description
        {problem_desc}

        # Required Function Signature
        Entry Point: {problem_data['entry_point']}

        # Test Feedback 
        {chr(10).join(test_feedback)}

        # Previous Solutions
        {chr(10).join(code_examples)}

        # Task
        You are generating improved candidate solution #{candidate_num}. Fix the issues in previous solutions.

        # REQUIRED RESPONSE FORMAT
        You must follow this exact format:

        1. Start with "## Issue Analysis"
        2. Briefly analyze what went wrong in previous solutions (2-3 sentences)
        3. Then write "## Improved Solution" 
        4. Briefly describe your improved approach (2-3 sentences)
        5. Then write EXACTLY "## Code Solution"
        6. Provide your Python solution in a code block that starts with ```python and ends with ```
        7. Do not include any other text or explanations after the code block

        EXAMPLE OUTPUT FORMAT:
        ## Issue Analysis
        [Brief analysis of issues]

        ## Improved Solution
        [Brief improved approach description]

        ## Code Solution
        ```python
        class Solution:
            def method(self, params):
                # Your improved solution here
                return result
        ```

        IMPORTANT: 
        - Your solution code MUST be wrapped in ```python and ``` tags exactly as shown
        - ONLY include the actual code between these tags
        - Make your code handle all edge cases
        - Follow the required function signature exactly: {problem_data['entry_point']}
        - Avoid indentation problems.
        """

        return prompt

    def _extract_solution_code(self, response: str) -> str:
        """
        Robustly extract the candidate solution code from a model response.
        Strategy:
        1. Slice the response at the first "## Code Solution" header (case-insensitive).
           If the header is missing we fallback to the whole response.
        2. Search that slice for all fenced code blocks whose opening fence starts
           with ``` or ```python / ```py (language tag optional).
           - We take the last fenced block - this is almost always the fresh answer,
             because previous reflection rounds add earlier blocks above.
        3. If no closed fence exists but an opening fence is found, treat everything
           after the opening fence as the code (handles missing closing fence).
        4. Fallback: locate a `class Solution` definition and grab everything until the
           next blank line or EOF.
        5. Final strip and return an empty string on total failure.
        """
        import re
        # Focus on the region after the dedicated header, if present.
        header_regex = re.compile(r'##\s*Code Solution', re.IGNORECASE)
        header_match = header_regex.search(response)
        chunk = response[header_match.end():] if header_match else response

        # Grab the last well-formed fenced code block (language tag optional).
        fence_regex = re.compile(
            r'```(?:python|py)?\s*([\s\S]*?)```',  # non-greedy
            flags=re.IGNORECASE
        )
        fenced_blocks = fence_regex.findall(chunk)
        if fenced_blocks:
            return fenced_blocks[-1].strip()

        # Handle an unterminated fence - take everything after the opening fence.
        open_fence = re.search(r'```(?:python|py)?\s*([\s\S]*)$', chunk, flags=re.IGNORECASE)
        if open_fence:
            return open_fence.group(1).strip()

        # Fallback: extract from a `class Solution` definition onward.
        class_match = re.search(r'class\s+Solution[\s\S]+', chunk)
        if class_match:
            return class_match.group(0).strip()

        # Nothing found.
        logger.warning("Could not extract solution code from response")
        return ""

    def _calculate_solution_hash(self, solution: str) -> str:
        """
        Calculate a hash for the solution to detect duplicates.

        Args:
            solution: Solution code string.

        Returns:
            Solution hash.
        """
        # Normalize the solution to ignore whitespace and comments
        normalized_solution = self._normalize_solution(solution)

        # Calculate hash
        return hashlib.sha256(normalized_solution.encode('utf-8')).hexdigest()

    def _normalize_solution(self, solution: str) -> str:
        """
        Normalize a solution to ignore insignificant differences.

        Args:
            solution: Original solution string.

        Returns:
            Normalized solution string.
        """
        import re

        # Remove comments
        solution = re.sub(r'#.*$', '', solution, flags=re.MULTILINE)

        # Remove empty lines and leading/trailing whitespace
        lines = [line.strip() for line in solution.splitlines() if line.strip()]

        # Join lines with a single space
        return ' '.join(lines)

    def _test_solution(self, problem_data: Dict[str, Any], solution: str) -> Dict[str, Any]:
        """
        Test a solution against the test cases.

        Args:
            problem_data: Problem data dictionary.
            solution: Solution code string.

        Returns:
            Dictionary with test results.
        """
        # Log the complete solution being tested
        logger.debug(f"TESTING SOLUTION:\n{solution}")

        # Log the test code
        logger.debug(f"TEST CODE:\n{problem_data.get('test', 'No test found')}")

        # Log the entry point
        logger.debug(f"ENTRY POINT: {problem_data.get('entry_point', 'No entry point found')}")

        # Fix any indentation issues in the solution code
        solution = self._fix_indentation(solution)

        # Prepare combined test code with proper indentation
        combined_code = self.test_runner.prepare_test_code(problem_data, solution)

        # Run the test and log the result
        result = self.test_runner.execute_test(combined_code)
        logger.debug(f"TEST RESULT: {json.dumps(result, indent=2)}")
        return result

    def _fix_indentation(self, code: str) -> str:
        """
        Fix indentation issues in Python code.
        - Ensures module-level code has zero indentation
        - Properly handles class and function definitions
        - Correctly handles import statements

        Args:
            code: The Python code string to fix

        Returns:
            The code with fixed indentation
        """
        if not code or code.isspace():
            return code

        # Split the code into lines
        lines = code.split('\n')

        # First, detect minimum indentation level for imports and class definitions
        min_indent = float('inf')
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):  # Skip empty lines and comments
                indent = len(line) - len(line.lstrip())
                if line.lstrip().startswith(('import ', 'from ', 'class ', 'def ')):
                    min_indent = min(min_indent, indent)

        # If no valid lines found, default to 0
        if min_indent == float('inf'):
            min_indent = 0

        # Process each line to ensure proper indentation
        fixed_lines = []
        current_block_indent = 0
        in_class_or_function = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue

            # Get current indentation
            current_indent = len(line) - len(line.lstrip())

            # Check for class/function definitions or module-level imports
            if stripped.startswith(('class ', 'def ')):
                # Start of new class or function
                if current_indent == min_indent:
                    # This is a top-level class or function
                    in_class_or_function = True
                    current_block_indent = current_indent
                    fixed_lines.append(stripped)  # No indentation for class/function definitions
                else:
                    # This is a nested class or function, preserve relative indentation
                    relative_indent = current_indent - min_indent
                    fixed_lines.append(' ' * relative_indent + stripped)
                continue

            # Check if line is an import or module-level statement
            if current_indent == min_indent and stripped.startswith(('import ', 'from ')):
                fixed_lines.append(stripped)  # No indentation for imports
                continue

            # For other lines, adjust indentation relative to the min_indent
            if current_indent >= min_indent:
                relative_indent = current_indent - min_indent
                fixed_lines.append(' ' * relative_indent + stripped)
            else:
                # If indentation is less than min_indent, treat as module level
                fixed_lines.append(stripped)

        # Reassemble the code
        fixed_code = '\n'.join(fixed_lines)
        return fixed_code

    def _save_results(self, problem_id: str, result: Dict[str, Any]) -> None:
        """
        Save the results to a file.

        Args:
            problem_id: Problem ID.
            result: Result dictionary.
        """
        # Create result directory if it doesn't exist
        result_dir = self.results_dir / "leetcode_solutions"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        result_file = result_dir / f"{problem_id}.json"

        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved results to {result_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
