# src/agents/code_refinement_agent.py
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.utils.file_utils import save_json
from src.utils.parsing import extract_code_blocks, extract_python_function, parse_execution_result
from src.utils.metrics import calculate_code_metrics, compare_solutions
from src.utils.tokenization import count_tokens


class CodeRefinementAgent(BaseAgent):
    """Agent that generates and refines code through self-reflection."""

    def reflect(self, initial_prompt: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the code generation and refinement loop.

        Args:
            initial_prompt: Initial coding task prompt
            task: Task details including constraints

        Returns:
            Dict containing all iterations of the refinement process
        """
        self.logger.info(f"Starting code refinement for task: {task.get('name', 'unnamed')}")

        # Initialize metrics
        self._start_metrics()

        # Store iteration results
        iterations = []
        current_prompt = initial_prompt
        best_solution = None
        best_solution_metrics = None

        # Main iteration loop
        for i in range(1, self.max_iterations + 1):
            self.logger.info(f"Iteration {i}/{self.max_iterations}")

            # Step 1: Generate code based on current prompt
            generation_input = self.prompt.format_generation(current_prompt, task)

            tokens_used = count_tokens(generation_input, self.model.config.get("model_name", "default"))
            solution, generation_time = self._measure_execution(self.model.generate, generation_input)

            # Record metrics
            self._record_generation(tokens_used, generation_time, True)

            # Step 2: Extract code from solution
            code_blocks = extract_code_blocks(solution)
            code = code_blocks[0] if code_blocks else solution

            # Step 3: Evaluate the generated code
            output, errors = self.evaluator.evaluate(code)
            execution_result = parse_execution_result(output, errors)

            # Step 4: Calculate code metrics
            code_metrics = calculate_code_metrics(code)

            # Step 5: Compare with previous solutions
            comparison = None
            if i > 1 and iterations:
                prev_solution = iterations[-1].get("code", "")
                comparison = compare_solutions(prev_solution, code)

            # Step 6: Check if this is the best solution so far
            is_best = False
            if best_solution is None or (
                    execution_result["success"] and not best_solution_metrics.get("success", False)):
                best_solution = code
                best_solution_metrics = execution_result
                is_best = True
            elif execution_result["success"] and best_solution_metrics.get("success", False):
                # Both successful, compare based on metrics (e.g., code complexity)
                if code_metrics.get("complexity", float("inf")) < best_solution_metrics.get("complexity", float("inf")):
                    best_solution = code
                    best_solution_metrics = execution_result
                    is_best = True

            # Step 7: Generate improved solution with reflection
            reflection_input = self.prompt.format_reflection(
                original_prompt=current_prompt,
                solution=solution,
                output=output,
                errors=errors,
                task=task
            )

            tokens_used = count_tokens(reflection_input, self.model.config.get("model_name", "default"))
            refined_solution, reflection_time = self._measure_execution(self.model.generate, reflection_input)

            # Record metrics
            self._record_generation(tokens_used, reflection_time, True)

            # Step 8: Extract refined code
            refined_code_blocks = extract_code_blocks(refined_solution)
            refined_code = refined_code_blocks[0] if refined_code_blocks else refined_solution

            # Log iteration results
            iteration_result = {
                "iteration": i,
                "timestamp": datetime.now().isoformat(),
                "prompt": current_prompt,
                "solution": solution,
                "code": code,
                "execution_output": output,
                "execution_errors": errors,
                "execution_result": execution_result,
                "code_metrics": code_metrics,
                "comparison": comparison,
                "is_best": is_best,
                "reflection_prompt": reflection_input,
                "refined_solution": refined_solution,
                "refined_code": refined_code
            }
            iterations.append(iteration_result)

            # Update prompt for next iteration
            current_prompt = refined_solution

            # Early stopping if we have a successful solution and early stopping is enabled
            if execution_result["success"] and self.config.get("early_stop_on_success", False):
                self.logger.info("Early stopping due to successful solution")
                break

        # Finalize metrics
        self._end_metrics()

        # Find the best solution across all iterations
        best_iteration = None
        for iteration in iterations:
            if iteration.get("is_best", False):
                best_iteration = iteration

        # Extract the specific function if requested
        extracted_function = None
        if best_solution and task.get("extract_function"):
            function_name = task.get("extract_function")
            extracted_function = extract_python_function(best_solution, function_name)

        # Prepare results
        results = {
            "task": task,
            "initial_prompt": initial_prompt,
            "iterations": iterations,
            "metrics": self.metrics,
            "best_solution": best_solution,
            "best_iteration": best_iteration["iteration"] if best_iteration else None,
            "extracted_function": extracted_function,
            "success": best_solution_metrics["success"] if best_solution_metrics else False
        }

        # Save results if configured
        if self.config.get("save_results", True):
            output_dir = self.config.get("output_dir", "data/results")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"{task.get('name', 'task')}_{timestamp}.json")
            save_json(results, output_file)
            self.logger.info(f"Results saved to {output_file}")

        self.logger.info(f"Completed code refinement with {len(iterations)} iterations")
        return results
    