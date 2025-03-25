# src/agents/patch_refinement_agent.py

import os
from typing import Dict, Any, List, Optional, Tuple
import time
from src.agents.base_agent import BaseAgent
from src.utils.file_utils import save_json
from src.utils.tokenization import count_tokens
from src.utils.metrics import compare_solutions, calculate_code_metrics
from src.utils.parsing import parse_execution_result
from src.utils.logging import get_logger


class PatchRefinementAgent(BaseAgent):
    """
    Agent that refines Git patches (diffs) through self-reflection,
    skipping all Python code parsing or execution.
    """

    def __init__(self, model_id: str, prompt_id: str, evaluator_id: str, config: Dict[str, Any]):
        super().__init__(model_id, prompt_id, evaluator_id, config)
        self.logger = get_logger(self.__class__.__name__)

    def reflect(self, initial_prompt: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main reflection loop to produce & refine Git patches.

        Args:
            initial_prompt: The initial prompt or description of the issue.
            task: Dictionary containing repository info, test info, etc.

        Returns:
            Dictionary with the final patch solution and iteration details.
        """
        self.logger.info(f"Starting patch refinement for task: {task.get('name', 'unnamed')}")

        # 1) Initialize metrics & iteration records
        self._start_metrics()
        iterations = []
        current_prompt = initial_prompt
        best_solution = None
        best_solution_metrics = None

        # 2) Main iteration loop
        for i in range(1, self.max_iterations + 1):
            self.logger.info(f"Iteration {i}/{self.max_iterations}")

            # -- (A) Generate patch from the model
            generation_input = self.prompt.format_generation(current_prompt, task)

            tokens_used = count_tokens(generation_input, self.model.config.get("model_name", "default"))
            # measure execution time of .generate()
            gen_solution, gen_time = self._measure_execution(self.model.generate, generation_input)
            self._record_generation(tokens_used, gen_time, success=True)

            # We'll call the patch text 'code' for consistency, but it's actually a Git diff
            patch_text = gen_solution  # no Python extraction

            # -- (B) Evaluate patch with the SWE-bench evaluator
            # Our evaluator might apply the patch & run tests
            self.evaluator.config["task"] = task
            output, errors = self.evaluator.evaluate(patch_text)
            exec_result = parse_execution_result(output, errors)

            # Even though this is not Python, we'll still compute some metrics
            # (like line counts or diff complexity) if you want
            # If you’d rather skip, set code_metrics = {}
            code_metrics = calculate_code_metrics(patch_text)

            # Compare with previous iteration
            comparison = None
            if i > 1 and iterations:
                prev_patch = iterations[-1].get("patch_text", "")
                comparison = compare_solutions(prev_patch, patch_text)

            # Check if best so far
            is_best = False
            if best_solution is None:
                best_solution = patch_text
                best_solution_metrics = exec_result
                is_best = True
            else:
                # If new patch passes and old one didn’t, or any custom logic
                if exec_result["success"] and not best_solution_metrics["success"]:
                    best_solution = patch_text
                    best_solution_metrics = exec_result
                    is_best = True

            # (C) Prepare reflection prompt for next iteration
            reflection_input = self.prompt.format_reflection(
                original_prompt=current_prompt,
                solution=gen_solution,
                output=output,
                errors=errors,
                task=task
            )

            tokens_used = count_tokens(reflection_input, self.model.config.get("model_name", "default"))
            refined_solution, reflection_time = self._measure_execution(self.model.generate, reflection_input)
            self._record_generation(tokens_used, reflection_time, success=True)

            # For the next iteration, the prompt is the refined patch
            current_prompt = refined_solution

            # Gather iteration info
            iteration_result = {
                "iteration": i,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "prompt": generation_input,
                "solution": gen_solution,
                "patch_text": patch_text,
                "execution_output": output,
                "execution_errors": errors,
                "execution_result": exec_result,
                "code_metrics": code_metrics,
                "comparison": comparison,
                "is_best": is_best,
                "reflection_prompt": reflection_input,
                "refined_solution": refined_solution
            }
            iterations.append(iteration_result)

            # Possibly do early stop
            if exec_result["success"] and self.config.get("early_stop_on_success", False):
                self.logger.info("Early stopping due to successful solution")
                break

        # 3) Finalize
        self._end_metrics()

        # Find best iteration
        best_iteration = None
        for iteration in iterations:
            if iteration.get("is_best", False):
                best_iteration = iteration

        # Prepare final results
        results = {
            "task": task,
            "initial_prompt": initial_prompt,
            "iterations": iterations,
            "metrics": self.metrics,
            "best_solution": best_solution,
            "best_iteration": best_iteration["iteration"] if best_iteration else None,
            "success": best_solution_metrics["success"] if best_solution_metrics else False
        }

        # Save results if configured
        if self.config.get("save_results", True):
            output_dir = self.config.get("output_dir", "data/results")
            os.makedirs(output_dir, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"{task.get('name', 'task')}_{timestamp}.json")
            save_json(results, output_file)
            self.logger.info(f"Results saved to {output_file}")

        self.logger.info(f"Completed patch refinement with {len(iterations)} iterations")
        return results
