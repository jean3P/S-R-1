import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.agents.tree_of_thought_diagnostic_agent import TreeOfThoughtDiagnosticAgent
from src.agents.base_agent import BaseAgent
from src.utils.file_utils import save_json, get_result_output_path
from src.utils.parsing import extract_code_blocks, extract_patches, parse_execution_result
from src.utils.metrics import calculate_code_metrics, calculate_patch_metrics, compare_solutions
from src.utils.tokenization import count_tokens

from src.context.code_summarizer import CodeSummarizer
from src.context.context_manager import ContextManager
from src.context.progressive_disclosure import ProgressiveDisclosure
from src.context.memory_manager import MemoryManager


class ImprovedCodeRefinementAgent(BaseAgent):
    """
    Improved agent that generates and refines code through self-reflection
    with intelligent context management and progressive disclosure.
    Can be used for both general code refinement and patch generation/refinement.
    """

    def __init__(self, model_id: str, prompt_id: str, evaluator_id: str, config: Dict[str, Any]):
        """
        Initialize the improved code refinement agent.

        Args:
            model_id: ID of the model to use
            prompt_id: ID of the prompt to use
            evaluator_id: ID of the evaluator to use
            config: Configuration dictionary
        """
        super().__init__(model_id, prompt_id, evaluator_id, config)

        # Initialize context management component

        self.code_summarizer = CodeSummarizer(config.get("summarizer_config"))
        self.context_manager = ContextManager(
            max_tokens=config.get("max_tokens", 4000),
            config=config.get("context_manager_config")
        )
        self.progressive_disclosure = ProgressiveDisclosure(config.get("disclosure_config"))
        self.memory_manager = MemoryManager(
            max_history_items=config.get("max_history_items", 10),
            config=config.get("memory_manager_config")
        )

        # Determine if we're in patch mode
        self.patch_mode = config.get("patch_mode", False)
        self.task = None  # Will store the task for later use
        self.logger.info(f"Agent initialized in {'patch' if self.patch_mode else 'code'} mode")

        self.use_tot_agent = config.get("use_tree_of_thought", False)
        self.tot_agent = None
        if self.use_tot_agent:
            self.logger.info("Creating an internal TreeOfThoughtDiagnosticAgent for mid-iteration analysis.")
            tot_config = config.get("tot_config", {})  # or re-use the same config, up to you
            self.tot_agent = TreeOfThoughtDiagnosticAgent(
                model_id=model_id,
                prompt_id=prompt_id,
                evaluator_id=evaluator_id,
                config=tot_config
            )

        self.logger.info(f"Agent initialized in {'patch' if self.patch_mode else 'code'} mode")

    def reflect(self, initial_prompt: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the code generation and refinement loop with intelligent context management.

        Args:
            initial_prompt: Initial coding task prompt
            task: Task details including constraints

        Returns:
            Dict containing all iterations of the refinement process
        """
        self.logger.info(f"Starting improved code refinement for task: {task.get('name', 'unnamed')}")
        self.task = task  # Store task for later use

        # Initialize metrics
        self._start_metrics()

        # Store iteration results
        iterations = []
        current_prompt = initial_prompt
        best_solution = None
        best_solution_metrics = None

        # Prepare file resources - analyze all relevant files
        file_paths = task.get("files", [])
        file_summaries = {}

        # Pre-process code files to create summaries instead of using full content
        for file_path in file_paths:
            self.logger.info(f"Analyzing file: {file_path}")
            file_summary = self.code_summarizer.summarize_file(file_path)
            file_summaries[file_path] = file_summary

        # Generate file index
        file_index = self.code_summarizer.generate_file_index(file_paths)

        # Add file information to task
        task["file_summaries"] = file_summaries
        task["file_index"] = file_index

        # Check if we're in repo-based task
        repo_info = task.get("repo_info", {})
        if repo_info:
            self.logger.info(f"Repository-based task detected: {repo_info.get('repo', 'unknown')}")
            # Force patch mode for repository tasks
            self.patch_mode = True

        # Prepare initial context
        initial_context = self.progressive_disclosure.create_initial_context(task)

        # Main iteration loop
        for i in range(1, self.max_iterations + 1):
            self.logger.info(f"Iteration {i}/{self.max_iterations}")

            # Get continuity context from memory manager
            continuity_context = self.memory_manager.build_continuity_context()

            # Get relevant context for this iteration using key terms from the current prompt
            relevant_context = self.context_manager.get_relevant_context(
                prompt=current_prompt,
                task=task,
                file_summaries=file_summaries,
                iteration=i
            )

            # Combine contexts
            combined_context = {
                "continuity": continuity_context,
                "relevant_code": relevant_context,
                "initial": initial_context if i == 1 else None,
                "iteration": i
            }

            # Step 1: Generate code based on current context
            generation_input = self.prompt.format_generation(
                prompt=current_prompt,
                context=combined_context,
                task=task
            )

            # Calculate token usage
            tokens_used = count_tokens(generation_input, self.model.config.get("model_name", "default"))
            self.logger.info(f"Generation prompt token usage: {tokens_used}")

            # Generate solution
            solution, generation_time = self._measure_execution(self.model.generate, generation_input)

            # Record metrics
            self._record_generation(tokens_used, generation_time, True)

            # Extract code or patch from solution based on mode
            if self.patch_mode:
                code_blocks = extract_patches(solution)
                self.logger.info(f"Extracted {len(code_blocks)} patches from solution")
            else:
                code_blocks = extract_code_blocks(solution)
                self.logger.info(f"Extracted {len(code_blocks)} code blocks from solution")

            code = code_blocks[0] if code_blocks else solution

            # STEP 2: (New) TOT-based analysis of the generated solution
            if self.use_tot_agent and self.tot_agent:
                self.logger.info("Using TreeOfThoughtDiagnosticAgent to analyze the generated solution.")

                # TOT prompt can reference the code or partial logs
                tot_prompt = (
                    "We have this partial solution:\n"
                    f"{code}\n\n"
                    "Please diagnose potential shape mismatches or unclear logic."
                )
                # TOT's task can pass anything TOT might need
                tot_task = {"solution_code": code, "additional_info": "No logs yet, for example."}

                tot_result = self.tot_agent.reflect(tot_prompt, tot_task)
                diag_text = tot_result.get("analysis", "")
                diag_fix = tot_result.get("fix", "")
                self.logger.info(f"TOT Diagnostic says: {diag_text} (Recommended fix: {diag_fix})")

                # Incorporate TOT analysis into current prompt
                current_prompt += (
                    f"\n\n# [TOT Analysis]\n{diag_text}\n"
                    f"# TOT recommended fix: {diag_fix}\n"
                )

            # Analyze solution to find areas that need more detail
            focus_areas = self._extract_focus_areas(solution)

            # Check if model is requesting more information
            if focus_areas:
                self.logger.info(f"Model identified focus areas: {focus_areas}")

                # Expand context with additional details in focus areas
                additional_context = self.progressive_disclosure.expand_context(focus_areas)
                # Then use it to update the combined_context:
                if additional_context:
                    combined_context.update({"expanded_details": additional_context})

                # Extract and handle explicit queries
                explicit_query = self._extract_explicit_query(solution)
                if explicit_query:
                    self.logger.info(f"Processing explicit query: {explicit_query}")

                    # Get additional context for the query
                    response, new_context = self.progressive_disclosure.respond_to_query(
                        query=explicit_query,
                        context=combined_context,
                        file_summaries=file_summaries
                    )

                    # Add the response to the prompt for the next generation
                    current_prompt += f"\n\nAdditional context: {response}"

            # Set task on evaluator
            self.evaluator.config["task"] = task

            # Evaluate the generated code or patch
            output, errors = self.evaluator.evaluate(code, task=task)
            self.logger.info(f"Evaluation complete: Success={not errors}")

            execution_result = parse_execution_result(output, errors)

            # Calculate metrics
            if self.patch_mode:
                code_metrics = calculate_patch_metrics(code)
                # Update test pass percentage if available in execution result
                if "test_results" in execution_result:
                    test_results = execution_result["test_results"]
                    if "total" in test_results and test_results["total"] > 0:
                        code_metrics["test_pass_percentage"] = (test_results["passed"] / test_results["total"]) * 100
            else:
                code_metrics = calculate_code_metrics(code)

            # Compare with previous solutions if not the first iteration
            comparison = None
            if i > 1 and iterations:
                prev_solution = iterations[-1].get("code", "")
                comparison = compare_solutions(prev_solution, code)

            # Check if this is the best solution so far
            is_best = self._determine_best_solution(
                code,
                execution_result,
                code_metrics,
                best_solution,
                best_solution_metrics
            )

            if is_best:
                best_solution = code
                best_solution_metrics = execution_result
                self.logger.info(f"New best solution found in iteration {i}")

            # Generate improved solution with reflection
            reflection_input = self.prompt.format_reflection(
                original_prompt=current_prompt,
                solution=solution,
                output=output,
                errors=errors,
                context=combined_context,
                task=task
            )

            # Calculate token usage for reflection
            reflection_tokens = count_tokens(reflection_input, self.model.config.get("model_name", "default"))
            self.logger.info(f"Reflection prompt token usage: {reflection_tokens}")

            # Generate refined solution
            refined_solution, reflection_time = self._measure_execution(self.model.generate, reflection_input)

            # Record metrics
            self._record_generation(reflection_tokens, reflection_time, True)

            # Extract refined code or patch
            if self.patch_mode:
                refined_code_blocks = extract_patches(refined_solution)
            else:
                refined_code_blocks = extract_code_blocks(refined_solution)

            refined_code = refined_code_blocks[0] if refined_code_blocks else refined_solution

            # Store this interaction for future reference
            self.memory_manager.store_interaction(
                prompt=current_prompt,
                response=solution,
                context=combined_context,
                result=execution_result
            )

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
                "refined_code": refined_code,
                "focus_areas": focus_areas,
                "token_usage": {
                    "generation": tokens_used,
                    "reflection": reflection_tokens
                }
            }
            iterations.append(iteration_result)

            # Update prompt for next iteration
            current_prompt = refined_solution

            # Early stopping if we have a successful solution and early stopping is enabled
            if execution_result.get("success", False) and self.config.get("early_stop_on_success", False):
                self.logger.info("Early stopping due to successful solution")
                break

        # Finalize metrics
        self._end_metrics()

        # Find the best solution across all iterations
        best_iteration = next((it for it in iterations if it.get("is_best")), None)

        # Prepare results
        results = {
            "task": task,
            "initial_prompt": initial_prompt,
            "model_id": self.model.config.get("model_id", "unknown"),
            "agent_id": self.__class__.__name__,
            "iterations": iterations,
            "metrics": self.metrics,
            "best_solution": best_solution,
            "best_iteration": best_iteration["iteration"] if best_iteration else None,
            "success": best_solution_metrics.get("success", False) if best_solution_metrics else False
        }

        # Save results if configured
        if self.config.get("save_results", True):
            output_path = get_result_output_path(
                task=task,
                model_id=self.model.config.get("model_id", "unknown"),
                agent_id=self.__class__.__name__,
                base_dir=self.config.get("output_dir", "results")
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_json(results, output_path)
            self.logger.info(f"Results saved to {output_path}")

        self.logger.info(f"Completed improved code refinement with {len(iterations)} iterations")
        return results

    def _extract_focus_areas(self, solution: str) -> List[str]:
        """
        Extract areas where the model needs more detailed information.

        Args:
            solution: Generated solution

        Returns:
            List of focus areas
        """
        # Look for explicit requests for more information
        focus_areas = []

        # Look for comments indicating uncertainty
        uncertainty_patterns = [
            r'# Need more information about (.*)',
            r'# TODO: Check (.*)',
            r'# FIXME: Verify (.*)',
            r'# Unclear: (.*)',
            r'# Not sure about (.*)',
            r'# Need details on (.*)'
        ]

        for pattern in uncertainty_patterns:
            matches = re.findall(pattern, solution)
            focus_areas.extend(matches)

        # Look for explicit mentions in text areas
        text_patterns = [
            r'I need more information about (.*?)\.',
            r'I would need to see (.*?) to implement this properly',
            r'Cannot implement (.*?) without more details',
            r'Unclear how (.*?) works'
        ]

        for pattern in text_patterns:
            matches = re.findall(pattern, solution)
            focus_areas.extend(matches)

        # Look for patch-specific uncertainties if in patch mode
        if self.patch_mode:
            patch_patterns = [
                r'I need to see the file (.*?) to create a proper patch',
                r'I\'m not sure about the correct file path for (.*?)',
                r'Need to verify line numbers in (.*?)',
                r'The patch might not apply correctly to (.*?)'
            ]

            for pattern in patch_patterns:
                matches = re.findall(pattern, solution)
                focus_areas.extend(matches)

        return focus_areas

    def _extract_explicit_query(self, solution: str) -> Optional[str]:
        """
        Extract explicit query from the solution.

        Args:
            solution: Generated solution

        Returns:
            Explicit query or None
        """
        query_patterns = [
            r'# Query: (.*)',
            r'# Question: (.*)',
            r'# Please provide: (.*)',
            r'# Need details on: (.*)',
            r'QUERY: (.*)',
            r'I need to know: (.*)'
        ]

        # Add patch-specific query patterns if in patch mode
        if self.patch_mode:
            query_patterns.extend([
                r'# File query: (.*)',
                r'# Repository query: (.*)',
                r'# Path query: (.*)',
                r'Can you provide the content of (.*?)\?'
            ])

        for pattern in query_patterns:
            matches = re.findall(pattern, solution)
            if matches:
                return matches[0].strip()

        return None

    def _determine_best_solution(self,
                                 current_solution: str,
                                 current_result: Dict[str, Any],
                                 current_metrics: Dict[str, Any],
                                 best_solution: Optional[str],
                                 best_result: Optional[Dict[str, Any]]) -> bool:
        """
        Determine if the current solution is better than the best solution so far.

        Args:
            current_solution: Current solution code
            current_result: Current execution result
            current_metrics: Current code metrics
            best_solution: Best solution so far
            best_result: Best execution result so far

        Returns:
            True if current solution is better, False otherwise
        """
        # If no best solution yet, current is best
        if best_solution is None:
            return True

        # If current succeeds and best fails, current is better
        if current_result.get("success", False) and not best_result.get("success", False):
            return True

        # If both succeed, compare other metrics
        if current_result.get("success", False) and best_result.get("success", False):
            # Different scoring systems for patch mode vs code mode
            if self.patch_mode:
                current_score = (
                    # Fewer errors is better
                        -1 * current_result.get("error_count", 0) * 10 +
                        # Smaller patches are generally better
                        -0.5 * current_metrics.get("lines_changed", 0) +
                        # Fewer files touched is better
                        -2 * current_metrics.get("files_touched", 1) +
                        # Test pass percentage
                        5 * current_metrics.get("test_pass_percentage", 0)
                )

                if best_solution == current_solution:
                    # If identical, keep the existing one
                    return False

                best_metrics = calculate_patch_metrics(best_solution)
                best_score = (
                        -1 * best_result.get("error_count", 0) * 10 +
                        -0.5 * best_metrics.get("lines_changed", 0) +
                        -2 * best_metrics.get("files_touched", 1) +
                        5 * best_metrics.get("test_pass_percentage", 0)
                )
            else:
                # Original scoring for code mode
                current_score = (
                    # Fewer errors is better
                        -1 * current_result.get("error_count", 0) * 10 +
                        # Lower complexity is better
                        -1 * current_metrics.get("complexity", 0) +
                        # More test coverage is better (if available)
                        current_metrics.get("test_coverage", 0) * 2 +
                        # Lower token count is better (more efficient)
                        -0.01 * len(current_solution.split())
                )

                best_metrics = calculate_code_metrics(best_solution)
                best_score = (
                        -1 * best_result.get("error_count", 0) * 10 +
                        -1 * best_metrics.get("complexity", 0) +
                        best_metrics.get("test_coverage", 0) * 2 +
                        -0.01 * len(best_solution.split())
                )

            return current_score > best_score

        # Stick with existing best solution
        return False

    def generate_solution(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Generate a solution based on the prompt and context.

        Args:
            prompt: Current prompt
            context: Current context

        Returns:
            Generated solution
        """
        # Format the generation prompt
        generation_input = self.prompt.format_generation(
            prompt=prompt,
            context=context,
            task=self.task
        )

        # Generate solution
        solution, _ = self._measure_execution(self.model.generate, generation_input)

        return solution

    def evaluate_solution(self, solution: str) -> Dict[str, Any]:
        """
        Evaluate a solution.

        Args:
            solution: Solution to evaluate

        Returns:
            Evaluation result
        """
        # Extract code or patch from solution based on mode
        if self.patch_mode:
            code_blocks = extract_patches(solution)
        else:
            code_blocks = extract_code_blocks(solution)

        code = code_blocks[0] if code_blocks else solution

        # Evaluate the code
        output, errors = self.evaluator.evaluate(code, task=self.task)
        result = parse_execution_result(output, errors)

        return result
