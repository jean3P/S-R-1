# src/agents/tree_of_thought_patch_agent.py

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.utils.parsing import extract_patches


class TreeOfThoughtPatchAgent(BaseAgent):
    """
    Agent that applies Tree of Thought reasoning to generate and refine GitHub patches.
    Explores multiple reasoning paths concurrently and selects the most promising solution.
    """

    def __init__(self, model_id: str, prompt_id: str, evaluator_id: str, config: Dict[str, Any]):
        super().__init__(model_id, prompt_id, evaluator_id, config)

        # ToT-specific parameters
        self.max_branches = config.get("max_branches", 3)  # Maximum branches to explore at each step
        self.max_depth = config.get("max_depth", 3)  # Maximum reasoning depth
        self.selection_strategy = config.get("selection_strategy", "best_first")  # Strategy for selecting branches
        self.temperature_schedule = config.get("temperature_schedule", [0.7, 0.5, 0.3])  # Decreasing temperature

    def reflect(self, initial_prompt: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Tree of Thought reasoning process for patch generation.

        Args:
            initial_prompt: Initial problem statement
            task: Task details including repository information

        Returns:
            Dictionary containing the reasoning process and results
        """
        self._start_metrics()
        self.task = task

        # Initialize the reasoning tree with the root node
        reasoning_tree = {
            "root": {
                "id": "root",
                "prompt": initial_prompt,
                "children": [],
                "depth": 0,
                "solution": None,
                "evaluation": None,
                "status": "pending"
            }
        }

        # Track active branches for exploration
        active_branches = ["root"]

        # Store the best solution found
        best_solution = {
            "patch": None,
            "score": float('-inf'),
            "node_id": None
        }

        # Execute ToT reasoning process
        for depth in range(1, self.max_depth + 1):
            self.logger.info(f"Exploring reasoning depth {depth}")

            # Set temperature for this depth
            temperature = self.temperature_schedule[min(depth - 1, len(self.temperature_schedule) - 1)]

            # Generate and evaluate new branches
            next_active_branches = []

            for parent_id in active_branches:
                parent_node = reasoning_tree[parent_id]

                # Generate branches (different reasoning paths)
                branches = self._generate_branches(parent_node, depth, temperature, self.max_branches)

                for branch_id, branch in branches.items():
                    reasoning_tree[branch_id] = branch
                    parent_node["children"].append(branch_id)

                    # Evaluate the solution if one was generated
                    if branch.get("solution"):
                        evaluation = self._evaluate_solution(branch["solution"])
                        branch["evaluation"] = evaluation

                        # Update best solution if better
                        score = self._calculate_solution_score(evaluation)
                        if score > best_solution["score"]:
                            best_solution["patch"] = branch["solution"]
                            best_solution["score"] = score
                            best_solution["node_id"] = branch_id

                    # Add promising branches to next iteration
                    if branch["status"] == "active":
                        next_active_branches.append(branch_id)

            # Update active branches for next iteration
            active_branches = self._select_branches(next_active_branches, reasoning_tree)

            # Early stopping if we've found a good solution
            if best_solution["score"] >= self.config.get("early_stop_threshold", 0.8):
                self.logger.info(f"Early stopping at depth {depth}: found good solution")
                break

            # Also stop if we have no active branches
            if not active_branches:
                self.logger.info(f"Stopping at depth {depth}: no active branches")
                break

        # Finalize metrics
        self._end_metrics()

        return {
            "task": task,
            "reasoning_tree": reasoning_tree,
            "best_solution": best_solution["patch"],
            "best_node_id": best_solution["node_id"],
            "metrics": self.metrics,
            "success": best_solution["score"] > 0
        }

    def _generate_branches(self, parent_node: Dict[str, Any], depth: int,
                           temperature: float, max_branches: int) -> Dict[str, Dict[str, Any]]:
        """Generate new reasoning branches from a parent node."""
        branches = {}

        # Create distinct reasoning prompts for different branches
        for i in range(max_branches):
            try:
                # Generate prompt that includes specific reasoning strategy or focus
                branch_prompt = self._create_branch_prompt(parent_node, depth, i)

                # Generate a solution for this branch
                self.logger.info(f"Generating branch {i+1}/{max_branches} at depth {depth} with temperature {temperature}")
                solution, generation_time = self._measure_execution(
                    self.model.generate,
                    branch_prompt
                )
                self.logger.info(f"Generation completed in {generation_time:.2f}s")

                # Extract patch if present
                patches = extract_patches(solution)
                patch = patches[0] if patches else None
                
                if patch:
                    self.logger.info(f"Extracted patch of length {len(patch)}")
                else:
                    self.logger.info("No patch extracted from response")

                # Create branch node
                branch_id = f"{parent_node['id']}-{depth}-{i}"
                branches[branch_id] = {
                    "id": branch_id,
                    "parent": parent_node["id"],
                    "depth": depth,
                    "branch_index": i,
                    "prompt": branch_prompt,
                    "reasoning": solution,
                    "solution": patch,
                    "evaluation": None,
                    "children": [],
                    "status": "active" if patch else "terminated"
                }
            except Exception as e:
                self.logger.error(f"Error generating branch {i} at depth {depth}: {str(e)}")
                # Create a fallback branch node
                branch_id = f"{parent_node['id']}-{depth}-{i}"
                branches[branch_id] = {
                    "id": branch_id,
                    "parent": parent_node["id"],
                    "depth": depth,
                    "branch_index": i,
                    "prompt": "Error generating prompt",
                    "reasoning": f"Error occurred: {str(e)}",
                    "solution": None,
                    "evaluation": None,
                    "children": [],
                    "status": "terminated"
                }

        return branches

    def _create_branch_prompt(self, parent_node: Dict[str, Any], depth: int, branch_index: int) -> str:
        """Create a prompt for a specific reasoning branch."""
        # Base prompt from parent
        base_prompt = parent_node["prompt"]

        # Different reasoning strategies based on branch_index
        strategies = [
            "Focus on identifying the minimal required changes",
            "Focus on ensuring tests will pass",
            "Focus on maintaining code style consistency"
        ]

        # Add specific guidance from the parent reasoning if available
        parent_reasoning = parent_node.get("reasoning", "")

        # Select a strategy based on branch index (cycle through if more branches than strategies)
        strategy = strategies[branch_index % len(strategies)]

        # Create comprehensive prompt template
        prompt_template = self.prompt.format_tot_reasoning(
            original_prompt=base_prompt,
            parent_reasoning=parent_reasoning,
            depth=depth,
            strategy=strategy,
            task=self.task
        )

        return prompt_template

    def _evaluate_solution(self, patch: str) -> Dict[str, Any]:
        """Evaluate a solution patch."""
        if not patch:
            return {"success": False, "error": "No patch generated"}

        # Use the evaluator to test the patch
        try:
            self.logger.info("Evaluating patch with evaluator")
            
            # Ensure task has required fields for SWE-bench evaluation
            if self.task and not self.task.get("repo_info"):
                self.logger.warning("Task missing repo_info, adding placeholder")
                self.task["repo_info"] = {
                    "repo": self.task.get("repo", "unknown"),
                    "base_commit": self.task.get("base_commit", "unknown")
                }
            
            # Wrap in try-except to catch any evaluator errors
            try:
                output, errors = self.evaluator.evaluate(patch, task=self.task)
            except Exception as eval_error:
                self.logger.error(f"Evaluator error: {str(eval_error)}")
                return {
                    "success": False,
                    "output": "",
                    "errors": f"Evaluation error: {str(eval_error)}"
                }
            
            # Log evaluation results
            if errors:
                self.logger.info(f"Evaluation failed with errors: {errors[:100]}...")
            else:
                self.logger.info("Evaluation succeeded")

            # Parse evaluation results
            return {
                "success": not errors,
                "output": output,
                "errors": errors
            }
        except Exception as e:
            self.logger.error(f"Error evaluating solution: {str(e)}")
            return {"success": False, "error": str(e)}

    def _calculate_solution_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate a score for a solution based on its evaluation."""
        if not evaluation:
            return float('-inf')

        # Start with base score
        score = 0.0

        # Success is the most important factor
        if evaluation.get("success", False):
            score += 0.8

        # Consider other factors like test results
        output = evaluation.get("output", "")
        if "passing" in output.lower() or "passed" in output.lower():
            score += 0.2

        # Penalize for errors
        if evaluation.get("errors"):
            score -= 0.3

        return score

    def _select_branches(self, branches: List[str], reasoning_tree: Dict[str, Dict[str, Any]]) -> List[str]:
        """Select which branches to continue exploring."""
        if not branches:
            return []

        if self.selection_strategy == "breadth_first":
            # Explore all branches up to max_branches
            return branches[:self.max_branches]

        elif self.selection_strategy == "best_first":
            # Sort branches by evaluation score and take the best ones
            scored_branches = []
            for branch_id in branches:
                branch = reasoning_tree[branch_id]
                score = 0.0

                # Calculate branch score
                if branch.get("evaluation"):
                    score = self._calculate_solution_score(branch["evaluation"])

                scored_branches.append((branch_id, score))

            # Sort by score (descending) and take top branches
            scored_branches.sort(key=lambda x: x[1], reverse=True)
            return [branch_id for branch_id, _ in scored_branches[:self.max_branches]]

        else:  # Default to all branches
            return branches
