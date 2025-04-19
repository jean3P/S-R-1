# solution/issue_solver.py
import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from ..models import create_model
from ..reasoning.chain_of_thought import ChainOfThought
from ..reasoning.tree_of_thought import TreeOfThought
from ..reasoning.self_reflection import SelfReflection
from ..data.data_loader import SWEBenchDataLoader
from ..evaluation.evaluator import Evaluator
from .code_generator import CodeGenerator
from .patch_creator import PatchCreator

logger = logging.getLogger(__name__)


class IssueSolver:
    """
    Main class for solving GitHub issues from the SWE-bench dataset.
    """

    def __init__(self, config, model_name=None, reasoning_type="chain_of_thought", num_iterations=3):
        """
        Initialize the issue solver.

        Args:
            config: Configuration object.
            model_name: Name of the model to use, list of models, or None to use all models.
            reasoning_type: Type of reasoning to use ('chain_of_thought' or 'tree_of_thought').
            num_iterations: Number of reflection iterations to perform (default: 3).
        """
        self.config = config
        self.reasoning_type = reasoning_type
        self.num_iterations = num_iterations

        # Create data loader
        self.data_loader = SWEBenchDataLoader(config)

        # Determine which models to use
        if model_name is None:
            # Use all available models
            self.model_names = ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]
            logger.info(f"Using all models: {', '.join(self.model_names)}")
        elif isinstance(model_name, list):
            # Use the provided list of models
            self.model_names = model_name
            logger.info(f"Using specified models: {', '.join(self.model_names)}")
        else:
            # Use a single model
            self.model_names = [model_name]
            logger.info(f"Using single model: {model_name}")

        # We'll initialize models lazily when needed
        self.models = {}

        # Create evaluator
        self.evaluator = Evaluator(config)

        # Create code generator and patch creator
        self.code_generator = CodeGenerator(config)
        self.patch_creator = PatchCreator(config)

    def _get_model(self, model_name: str):
        """Get or initialize a model."""
        if model_name not in self.models:
            logger.info(f"Initializing model: {model_name}")
            start_time = time.time()
            try:
                self.models[model_name] = create_model(model_name, self.config)
                logger.info(f"Model {model_name} initialized in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {str(e)}")
                raise
        return self.models[model_name]

    def solve_issue(self, issue_id: str, specific_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a specific issue.

        Args:
            issue_id: ID of the issue to solve.
            specific_model: If provided, only use this model.

        Returns:
            Dictionary containing the solutions.
        """
        # Load the issue
        issue = self.data_loader.load_issue(issue_id)
        if not issue:
            raise ValueError(f"Issue {issue_id} not found")

        # Get issue description and codebase context
        issue_description = self.data_loader.get_issue_description(issue)
        codebase_context = self.data_loader.get_codebase_context(issue)

        # Get ground truth solution for evaluation
        ground_truth = self.data_loader.get_solution_patch(issue)

        # Determine which models to use for this issue
        models_to_use = [specific_model] if specific_model else self.model_names

        # Solve the issue with each model
        solutions = {}
        for model_name in models_to_use:
            try:
                # Get the model
                model = self._get_model(model_name)
                logger.info(f"Solving issue {issue_id} with {model_name}")

                # Create reasoning components
                if self.reasoning_type == "chain_of_thought":
                    reasoner = ChainOfThought(self.config, model)
                else:
                    reasoner = TreeOfThought(self.config, model)

                # Create self-reflection component
                reflector = SelfReflection(self.config, model)

                # Solve with 3 iterations for each model
                model_solutions = []
                for iteration in range(self.num_iterations):
                    logger.info(f"Starting iteration {iteration + 1}/{self.num_iterations} for {model_name}")

                    # Time the solution process
                    start_time = time.time()

                    # Generate initial solution
                    solution_data = reasoner.solve(issue_description, codebase_context)

                    # Extract the solution
                    if self.reasoning_type == "chain_of_thought":
                        initial_solution = solution_data["solution"]
                    else:
                        initial_solution = solution_data["implementation"]

                    # Apply self-reflection
                    refined_data = reflector.refine_solution(initial_solution, issue_description, codebase_context)
                    final_solution = refined_data["final_solution"]

                    # Generate code from the solution
                    code = self.code_generator.generate_code(final_solution, issue)

                    # Create patch
                    patch = self.patch_creator.create_patch(code, issue)

                    # Calculate execution time
                    execution_time = time.time() - start_time

                    # Evaluate the solution
                    evaluation = self.evaluator.evaluate_solution(
                        issue=issue,
                        solution_patch=patch,
                        ground_truth=ground_truth
                    )

                    # Save the solution data
                    model_solutions.append({
                        "iteration": iteration + 1,
                        "reasoning_steps": solution_data.get("steps", []),
                        "initial_solution": initial_solution,
                        "reflections": refined_data.get("reflections", []),
                        "final_solution": final_solution,
                        "generated_code": code,
                        "patch": patch,
                        "execution_time": execution_time,
                        "evaluation": evaluation
                    })

                solutions[model_name] = model_solutions

            except Exception as e:
                logger.error(f"Error solving issue {issue_id} with model {model_name}: {str(e)}")
                solutions[model_name] = [{"error": str(e)}]

        return {
            "issue_id": issue_id,
            "issue_description": issue_description,
            "solutions": solutions
        }

    def solve_multiple_issues(self, issue_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Solve multiple issues.

        Args:
            issue_ids: List of issue IDs to solve.

        Returns:
            List of dictionaries containing the solutions.
        """
        results = []
        for issue_id in issue_ids:
            try:
                solution = self.solve_issue(issue_id)
                results.append(solution)
            except Exception as e:
                logger.error(f"Error solving issue {issue_id}: {str(e)}")
                results.append({
                    "issue_id": issue_id,
                    "error": str(e)
                })

        return results
