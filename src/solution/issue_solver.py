# solution/issue_solver.py

import json
import logging
import time
from typing import Dict, List, Any, Optional
import re

from ..models import create_model
from ..reasoning.chain_of_thought import ChainOfThought
from ..reasoning.tree_of_thought import TreeOfThought
from ..reasoning.self_reflection import SelfReflection
from ..data.data_loader import SWEBenchDataLoader
from ..evaluation.evaluator import Evaluator
from ..solution.code_generator import CodeGenerator
from ..solution.improved_patch_creator import ImprovedPatchCreator
from ..utils.repository_explorer import RepositoryExplorer
from ..utils.patch_validator import PatchValidator
from ..utils.bug_locator import BugLocator
from ..utils.bug_fixer import BugFixer


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
        self.patch_creator = ImprovedPatchCreator(config)

        # NEW: Create repository explorer and patch validator
        self.repo_explorer = RepositoryExplorer(config)
        self.patch_validator = PatchValidator(config)

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
        Solve a specific issue with a focused bug-fixing approach.

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

        # Ensure repository exists
        if not self.repo_explorer.ensure_repository_exists(issue):
            return {"error": f"Failed to download repository for issue {issue_id}"}

        # Repository exploration phase
        logger.info(f"Starting repository exploration for issue {issue_id}")
        repo_exploration = self.repo_explorer.explore_repository(issue)
        logger.info(
            f"Repository exploration completed with {len(repo_exploration.get('relevant_files', []))} relevant files")

        # Add repository exploration to issue
        issue["repository_exploration"] = repo_exploration

        # Create enhanced context with repository information
        repo_info_str = self._format_repository_info(repo_exploration)

        # Get issue description and codebase context
        issue_description = self.data_loader.get_issue_description(issue)
        codebase_context = self.data_loader.get_codebase_context(issue)

        # Add repository information to context
        enhanced_context = f"{codebase_context}\n\n{repo_info_str}"

        # Get hints if available
        hints = self.data_loader.get_hints(issue)

        # Append hints to the issue description if available
        if hints:
            issue_description_with_hints = f"{issue_description}\n\nADDITIONAL HINTS:\n{hints}"
            logger.info(f"Using hints for issue {issue_id}")
        else:
            issue_description_with_hints = issue_description

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

                # Create the bug locator and bug fixer
                bug_locator = BugLocator(model)
                bug_fixer = BugFixer(model)

                model_solutions = []

                # Step 1: First locate the bug
                logger.info(f"Locating bug for issue {issue_id} with {model_name}")
                bug_location = bug_locator.locate_bug(issue_description_with_hints, repo_exploration)
                logger.info(f"Bug located: {bug_location}")

                # Step 2: Generate and refine solutions for each iteration
                for iteration in range(self.num_iterations):
                    logger.info(f"Starting iteration {iteration + 1}/{self.num_iterations} for {model_name}")

                    # Time the solution process
                    start_time = time.time()

                    # Generate initial solution using reasoning approach
                    context_with_bug_location = f"{enhanced_context}\n\nBUG LOCATION:\n{json.dumps(bug_location, indent=2)}"

                    # Use reasoning to generate initial solution
                    solution_data = reasoner.solve(issue_description_with_hints, context_with_bug_location)

                    # Extract the initial solution based on reasoning type
                    if self.reasoning_type == "chain_of_thought":
                        initial_solution = solution_data["solution"]
                    else:  # tree_of_thought
                        initial_solution = solution_data["implementation"]

                    # Apply self-reflection to refine the solution
                    if iteration > 0:  # Only use reflection for iterations after the first
                        refined_data = reflector.refine_solution(
                            initial_solution,
                            issue_description_with_hints,
                            context_with_bug_location
                        )
                        final_solution = refined_data["final_solution"]
                        reflections = refined_data.get("reflections", [])
                    else:
                        final_solution = initial_solution
                        reflections = []

                    # Generate patch from the solution
                    patch = bug_fixer.generate_fix(bug_location, issue_description_with_hints, repo_exploration)

                    # Validate the patch
                    validation_result = self.patch_validator.validate_patch(patch, issue_id)
                    logger.info(f"Patch validation result: {validation_result.get('success', False)}")

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
                        "reflections": reflections,
                        "final_solution": final_solution,
                        "bug_location": bug_location,
                        "patch": patch,
                        "patch_validation": validation_result,
                        "execution_time": execution_time,
                        "evaluation": evaluation,
                        "used_hints": hints is not None
                    })

                    # If validation failed, refine the bug location for next iteration
                    if not validation_result.get("success", False) and iteration + 1 < self.num_iterations:
                        feedback = validation_result.get("feedback", "")
                        logger.info(f"Refining bug location based on validation feedback")

                        # Update bug location with validation feedback for next iteration
                        bug_location = self._refine_bug_location(model, bug_location, feedback,
                                                                 issue_description_with_hints)

                solutions[model_name] = model_solutions

            except Exception as e:
                logger.error(f"Error solving issue {issue_id} with model {model_name}: {str(e)}")
                solutions[model_name] = [{"error": str(e)}]

        return {
            "issue_id": issue_id,
            "issue_description": issue_description,
            "repository_exploration": repo_exploration,
            "ground_truth_patch": ground_truth,
            "hints_available": hints is not None,
            "solutions": solutions
        }

    def _refine_bug_location(self, model, bug_location: Dict[str, Any],
                             feedback: str, issue_description: str) -> Dict[str, Any]:
        """Refine bug location based on validation feedback."""
        logger.info(f"Refining bug location based on validation feedback")

        refined_prompt = f"""
        Based on the following patch validation feedback, refine your bug location analysis:

        ISSUE DESCRIPTION:
        {issue_description}

        PREVIOUS BUG LOCATION:
        {json.dumps(bug_location, indent=2)}

        VALIDATION FEEDBACK:
        {feedback}

        Provide a revised bug location that will help create a more accurate patch.
        Include specific file, function, and line number details.
        """

        response = model.generate(refined_prompt)

        # Parse the refined bug location
        file_match = re.search(r'FILE:\s*(.+?)(?:\n|$)', response)
        function_match = re.search(r'FUNCTION:\s*(.+?)(?:\n|$)', response)
        line_match = re.search(r'LINE NUMBERS?:\s*(.+?)(?:\n|$)', response)
        issue_match = re.search(r'ISSUE:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)

        refined_location = {
            "file": file_match.group(1).strip() if file_match else bug_location.get("file"),
            "function": function_match.group(1).strip() if function_match else bug_location.get("function"),
            "line_numbers": line_match.group(1).strip() if line_match else bug_location.get("line_numbers"),
            "issue": issue_match.group(1).strip() if issue_match else bug_location.get("issue")
        }

        return refined_location

    def _format_repository_info(self, repo_exploration: Dict[str, Any]) -> str:
        """
        Format repository exploration results as a string for context.
        """
        if not repo_exploration or "error" in repo_exploration:
            return "# Repository exploration failed"

        repo_info = "# REPOSITORY STRUCTURE INFORMATION\n\n"

        # Add relevant files
        if "relevant_files" in repo_exploration:
            relevant_files = repo_exploration.get("relevant_files", [])
            repo_info += f"## Relevant Files ({len(relevant_files)})\n"
            for file_path in relevant_files:
                repo_info += f"* {file_path}\n"
            repo_info += "\n"

        # Add file contents summary
        if "file_contents" in repo_exploration:
            file_contents = repo_exploration.get("file_contents", {})
            repo_info += "## File Contents Summary\n\n"

            for file_path, content_info in file_contents.items():
                if "error" in content_info:
                    continue

                repo_info += f"### {file_path}\n"
                repo_info += f"* Lines: {content_info.get('lines_count', 'Unknown')}\n"

                # List functions
                functions = content_info.get("functions", [])
                if functions:
                    repo_info += "* Functions:\n"
                    for func in functions[:5]:  # Limit to 5 functions
                        repo_info += f"  * {func['name']} (Lines {func['start_line']}-{func['end_line']})\n"
                    if len(functions) > 5:
                        repo_info += f"  * ... and {len(functions) - 5} more functions\n"

                # List classes
                classes = content_info.get("classes", [])
                if classes:
                    repo_info += "* Classes:\n"
                    for cls in classes[:5]:  # Limit to 5 classes
                        repo_info += f"  * {cls['name']} (Lines {cls['start_line']}-{cls['end_line']})\n"
                    if len(classes) > 5:
                        repo_info += f"  * ... and {len(classes) - 5} more classes\n"

                repo_info += "\n"

        return repo_info

    def _generate_validation_feedback(self, previous_patches: List[Dict[str, Any]], issue_id: str) -> str:
        """
        Generate feedback based on previous patch validation results.
        """
        feedback = "# PATCH VALIDATION FEEDBACK\n\n"

        for i, patch_info in enumerate(previous_patches):
            validation = patch_info.get("validation", {})
            success = validation.get("success", False)

            feedback += f"## Patch Attempt {i + 1}\n"
            feedback += f"Status: {'SUCCESS' if success else 'FAILURE'}\n\n"

            if not success and "feedback" in validation:
                feedback += "Validation feedback:\n"
                feedback += validation["feedback"]
                feedback += "\n\n"

        feedback += "Please use this feedback to improve your next patch attempt.\n"
        return feedback

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

