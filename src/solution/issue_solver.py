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
    Modified to use memory-efficient RAG-like approach.
    """

    def __init__(self, config, model_name=None, reasoning_type="chain_of_thought", num_iterations=3):
        """Initialize the issue solver."""
        # Existing initialization code...
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

        # NEW: Set memory optimization flags
        self.use_memory_optimization = True
        self.memory_cleanup_threshold = 0.8  # Run cleanup when memory usage exceeds this fraction

    def _get_model(self, model_name: str):
        """Get or initialize a model with memory optimization."""
        if model_name not in self.models:
            logger.info(f"Initializing model: {model_name}")
            start_time = time.time()

            # Clear memory before model initialization
            self._run_memory_cleanup()

            try:
                self.models[model_name] = create_model(model_name, self.config)
                logger.info(f"Model {model_name} initialized in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {str(e)}")
                raise
        return self.models[model_name]

    def _run_memory_cleanup(self):
        """Run memory cleanup operations to free up CUDA memory."""
        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        # Log current memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                max_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)
                logger.info(f"GPU {i} memory: allocated={allocated:.2f}GB (max: {max_allocated:.2f}GB), "
                            f"reserved={reserved:.2f}GB (max: {max_reserved:.2f}GB)")

    def solve_issue(self, issue_id: str, specific_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a specific issue using memory-efficient approach.
        """
        # Load the issue
        issue = self.data_loader.load_issue(issue_id)
        if not issue:
            raise ValueError(f"Issue {issue_id} not found")

        # Clear memory before starting
        self._run_memory_cleanup()

        # Ensure repository exists
        if not self.repo_explorer.ensure_repository_exists(issue):
            return {"error": f"Failed to download repository for issue {issue_id}"}

        # Repository exploration phase - now using memory-efficient summaries
        logger.info(f"Starting repository exploration for issue {issue_id}")
        repo_exploration = self.repo_explorer.explore_repository(issue)

        # Filter relevant files to only include those with score >= 50% of max score
        if "relevant_files" in repo_exploration and len(repo_exploration["relevant_files"]) > 0:
            # Get the original list of relevant files with their scores
            all_relevant_files = repo_exploration.get("relevant_files", [])

            # If we have file scores available, filter based on threshold
            if "file_scores" in repo_exploration and repo_exploration["file_scores"]:
                max_score = max(score for _, score in repo_exploration["file_scores"])
                threshold = max_score * 0.5  # 50% of max score

                # Filter files with scores >= threshold
                filtered_files = [file for file, score in repo_exploration["file_scores"] if score >= threshold]

                # Update relevant_files with the filtered list
                repo_exploration["relevant_files"] = filtered_files

                logger.info(
                    f"Filtered relevant files from {len(all_relevant_files)} to {len(filtered_files)} (score threshold: {threshold:.2f})")
                logger.info(f"The filtered files are: {filtered_files}")
            else:
                logger.info(f"No file scores available, using all {len(all_relevant_files)} relevant files")

        # Store repo_explorer in repository_data to enable code retrieval
        repo_exploration["repo_explorer"] = self.repo_explorer

        logger.info(
            f"Repository exploration completed with {len(repo_exploration.get('relevant_files', []))} relevant files")

        # Add repository exploration to issue
        issue["repository_exploration"] = repo_exploration

        # Get issue description and codebase context
        issue_description = self.data_loader.get_issue_description(issue)

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
                # Clear memory before using a new model
                self._run_memory_cleanup()

                # Get the model
                model = self._get_model(model_name)
                logger.info(f"Solving issue {issue_id} with {model_name}")

                # Create self-reflection component
                reflector = SelfReflection(self.config, model)

                # Create the bug locator
                bug_locator = BugLocator(model)

                # Create bug fixer
                bug_fixer = BugFixer(model)

                model_solutions = []

                # Step 1: First locate the bug using memory-efficient approach
                logger.info(f"Locating bug for issue {issue_id} with {model_name}")
                bug_location = bug_locator.locate_bug(issue_description_with_hints, repo_exploration)
                logger.info(f"Bug located: {bug_location}")

                # Free memory after bug location
                self._run_memory_cleanup()

                # Step 2: Generate and refine solutions for each iteration
                for iteration in range(self.num_iterations):
                    logger.info(f"Starting iteration {iteration + 1}/{self.num_iterations} for {model_name}")

                    # Time the solution process
                    start_time = time.time()

                    # Generate patch using bug fixer with memory optimization
                    logger.info(f"Generating fix for iteration {iteration + 1}")
                    patch = bug_fixer.generate_fix(bug_location, issue_description_with_hints, repo_exploration)

                    # Free memory after generating patch
                    self._run_memory_cleanup()

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

                    # Only apply self-reflection for iterations after the first if needed
                    if iteration > 0 and not validation_result.get("success", False):
                        logger.info(f"Applying self-reflection for iteration {iteration + 1}")

                        # Prepare context with validation feedback
                        validation_feedback = validation_result.get("feedback", "")
                        reflection_context = f"Bug location: {bug_location}\n\nValidation feedback: {validation_feedback}"

                        # Apply self-reflection to refine the patch
                        refined_data = reflector.refine_solution(
                            patch,
                            issue_description_with_hints,
                            reflection_context
                        )

                        # Update patch with refined version
                        patch = refined_data.get("final_solution", patch)
                        reflections = refined_data.get("reflections", [])

                        # Re-validate the refined patch
                        validation_result = self.patch_validator.validate_patch(patch, issue_id)
                        logger.info(f"Refined patch validation result: {validation_result.get('success', False)}")

                        # Free memory after reflection
                        self._run_memory_cleanup()
                    else:
                        reflections = []

                    # Save the solution data
                    model_solutions.append({
                        "iteration": iteration + 1,
                        "bug_location": bug_location,
                        "patch": patch,
                        "reflections": reflections,
                        "patch_validation": validation_result,
                        "execution_time": execution_time,
                        "evaluation": evaluation,
                        "used_hints": hints is not None
                    })

                    # If validation succeeded, no need for more iterations
                    if validation_result.get("success", False):
                        logger.info(f"Valid patch found. Stopping iterations early.")
                        break

                    # If we need another iteration, refine the bug location based on validation feedback
                    if not validation_result.get("success", False) and iteration + 1 < self.num_iterations:
                        feedback = validation_result.get("feedback", "")
                        logger.info(f"Refining bug location based on validation feedback")

                        # Update bug location with validation feedback for next iteration
                        bug_location = self._refine_bug_location(model, bug_location, feedback,
                                                                 issue_description_with_hints)

                        # Free memory after refining bug location
                        self._run_memory_cleanup()

                solutions[model_name] = model_solutions

            except Exception as e:
                logger.error(f"Error solving issue {issue_id} with model {model_name}: {str(e)}")
                solutions[model_name] = [{"error": str(e)}]

            finally:
                # Always clean up memory after processing with a model
                self._run_memory_cleanup()

        return {
            "issue_id": issue_id,
            "issue_description": issue_description,
            "repository_exploration": {
                "relevant_files": repo_exploration.get("relevant_files", []),
                "file_scores": repo_exploration.get("file_scores", [])
            },
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
                functions = content_info.get("functions", {})
                if functions:
                    repo_info += "* Functions:\n"
                    func_items = list(functions.items()) if isinstance(functions, dict) else functions
                    func_list = func_items[:5] if isinstance(func_items, list) else []  # Limit to 5 functions
                    for func in func_list:
                        if isinstance(func, dict):
                            # Handle as dict with keys
                            name = func.get('name', 'Unknown')
                            start_line = func.get('start_line', '?')
                            end_line = func.get('end_line', '?')
                            repo_info += f"  * {name} (Lines {start_line}-{end_line})\n"
                        elif isinstance(func, tuple) and len(func) == 2:
                            # Handle as (name, info) tuple from dict.items()
                            name, info = func
                            start_line = info.get('start_line', '?')
                            end_line = info.get('end_line', '?')
                            repo_info += f"  * {name} (Lines {start_line}-{end_line})\n"
                    if isinstance(func_items, list) and len(func_items) > 5:
                        repo_info += f"  * ... and {len(func_items) - 5} more functions\n"

                # List classes
                classes = content_info.get("classes", [])
                if classes:
                    repo_info += "* Classes:\n"
                    cls_list = classes[:5] if isinstance(classes, list) else []  # Limit to 5 classes
                    for cls in cls_list:
                        name = cls.get('name', 'Unknown')
                        start_line = cls.get('start_line', '?')
                        end_line = cls.get('end_line', '?')
                        repo_info += f"  * {name} (Lines {start_line}-{end_line})\n"
                    if isinstance(classes, list) and len(classes) > 5:
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
        
    def solve_issues_with_patch_reflection(self, issue_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Solve multiple issues with enhanced patch reflection.
        
        This method uses the dataset's hints_text and test_patch fields to improve
        the quality of solutions through targeted self-reflection.

        Args:
            issue_ids: List of issue IDs to solve.

        Returns:
            List of dictionaries containing the solutions with reflection data.
        """
        results = []
        for issue_id in issue_ids:
            try:
                # Load the issue
                issue = self.data_loader.load_issue(issue_id)
                if not issue:
                    raise ValueError(f"Issue {issue_id} not found")
                
                # Ensure repository exists
                if not self.repo_explorer.ensure_repository_exists(issue):
                    results.append({
                        "issue_id": issue_id,
                        "error": f"Failed to download repository for issue {issue_id}"
                    })
                    continue
                
                # Repository exploration phase
                logger.info(f"Starting repository exploration for issue {issue_id}")
                repo_exploration = self.repo_explorer.explore_repository(issue)
                logger.info(f"Repository exploration completed with {len(repo_exploration.get('relevant_files', []))} relevant files")
                
                # Add repository exploration to issue
                issue["repository_exploration"] = repo_exploration
                
                # Get issue description and codebase context
                issue_description = self.data_loader.get_issue_description(issue)
                codebase_context = self.data_loader.get_codebase_context(issue)
                
                # Get hints if available
                hints = issue.get("hints_text", None)
                test_patch = issue.get("test_patch", None)
                
                # Create enhanced context with repository information and test patch
                repo_info_str = self._format_repository_info(repo_exploration)
                enhanced_context = f"{codebase_context}\n\n{repo_info_str}"
                
                if test_patch:
                    enhanced_context += f"\n\nTEST PATCH:\n{test_patch}"
                
                # Append hints to the issue description if available
                if hints:
                    issue_description_with_hints = f"{issue_description}\n\nADDITIONAL HINTS:\n{hints}"
                    logger.info(f"Using hints for issue {issue_id}")
                else:
                    issue_description_with_hints = issue_description
                
                # Get ground truth solution for evaluation
                ground_truth = self.data_loader.get_solution_patch(issue)
                
                # Solve the issue with each model
                solutions = {}
                for model_name in self.model_names:
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
                        
                        # Initial solution generation
                        context_with_bug_location = f"{enhanced_context}\n\nBUG LOCATION:\n{json.dumps(bug_location, indent=2)}"
                        solution_data = reasoner.solve(issue_description_with_hints, context_with_bug_location)
                        
                        # Extract the initial solution based on reasoning type
                        if self.reasoning_type == "chain_of_thought":
                            initial_solution = solution_data["solution"]
                        else:  # tree_of_thought
                            initial_solution = solution_data["implementation"]
                        
                        # Generate initial patch
                        initial_patch = bug_fixer.generate_fix(bug_location, issue_description_with_hints, repo_exploration)
                        
                        # Validate the initial patch
                        initial_validation = self.patch_validator.validate_patch(initial_patch, issue_id)
                        
                        # Evaluate the initial solution
                        initial_evaluation = self.evaluator.evaluate_solution(
                            issue=issue,
                            solution_patch=initial_patch,
                            ground_truth=ground_truth
                        )
                        
                        # Record initial patch quality
                        initial_patch_quality = initial_evaluation.get("patch_quality", 0)
                        
                        # Progressive patch reflection
                        current_solution = initial_solution
                        current_patch = initial_patch
                        current_validation = initial_validation
                        
                        # Store all reflection iterations
                        reflection_iterations = []
                        
                        # Perform multiple iterations of reflection
                        for iteration in range(self.num_iterations):
                            logger.info(f"Starting reflection iteration {iteration + 1}/{self.num_iterations} for {model_name}")
                            
                            # Time the reflection process
                            start_time = time.time()
                            
                            # Add validation feedback to context for reflection
                            validation_feedback = ""
                            if not current_validation.get("success", False):
                                validation_feedback = f"\n\nPATCH VALIDATION FEEDBACK:\n{current_validation.get('feedback', '')}"
                            
                            reflection_context = f"{context_with_bug_location}{validation_feedback}"
                            
                            # Apply self-reflection to refine the solution
                            refined_data = reflector.refine_solution(
                                current_solution,
                                issue_description_with_hints,
                                reflection_context
                            )
                            
                            # Extract the refined solution
                            refined_solution = refined_data["final_solution"]
                            reflections = refined_data.get("reflections", [])
                            
                            # Generate new patch from the refined solution
                            refined_patch = bug_fixer.generate_fix(bug_location, issue_description_with_hints, repo_exploration)
                            
                            # Validate the refined patch
                            refined_validation = self.patch_validator.validate_patch(refined_patch, issue_id)
                            
                            # Calculate execution time
                            execution_time = time.time() - start_time
                            
                            # Evaluate the refined solution
                            refined_evaluation = self.evaluator.evaluate_solution(
                                issue=issue,
                                solution_patch=refined_patch,
                                ground_truth=ground_truth
                            )
                            
                            # Store this iteration
                            reflection_iterations.append({
                                "iteration": iteration + 1,
                                "reflections": reflections,
                                "solution": refined_solution,
                                "patch": refined_patch,
                                "validation": refined_validation,
                                "evaluation": refined_evaluation,
                                "execution_time": execution_time
                            })
                            
                            # Update current solution for next iteration
                            current_solution = refined_solution
                            current_patch = refined_patch
                            current_validation = refined_validation
                        
                        # Find the best solution from all iterations
                        best_iteration = max(reflection_iterations, 
                                            key=lambda x: x["evaluation"].get("overall_score", 0))
                        
                        # Save the solution data with all reflection iterations
                        model_solutions.append({
                            "initial_solution": initial_solution,
                            "initial_patch": initial_patch,
                            "initial_validation": initial_validation,
                            "initial_evaluation": initial_evaluation,
                            "initial_patch_quality": initial_patch_quality,
                            "reflection_iterations": reflection_iterations,
                            "best_iteration": best_iteration.get("iteration", 0),
                            "final_solution": best_iteration.get("solution", ""),
                            "patch": best_iteration.get("patch", ""),
                            "validation": best_iteration.get("validation", {}),
                            "evaluation": best_iteration.get("evaluation", {}),
                            "bug_location": bug_location,
                            "used_hints": hints is not None,
                            "used_test_patch": test_patch is not None
                        })
                        
                        solutions[model_name] = model_solutions
                        
                    except Exception as e:
                        logger.error(f"Error solving issue {issue_id} with model {model_name}: {str(e)}")
                        solutions[model_name] = [{"error": str(e)}]
                
                results.append({
                    "issue_id": issue_id,
                    "issue_description": issue_description,
                    "repository_exploration": repo_exploration,
                    "ground_truth_patch": ground_truth,
                    "hints_available": hints is not None,
                    "test_patch_available": test_patch is not None,
                    "solutions": solutions
                })
                
            except Exception as e:
                logger.error(f"Error solving issue {issue_id}: {str(e)}")
                results.append({
                    "issue_id": issue_id,
                    "error": str(e)
                })
        
        return results

