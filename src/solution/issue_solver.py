# src/solution/issue_solver.py
import gc
import json
import logging
import time
from typing import Dict, List, Any, Optional
import re

import torch
import gc

from ..utils.patch_formatting_system import PatchFormattingSystem
from ..utils.llm_guidance import LLMCodeLocationGuidance
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
        Solve a specific issue using memory-efficient approach with improved SWE-bench attribute usage.
        """
        # Load the issue
        issue = self.data_loader.load_issue(issue_id)
        logger.info(f"Issue id: {issue}")
        if not issue:
            raise ValueError(f"Issue {issue_id} not found")

        # Clear memory before starting
        self._run_memory_cleanup()

        # Ensure repository exists
        if not self.repo_explorer.ensure_repository_exists(issue):
            return {"error": f"Failed to download repository for issue {issue_id}"}

        # Prepare repository with base commit for analysis
        if not self.data_loader.prepare_repository_for_analysis(issue):
            return {"error": f"Failed to prepare repository for analysis with base commit"}

        # Repository exploration phase using RAG
        logger.info(f"Starting repository exploration for issue {issue_id}")
        repo_exploration = self.repo_explorer.explore_repository(issue)

        # Add repository exploration to issue
        issue["repository_exploration"] = repo_exploration

        # Get issue description and test information
        issue_description = self.data_loader.get_issue_description(issue)

        # Get hints if available
        hints = self.data_loader.get_hints(issue)

        # Get test patch if available
        test_patch = self.data_loader.get_test_patch(issue)

        # Get FAIL_TO_PASS and PASS_TO_PASS tests
        fail_to_pass = self.data_loader.get_fail_to_pass_tests(issue)
        pass_to_pass = self.data_loader.get_pass_to_pass_tests(issue)

        # Create a more comprehensive context using all available information
        context = ""

        # Add issue description
        context += f"# ISSUE DESCRIPTION\n{issue_description}\n\n"

        # Add hints if available
        if hints:
            issue_description_with_hints = f"{issue_description}\n\nADDITIONAL HINTS:\n{hints}"
            logger.info(f"Using hints for issue {issue_id}")
        else:
            issue_description_with_hints = issue_description

        # Add test patch if available
        if test_patch:
            context += f"# TEST PATCH\n```python\n{test_patch}\n```\n\n"
            logger.info(f"Using test patch for issue {issue_id}")

        # Add test information
        if fail_to_pass:
            context += f"# TESTS THAT SHOULD BE FIXED\n"
            for test in fail_to_pass:
                context += f"* {test}\n"
            context += "\n"

        # Add repository exploration information using RAG
        context += self._format_repository_info_with_rag(repo_exploration)

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

                reflector.current_issue = issue  # This passes the full issue object to the reflector

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
                    raw_patch = bug_fixer.generate_fix(bug_location, issue_description_with_hints, repo_exploration)

                    # Process with enhanced patch formatting
                    repo_name = issue.get("repo", "")
                    patch_result = process_llm_output_with_enhanced_formatting(
                        raw_patch,
                        repo_name,
                        issue_id,
                        self.config
                    )

                    # Get the formatted patch for validation
                    formatted_patch = patch_result["formatted_patch"]

                    # Free memory after generating patch
                    self._run_memory_cleanup()

                    # Validate the patch
                    validation_result = self.patch_validator.validate_patch(formatted_patch, issue_id)
                    logger.info(f"Patch validation result: {validation_result.get('success', False)}")

                    # Calculate execution time
                    execution_time = time.time() - start_time

                    # Evaluate the solution
                    evaluation = self.evaluator.evaluate_solution(
                        issue=issue,
                        solution_patch=formatted_patch,
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
                            formatted_patch,
                            issue_description_with_hints,
                            reflection_context
                        )

                        # Update patch with refined version
                        patch = refined_data.get("final_solution", formatted_patch)
                        reflections = refined_data.get("reflections", [])

                        if not self.data_loader.prepare_repository_for_testing(issue):
                            logger.warning(f"Failed to prepare environment for testing, using base commit state")

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
                        "patch": formatted_patch,
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
            "test_patch_available": test_patch is not None,
            "fail_to_pass_tests": fail_to_pass,
            "pass_to_pass_tests": pass_to_pass,
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
        the quality of solutions through targeted self-reflection with patch validation.

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
                logger.info(f"Issue id: {issue}")
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
                logger.info(
                    f"Repository exploration completed with {len(repo_exploration.get('relevant_files', []))} relevant files")

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

                        # Create self-reflection component with patch validation
                        reflector = SelfReflection(self.config, model)

                        # Create the bug locator and bug fixer
                        bug_locator = BugLocator(model)
                        bug_fixer = BugFixer(model, self.config)  # Pass config for patch formatting

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
                        initial_patch = bug_fixer.generate_fix(bug_location, issue_description_with_hints,
                                                               repo_exploration)

                        if not self.data_loader.prepare_repository_for_testing(issue):
                            logger.warning(f"Failed to prepare environment for testing, using base commit state")

                        # Validate the initial patch
                        initial_validation = self.patch_validator.validate_patch(initial_patch, issue_id)
                        logger.info(f"Patch validation result: {initial_validation.get('success', False)}")

                        # Evaluate the initial solution
                        initial_evaluation = self.evaluator.evaluate_solution(
                            issue=issue,
                            solution_patch=initial_patch,
                            ground_truth=ground_truth
                        )

                        # Record initial patch quality
                        initial_patch_quality = initial_evaluation.get("patch_quality", 0)

                        # Apply self-reflection with patch validation for refinement
                        logger.info(f"Starting reflection with patch validation for {model_name}")
                        reflection_result = reflector.refine_solution(
                            initial_solution,
                            issue_description_with_hints,
                            context_with_bug_location
                        )

                        # Extract the final solution and patch
                        final_solution = reflection_result["final_solution"]
                        final_patch = reflection_result.get("formatted_patch", "")
                        if not final_patch:
                            # Extract patch from the solution if not provided
                            from ..utils.enhanced_patch_formatter import EnhancedPatchFormatter
                            patch_formatter = EnhancedPatchFormatter(self.config)
                            extracted_patch = self._extract_patch(final_solution)
                            final_patch = patch_formatter.format_patch(extracted_patch, issue.get("repo", ""))

                        # Validate the final patch
                        final_validation = self.patch_validator.validate_patch(final_patch, issue_id)

                        # Evaluate the final solution
                        final_evaluation = self.evaluator.evaluate_solution(
                            issue=issue,
                            solution_patch=final_patch,
                            ground_truth=ground_truth
                        )

                        # Save the solution data with all reflection iterations
                        model_solutions.append({
                            "initial_solution": initial_solution,
                            "initial_patch": initial_patch,
                            "initial_validation": initial_validation,
                            "initial_evaluation": initial_evaluation,
                            "initial_patch_quality": initial_patch_quality,
                            "reflection_result": reflection_result,
                            "final_solution": final_solution,
                            "final_patch": final_patch,
                            "final_validation": final_validation,
                            "final_evaluation": final_evaluation,
                            "bug_location": bug_location,
                            "used_hints": hints is not None,
                            "used_test_patch": test_patch is not None,
                            "success": final_validation.get("success", False)
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

    def _extract_patch(self, solution: str) -> str:
        """Extract a Git patch from the solution text."""
        # Look for a git diff format
        diff_pattern = r'(diff --git.*?)(?:\Z|(?=^```|\n\n\n))'
        diff_match = re.search(diff_pattern, solution, re.MULTILINE | re.DOTALL)

        if diff_match:
            return diff_match.group(1).strip()

        # Look for content inside code blocks that might contain patches
        code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
        code_match = re.search(code_block_pattern, solution, re.MULTILINE | re.DOTALL)

        if code_match:
            content = code_match.group(1).strip()
            if content.startswith('diff --git') or ('---' in content and '+++' in content):
                return content

        # No valid patch found
        return ""

    def _format_repository_info_with_rag(self, repo_exploration: Dict[str, Any]) -> str:
        """
        Format repository exploration results as a string for context using RAG data.

        Args:
            repo_exploration: Repository exploration results from RAG.

        Returns:
            Formatted string for model context.
        """
        if not repo_exploration or "error" in repo_exploration:
            return "# Repository exploration failed"

        repo_info = "# REPOSITORY STRUCTURE INFORMATION\n\n"

        # Add relevant files
        if "relevant_files" in repo_exploration:
            relevant_files = repo_exploration.get("relevant_files", [])
            repo_info += f"## Relevant Files ({len(relevant_files)})\n"

            # Include scores if available
            file_scores = repo_exploration.get("file_scores", [])
            if file_scores:
                # Format as table
                repo_info += "| File | Relevance |\n| ---- | -------- |\n"
                for file_path, score in file_scores[:10]:  # Limit to top 10
                    repo_info += f"| {file_path} | {score:.2f} |\n"
            else:
                # Simple list
                for file_path in relevant_files[:10]:
                    repo_info += f"* {file_path}\n"
            repo_info += "\n"

        # Add test information
        if "fail_to_pass_tests" in repo_exploration and repo_exploration["fail_to_pass_tests"]:
            repo_info += "## Tests Needing Fixes\n"
            for test in repo_exploration["fail_to_pass_tests"]:
                repo_info += f"* {test}\n"
            repo_info += "\n"

        # Add key terms
        if "key_terms" in repo_exploration and repo_exploration["key_terms"]:
            repo_info += "## Key Terms\n* " + "\n* ".join(repo_exploration["key_terms"]) + "\n\n"

        # Add identified functions
        if "functions" in repo_exploration and repo_exploration["functions"]:
            repo_info += "## Relevant Functions\n* " + "\n* ".join(repo_exploration["functions"]) + "\n\n"

        # Add file contents summary with most relevant parts first
        if "file_contents" in repo_exploration:
            file_contents = repo_exploration.get("file_contents", {})
            repo_info += "## Most Relevant Code Sections\n\n"

            # Sort file_contents by relevance_score
            sorted_files = sorted(
                file_contents.items(),
                key=lambda x: x[1].get("relevance_score", 0),
                reverse=True
            )

            for file_path, content_info in sorted_files[:5]:  # Limit to top 5 files
                if "error" in content_info:
                    continue

                repo_info += f"### {file_path}\n"

                # Show functions in this file that are most relevant
                functions = content_info.get("functions", {})
                if functions:
                    # Sort functions by likely relevance
                    func_items = sorted(functions.items())
                    for name, func_info in func_items[:3]:  # Top 3 functions
                        start_line = func_info.get('start_line', '?')
                        end_line = func_info.get('end_line', '?')
                        repo_info += f"#### Function: {name} (Lines {start_line}-{end_line})\n"

                        code = func_info.get('code', '')
                        if code:
                            repo_info += "```python\n" + code + "\n```\n\n"

                # Show classes
                classes = content_info.get("classes", [])
                if classes:
                    for cls in classes[:2]:  # Top 2 classes
                        name = cls.get('name', 'Unknown')
                        repo_info += f"#### Class: {name}\n"

                        code = cls.get('code', '')
                        if code:
                            repo_info += "```python\n" + code + "\n```\n\n"

        return repo_info


def run_with_memory_efficient_llm_guidance(config, data_loader, issue_ids, model_name,
                                           suspected_files, max_iterations, reasoning_type,
                                           reflection_iterations):
    """
    Run issues with memory-efficient LLM Code Location Guidance Framework.
    This version incorporates memory optimizations to reduce CUDA memory usage.

    Args:
        config: Configuration object.
        data_loader: SWEBenchDataLoader instance.
        issue_ids: List of issue IDs to process.
        model_name: Name of the model to use.
        suspected_files: List of suspected files.
        max_iterations: Maximum number of guidance iterations.
        reasoning_type: Type of reasoning to use.
        reflection_iterations: Number of self-reflection iterations.

    Returns:
        List of results.
    """

    # Initialize the LLM guidance framework
    guidance = LLMCodeLocationGuidance(config)

    # Initialize repository explorer if needed
    from ..utils.repository_explorer import RepositoryExplorer
    repo_explorer = RepositoryExplorer(config)

    # Initialize patch validator
    patch_validator = PatchValidator(config)

    # Initialize results list
    results = []

    # Process each issue
    for issue_id in issue_ids:
        logging.info(f"Processing issue {issue_id} with memory-efficient LLM guidance")

        # Clear memory before processing new issue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Load the issue
        issue = data_loader.load_issue(issue_id)
        logger.info(f"Issue id: {issue}")
        if not issue:
            logging.error(f"Issue {issue_id} not found")
            results.append({
                "issue_id": issue_id,
                "error": "Issue not found in dataset",
                "solutions": {}
            })
            continue

        # Get issue description
        issue_description = data_loader.get_issue_description(issue)
        logging.info(f"Loaded issue description with {len(issue_description)} characters")

        # Get repository exploration data
        try:
            repo_exploration = repo_explorer.explore_repository(issue)
        except Exception as e:
            logging.warning(f"Error exploring repository: {e}")
            repo_exploration = {
                "relevant_files": [],
                "file_scores": []
            }

        # Get additional metadata
        hints = data_loader.get_hints(issue)
        test_patch = data_loader.get_test_patch(issue)
        fail_to_pass = data_loader.get_fail_to_pass_tests(issue)
        pass_to_pass = data_loader.get_pass_to_pass_tests(issue)
        ground_truth = data_loader.get_solution_patch(issue)

        # Create the initial guidance prompt
        initial_prompt = guidance.create_guidance_prompt(
            issue,
            suspected_files=suspected_files
        )

        # Get model(s) to use
        if isinstance(model_name, list):
            models_to_use = model_name
        else:
            models_to_use = [model_name]

        # Process with each model
        issue_results = {
            "issue_id": issue_id,
            "issue_description": issue_description,
            "repository_exploration": repo_exploration,
            "ground_truth_patch": ground_truth,
            "hints_available": hints is not None,
            "test_patch_available": test_patch is not None,
            "fail_to_pass_tests": fail_to_pass,
            "pass_to_pass_tests": pass_to_pass,
            "solutions": {}
        }

        for model_id in models_to_use:
            logging.info(f"Using model {model_id} for issue {issue_id}")

            # Clear memory before loading model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Initialize model
            try:
                model = create_model(model_id, config)
            except Exception as e:
                logging.error(f"Error initializing model {model_id}: {e}")
                issue_results["solutions"][model_id] = [{
                    "error": f"Model initialization failed: {str(e)}",
                    "iteration": 1
                }]
                continue

            # Track iterations
            guided_iterations = []
            current_prompt = initial_prompt

            # Run guidance iterations
            for iteration in range(max_iterations):
                logging.info(f"Guidance iteration {iteration + 1}/{max_iterations}")

                # Clear memory before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                # Generate analysis with model
                try:
                    analysis_response = model.generate(current_prompt)
                    logging.info(f"Generated response with {len(analysis_response)} characters")
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logging.warning("CUDA OOM during guidance. Trying with reduced context...")
                        # Truncate prompt and try again
                        truncated_prompt = _truncate_prompt(current_prompt)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        analysis_response = model.generate(truncated_prompt)
                    else:
                        logging.error(f"Error generating response: {e}")
                        analysis_response = f"Error occurred: {str(e)}"

                # Extract patch from response
                patch = guidance.extract_patch_from_response(analysis_response)

                # If no patch found, try to create a minimal one based on the model's explanation
                if not patch or len(patch.strip()) < 10:
                    logging.warning("No valid patch extracted, trying to create a minimal one")
                    patch = guidance.create_minimal_patch_from_response(analysis_response, issue)

                logging.info(f"Extracted/created patch with {len(patch)} characters")

                # Clear memory before validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                # Validate patch
                validation_result = patch_validator.validate_patch(patch, issue_id)
                success = validation_result.get("success", False)
                logging.info(f"Patch validation result: {success}")

                # Record iteration
                guided_iterations.append({
                    "iteration": iteration + 1,
                    "response": analysis_response,
                    "patch": patch,
                    "validation": validation_result
                })

                # If patch is valid or we've reached max iterations, proceed to solver
                if success or iteration + 1 >= max_iterations:
                    break

                # Otherwise, refine prompt with feedback
                feedback = f"The patch has issues: {validation_result.get('feedback', 'Unknown validation error')}"
                current_prompt = guidance.apply_feedback(current_prompt, analysis_response, feedback)

            # Apply self-reflection if any valid patch was found
            final_solutions = []

            # Always process all iterations, even if none have valid patches
            # This ensures we at least have something in our results
            for iteration in guided_iterations:
                # Create a solution entry regardless of patch quality
                solution = {
                    "iteration": iteration["iteration"],
                    "guidance_response": iteration["response"],
                    "patch": iteration["patch"],
                    "patch_validation": iteration["validation"],
                    "guided_fix": True
                }

                # If we should apply self-reflection, do so
                if reflection_iterations > 0 and len(iteration["patch"]) > 0:
                    from ..reasoning.self_reflection import SelfReflection

                    # Clear memory before reflection
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    try:
                        reflector = SelfReflection(config, model)

                        # Set the current issue for proper issue ID extraction
                        reflector.current_issue = issue

                        # Ensure issue_description is a string
                        issue_desc_str = issue_description if isinstance(issue_description, str) else str(
                            issue_description)

                        # Create context for reflection
                        reflection_context = (
                            f"Initial patch based on guided analysis:\n{iteration['patch']}\n\n"
                            f"Validation feedback: {iteration['validation'].get('feedback', '')}"
                        )

                        # Apply self-reflection to refine the solution
                        refined_data = reflector.refine_solution(
                            iteration["patch"],
                            issue_desc_str,
                            reflection_context
                        )

                        # Update the solution with reflection results
                        solution["reflections"] = refined_data.get("reflections", [])
                        solution["final_solution"] = refined_data.get("final_solution", iteration["patch"])
                    except Exception as e:
                        logging.error(f"Error during reflection: {e}")
                        solution["reflection_error"] = str(e)

                final_solutions.append(solution)

            # Always add solutions to results, even if empty
            issue_results["solutions"][model_id] = final_solutions
            logging.info(f"Added {len(final_solutions)} solutions for model {model_id}")

            # Clean up model to free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Add issue results to overall results
        results.append(issue_results)
        logging.info(f"Added results for issue {issue_id}")

    return results


def _truncate_prompt(prompt: str, max_length: int = 8000) -> str:
    """
    Truncate a prompt to reduce memory usage.

    Args:
        prompt: The original prompt
        max_length: Maximum length to keep

    Returns:
        Truncated prompt
    """
    if len(prompt) <= max_length:
        return prompt

    # Add a note about truncation
    truncation_note = "\n[Note: The context was too long and has been truncated.]\n\n"

    # Keep the beginning and end of the prompt
    beginning_length = max_length // 2
    ending_length = max_length - beginning_length - len(truncation_note)

    return prompt[:beginning_length] + truncation_note + prompt[-ending_length:]


def process_llm_output_with_enhanced_formatting(
        patch: str,
        repo_name: str,
        issue_id: str,
        config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process LLM-generated patch output with enhanced formatting.

    Args:
        patch: The raw patch string from the LLM
        repo_name: Repository name
        issue_id: Issue ID for validation
        config: Configuration object

    Returns:
        Dictionary with processing results
    """
    # Initialize the patch formatting system
    patch_system = PatchFormattingSystem(config)

    # Format and validate the patch
    result = patch_system.format_and_validate(patch, repo_name, issue_id)

    if result["success"]:
        logger.info(f"Successfully formatted and validated patch for issue {issue_id}")
    else:
        logger.warning(f"Patch validation failed for issue {issue_id}")
        if "validation" in result and "feedback" in result["validation"]:
            logger.warning(f"Validation feedback: {result['validation']['feedback']}")

    # Create a summary of the patch for logging/reporting
    summary = patch_system.summarize_patch(result["formatted_patch"])
    logger.info(f"Patch summary:\n{summary}")

    return result