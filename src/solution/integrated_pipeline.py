# src/solution/integreated_pipeline.py

import logging
import re
import time
from pathlib import Path
import tempfile
import subprocess
import os
from typing import Dict, Any, Optional
import torch
import gc
import json

from ..data.astropy_synthetic_dataloader import AstropySyntheticDataLoader
from ..models import create_model
from ..utils.patch_validator import PatchValidator
from ..reasoning.enhanced_chain_of_thought import EnhancedChainOfThought
from ..reasoning.self_reflection import SelfReflection

logger = logging.getLogger(__name__)


class IntegratedBugFixingPipeline:
    """
    Implements a bug fixing pipeline that uses the output of
    the bug detector, Chain of Thought solution development,
    and Self-Reflection to iteratively generate and validate fixes.
    """

    def __init__(self, config, model_name=None):
        """
        Initialize the bug fixing pipeline.

        Args:
            config: Configuration object.
            model_name: Name of the model to use, or None to use default.
        """
        self.config = config
        self.model_name = model_name or config.get("default_model", "deepseek-r1-distill")
        self.model = None  # Lazy-load model on demand

        # Initialize core components
        self.data_loader = AstropySyntheticDataLoader(config)
        self.patch_validator = PatchValidator(config)

        # Configure parameters
        self.max_cot_iterations = config.get("max_cot_iterations", 3)  # Default to 3 iterations
        self.max_reflection_iterations = config.get("max_reflection_iterations", 3)  # Default to 3 iterations
        self.max_total_iterations = config.get("max_total_iterations", 8)
        self.max_test_runs = config.get("max_test_runs", 3)  # Maximum attempts to run tests
        self.test_timeout = config.get("test_timeout", 300)  # Timeout for test execution in seconds

        # Memory optimization
        self.use_memory_optimization = config.get("memory_efficient", True)

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

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(f"GPU {i} memory: allocated={allocated:.2f}GB (max: {max_allocated:.2f}GB), "
                            f"reserved={reserved:.2f}GB")

    def solve_issue(self, issue_id: str, previous_solution=None):
        """
        Solve an issue using the pipeline with the context-based bug detector.

        Args:
            issue_id: Issue ID to solve.
            previous_solution: Optional dictionary with previous best solution and feedback.

        Returns:
            Dictionary with solution results.
        """
        logger.info(f"Starting bug fixing pipeline for issue {issue_id}")
        start_time = time.time()

        # Handle previous solution feedback if available
        if previous_solution:
            logger.info(f"Using feedback from previous iteration with depth {previous_solution.get('depth', 0.0)}")

        # Load issue
        issue = self.data_loader.load_issue(issue_id)
        if not issue:
            return {"error": f"Issue {issue_id} not found"}

        # Get issue description
        issue_description = self.data_loader.get_issue_description(issue)

        # Initialize result tracking
        result = {
            "issue_id": issue_id,
            "phases": [],
            "solution": None,
            "success": False,
            "depth_scores": {
                "location_specificity": 0.0,
                "solution_completeness": 0.0,
                "combined": 0.0
            },
            "previous_solution_used": previous_solution is not None
        }

        # Apply previous solution feedback if available
        if previous_solution:
            result["previous_solution"] = {
                "depth": previous_solution.get("depth", 0.0),
                "feedback": previous_solution.get("feedback", []),
                "iteration": previous_solution.get("iteration", 0),
                "solution": previous_solution.get("solution")
            }

        # Run the pipeline
        try:
            # Phase 1: Load bug location from bug_locations.json file
            bug_location = self._load_bug_location(issue_id)

            if not bug_location or "error" in bug_location:
                logger.error(f"Failed to load bug location: {bug_location.get('error', 'Unknown error')}")
                return {"error": f"Failed to load bug location: {bug_location.get('error', 'Unknown error')}"}

            # Calculate location specificity
            location_specificity = self._calculate_location_specificity(bug_location)

            # Add phase info
            result["phases"].append({
                "name": "bug_detection",
                "bug_location": bug_location,
                "depth": location_specificity,
                "early_stopped": False
            })

            result["depth_scores"]["location_specificity"] = location_specificity

            # Phase 2: Self-Reflection and Chain of Thought Solution Development
            # Use previous patch if available
            initial_patch = None
            initial_solution_text = None
            if previous_solution and previous_solution.get("solution"):
                prev_sol = previous_solution.get("solution")
                if isinstance(prev_sol, dict):
                    initial_patch = prev_sol.get("patch")
                    initial_solution_text = prev_sol.get("solution_text")
                    if initial_patch:
                        logger.info(f"Using patch from previous solution as starting point")

            # Run the self-reflection and CoT solution phase
            solution_result = self._run_enhanced_solution_phase(
                issue,
                issue_description,
                bug_location,
                initial_patch,
                initial_solution_text
            )

            result["phases"].append({
                "name": "enhanced_solution",
                "iterations": solution_result["iterations"],
                "depth": solution_result["depth"],
                "early_stopped": solution_result["early_stopped"],
                "test_runs": sum(1 for iteration in solution_result.get("iterations", [])
                                 if iteration.get("test_validation", {}).get("test_run", False))
            })

            result["depth_scores"]["solution_completeness"] = solution_result["depth"]
            result["total_iterations"] = len(solution_result["iterations"])

            # Store final solution
            result["solution"] = solution_result["solution"]
            result["success"] = solution_result.get("success", False)

            # Calculate combined depth score
            result["depth_scores"]["combined"] = self._calculate_combined_depth(result["depth_scores"])

            return self._finalize_result(result, issue, start_time)

        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
            result["error"] = str(e)
            return result

    def _load_bug_location(self, issue_id: str) -> Dict[str, Any]:
        """
        Load the bug location from the bug_locations.json file.

        Args:
            issue_id: Issue ID to load bug location for.

        Returns:
            Dictionary with bug location information.
        """
        logger.info(f"Loading bug location for issue {issue_id}")

        # Common JSON file paths for bug locations
        bug_locations_json_paths = [
            "/storage/homefs/jp22b083/SSI/S-R-1/results/enhanced_bug_detector/bug_locations.json"
        ]

        # Try loading from the combined JSON file
        for path in bug_locations_json_paths:
            try:
                with open(path, 'r') as f:
                    all_bug_locations = json.load(f)
                    # Find matching issue by ID
                    for bug_location in all_bug_locations:
                        if bug_location.get("issue_id") == issue_id:
                            logger.info(f"Found bug location for issue {issue_id}")
                            # Extract only the necessary information as specified
                            return {
                                "file": bug_location.get("file", ""),
                                "function": bug_location.get("function", ""),
                                "line_start": bug_location.get("line_start", 0),
                                "line_end": bug_location.get("line_end", 0),
                                "code_content": bug_location.get("code_content", ""),
                                "bug_type": bug_location.get("bug_type", ""),
                                "bug_description": bug_location.get("bug_description", ""),
                                "issue_id": issue_id,
                                "test_file": bug_location.get("test_file", ""),
                                "test_function": bug_location.get("test_function", "")
                            }
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.error(f"Error loading bug locations from {path}: {str(e)}")
                continue

        # Fallback: Get basic info from issue data
        logger.warning(f"No bug location file found for {issue_id}, using issue data")
        issue = self.data_loader.load_issue(issue_id)

        if issue:
            failing_code = issue.get("FAIL_TO_PASS", "")
            function_name = self._extract_function_name(failing_code)

            return {
                "file": issue.get("impl_file_path", "unknown_file.py"),
                "function": function_name or issue.get("impl_function_name", "unknown_function"),
                "code_content": failing_code,
                "confidence": 0.5,
                "issue_id": issue_id,
                "bug_type": "unknown",
                "bug_description": "Bug location could not be found in detector output"
            }

        # Error case
        return {
            "error": f"Could not load bug location for issue {issue_id}",
            "file": None,
            "function": None
        }

    def _extract_function_name(self, code: str) -> str:
        """Extract the function name from a code snippet."""
        # Try to extract method name from both function and class definitions
        pattern = r"(class|def)\s+([a-zA-Z0-9_]+)\s*[\(:]"
        matches = re.findall(pattern, code)

        if matches:
            for match_type, name in matches:
                # Return class name if it's a class definition
                if match_type == 'class':
                    return name

            # Otherwise return the first function/method name
            for match_type, name in matches:
                if match_type == 'def':
                    return name

        return ""

    def _calculate_location_specificity(self, bug_location: Dict[str, Any]) -> float:
        """
        Calculate location specificity score based on completeness.

        Args:
            bug_location: Bug location dictionary.

        Returns:
            Specificity score (0-1).
        """
        score = 0.0

        # Check file identification
        if bug_location.get("file"):
            score += 0.3

        # Check function identification
        if bug_location.get("function"):
            score += 0.3

        # Check line number identification
        if bug_location.get("bug_lines") or bug_location.get("line_start"):
            score += 0.2

        # Check issue explanation
        if bug_location.get("bug_description") and len(bug_location.get("bug_description", "")) > 10:
            score += 0.2

        return min(1.0, score)

    def _run_enhanced_solution_phase(
            self,
            issue: Dict[str, Any],
            issue_description: str,
            bug_location: Dict[str, Any],
            initial_patch: Optional[str] = None,
            initial_solution_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the enhanced solution phase with self-reflection and chain of thought.

        This method implements the new strategy:
        1. Start with self-reflection to understand the bug
        2. Run multiple iterations of chain of thought to generate potential fixes
        3. Validate each potential fix and test on the branch
        4. If needed, apply additional self-reflection iterations

        Args:
            issue: Issue dictionary.
            issue_description: Issue description text.
            bug_location: Bug location information.
            initial_patch: Optional patch from previous iteration.
            initial_solution_text: Optional solution text from previous iteration.

        Returns:
            Dictionary with phase results.
        """
        logger.info("Starting Enhanced Solution Development phase with Self-Reflection and Chain of Thought")
        model = self._get_model()

        # Initialize Chain of Thought reasoner
        cot_reasoner = EnhancedChainOfThought(self.config, model)

        # Initialize Self-Reflection
        reflector = SelfReflection(self.config, model)
        reflector.current_issue = issue  # Set current issue for proper context

        # Initialize result tracking
        result = {
            "iterations": [],
            "solution": None,
            "depth": 0.0,
            "early_stopped": False,
            "success": False,
            "previous_solution_used": (initial_patch is not None or initial_solution_text is not None)
        }

        # Format context
        context = self._format_solution_context(
            issue_description,
            bug_location,
            issue
        )

        # If we have initial solution or patch from previous iteration, incorporate it
        if initial_solution_text:
            logger.info("Using solution text from previous iteration")
            context += f"\n\nPREVIOUS SOLUTION ANALYSIS:\n{initial_solution_text[:1000]}...\n"
            result["previous_solution_text_used"] = True

        if initial_patch:
            logger.info("Using patch from previous iteration as starting point")
            context += f"\n\nPREVIOUS PATCH ATTEMPT:\n{initial_patch}\n"
            result["previous_patch_used"] = True

        # Get issue_id for validation
        issue_id = issue.get("instance_id") or issue.get("branch_name")
        if not issue_id:
            issue_id = "unknown"

        # Keep track of test feedback for improving solutions
        test_feedback = []
        current_solution = initial_solution_text
        current_depth = 0.0 if not initial_solution_text else 0.5  # Start with some depth if using previous solution

        # Phase 1: Initial Self-Reflection to understand the bug
        logger.info("Phase 1: Initial Self-Reflection to understand the bug")
        try:
            # Create a focused self-reflection prompt for understanding the bug
            reflection_prompt = self._create_reflection_prompt(
                issue_description,
                bug_location,
                "INITIAL_REFLECTION",  # Signal this is the initial reflection
                context
            )

            # Generate the reflection - using refine_solution since reflect_on_bug doesn't exist
            reflection_result = reflector.refine_solution(
                "",  # No initial solution for first reflection
                issue_description,
                reflection_prompt  # Using the prompt as context
            )

            # Get reflection from the result
            if isinstance(reflection_result, dict):
                # If it returns a structured result (expected)
                bug_understanding = reflection_result.get("reflection", "")
                if not bug_understanding and reflection_result.get("reflections") and reflection_result["reflections"]:
                    # Try to get from the reflections list if available
                    first_reflection = reflection_result["reflections"][0]
                    bug_understanding = first_reflection.get("reflection", "")
            else:
                # Fallback if the return value is not as expected
                bug_understanding = str(reflection_result)

            # Add the reflection to our context for future iterations
            enhanced_context = context + f"\n\nBUG ANALYSIS:\n{bug_understanding}\n"

            # Track the initial reflection
            result["iterations"].append({
                "phase": "initial_reflection",
                "reflection": bug_understanding,
                "depth": 0.5  # Initial reflection has a fixed depth score
            })

        except Exception as e:
            logger.error(f"Error in initial self-reflection: {e}", exc_info=True)
            bug_understanding = "Error in initial reflection."
            enhanced_context = context

        # Phase 2: Chain of Thought iterations to generate potential fixes
        logger.info(f"Phase 2: Chain of Thought iterations ({self.max_cot_iterations} iterations)")
        potential_patches = []

        for i in range(self.max_cot_iterations):
            logger.info(f"Chain of Thought iteration {i + 1}/{self.max_cot_iterations}")

            # Clear memory before generation
            self._run_memory_cleanup()

            # Update the context with any test feedback
            iteration_context = enhanced_context
            if test_feedback:
                feedback_text = "\n\n".join([f"TEST FEEDBACK (Iteration {idx + 1}):\n{feedback}"
                                             for idx, feedback in enumerate(test_feedback)])
                iteration_context += f"\n\nTEST FEEDBACK FROM PREVIOUS ATTEMPTS:\n{feedback_text}\n"

            # Generate a solution with Chain of Thought reasoning
            try:
                solution_data = cot_reasoner.solve_with_validation(
                    issue_id,
                    issue_description,
                    iteration_context,
                    bug_location
                )

                if isinstance(solution_data, dict) and "solution" in solution_data:
                    solution = solution_data["solution"]
                else:
                    solution = solution_data

            except Exception as e:
                logger.error(f"Error generating solution in CoT iteration {i + 1}: {e}", exc_info=True)
                solution = f"Error generating solution in iteration {i + 1}."

            # Extract patch from the solution
            patch = self._extract_patch_from_solution(solution)

            # Calculate solution completeness
            solution_completeness = self._calculate_solution_completeness(solution, patch is not None)

            # Track this iteration
            iteration_result = {
                "phase": f"chain_of_thought_{i + 1}",
                "solution": solution,
                "patch": patch,
                "depth": solution_completeness
            }

            # Validate the patch if we have one
            if patch:
                # Run basic validation
                validation_result = self.patch_validator.validate_patch(patch, issue_id)
                iteration_result["validation"] = validation_result

                # If validation succeeds, test the patch
                if validation_result.get("success", False):
                    test_validation = self.validate_patch_with_tests(patch, issue_id, issue)
                    iteration_result["test_validation"] = test_validation

                    # Record test feedback for future iterations
                    if test_validation.get("test_run", False):
                        test_output = test_validation.get("test_output", "")
                        if test_output:
                            # Limit test output to avoid excessive context
                            if len(test_output) > 2000:
                                test_output = test_output[:1000] + "\n...\n" + test_output[-1000:]
                            test_feedback.append(test_output)

                    # If patch passes tests, we found a valid solution
                    if test_validation.get("success", False):
                        result["solution"] = {
                            "patch": patch,
                            "validation": test_validation,
                            "solution_text": solution,
                            "bug_location": bug_location
                        }
                        result["success"] = True
                        result["early_stopped"] = True
                        logger.info(f"Found valid patch in CoT iteration {i + 1}")

                # Save potential patch for later use
                potential_patches.append({
                    "patch": patch,
                    "solution": solution,
                    "validation": validation_result,
                    "test_validation": iteration_result.get("test_validation"),
                    "depth": solution_completeness
                })

            # Store this iteration
            result["iterations"].append(iteration_result)
            current_solution = solution
            current_depth = max(current_depth, solution_completeness)

            # Early stopping if we found a valid solution
            if result["success"]:
                break

        # If we haven't found a valid solution yet, try additional self-reflection
        if not result["success"] and potential_patches:
            logger.info(f"Phase 3: Additional Self-Reflection iterations ({self.max_reflection_iterations} iterations)")

            # Get the best potential patch so far
            best_patch_info = max(potential_patches, key=lambda x: x.get("depth", 0))
            best_patch = best_patch_info["patch"]
            best_solution = best_patch_info["solution"]

            # Run self-reflection iterations to improve the solution
            for i in range(self.max_reflection_iterations):
                logger.info(f"Self-Reflection iteration {i + 1}/{self.max_reflection_iterations}")

                # Clear memory before generation
                self._run_memory_cleanup()

                # Update context with all test feedback
                reflection_context = enhanced_context
                if test_feedback:
                    feedback_text = "\n\n".join([f"TEST FEEDBACK (Iteration {idx + 1}):\n{feedback}"
                                                 for idx, feedback in enumerate(test_feedback)])
                    reflection_context += f"\n\nTEST FEEDBACK FROM PREVIOUS ATTEMPTS:\n{feedback_text}\n"

                # Run self-reflection to improve the solution
                try:
                    reflection_result = reflector.refine_solution(
                        best_solution,
                        issue_description,
                        reflection_context
                    )

                    reflection = reflection_result.get("reflection", "")
                    improved_solution = reflection_result.get("final_solution", best_solution)

                except Exception as e:
                    logger.error(f"Error in self-reflection iteration {i + 1}: {e}", exc_info=True)
                    reflection = f"Error in self-reflection iteration {i + 1}."
                    improved_solution = best_solution

                # Extract patch from improved solution
                improved_patch = self._extract_patch_from_solution(improved_solution)

                # Calculate solution completeness
                solution_completeness = self._calculate_solution_completeness(improved_solution,
                                                                              improved_patch is not None)

                # Track this iteration
                iteration_result = {
                    "phase": f"self_reflection_{i + 1}",
                    "reflection": reflection,
                    "solution": improved_solution,
                    "patch": improved_patch,
                    "depth": solution_completeness
                }

                # Validate and test the improved patch
                if improved_patch:
                    # Run basic validation
                    validation_result = self.patch_validator.validate_patch(improved_patch, issue_id)
                    iteration_result["validation"] = validation_result

                    # If validation succeeds, test the patch
                    if validation_result.get("success", False):
                        test_validation = self.validate_patch_with_tests(improved_patch, issue_id, issue)
                        iteration_result["test_validation"] = test_validation

                        # Record test feedback for future iterations
                        if test_validation.get("test_run", False):
                            test_output = test_validation.get("test_output", "")
                            if test_output:
                                # Limit test output to avoid excessive context
                                if len(test_output) > 2000:
                                    test_output = test_output[:1000] + "\n...\n" + test_output[-1000:]
                                test_feedback.append(test_output)

                        # If patch passes tests, we found a valid solution
                        if test_validation.get("success", False):
                            result["solution"] = {
                                "patch": improved_patch,
                                "validation": test_validation,
                                "solution_text": improved_solution,
                                "bug_location": bug_location
                            }
                            result["success"] = True
                            logger.info(f"Found valid patch in self-reflection iteration {i + 1}")
                            break

                    # Update best patch if this one is better
                    if solution_completeness > best_patch_info["depth"]:
                        best_patch = improved_patch
                        best_solution = improved_solution
                        best_patch_info = {
                            "patch": improved_patch,
                            "solution": improved_solution,
                            "validation": validation_result,
                            "test_validation": iteration_result.get("test_validation"),
                            "depth": solution_completeness
                        }

                # Store this iteration
                result["iterations"].append(iteration_result)
                current_solution = improved_solution
                current_depth = max(current_depth, solution_completeness)

                # Early stopping if we found a valid solution
                if result["success"]:
                    break

        # If we still don't have a valid solution, use the best attempt
        if not result["solution"] and result["iterations"]:
            # Find the best iteration based on depth score
            try:
                best_iteration = max(result["iterations"], key=lambda x: x.get("depth", 0))
                result["solution"] = {
                    "patch": best_iteration.get("patch"),
                    "validation": best_iteration.get("test_validation", best_iteration.get("validation")),
                    "solution_text": best_iteration.get("solution"),
                    "bug_location": bug_location
                }
            except Exception as e:
                logger.error(f"Error selecting best iteration: {e}")
                # Use the last iteration as fallback
                if result["iterations"]:
                    last_iteration = result["iterations"][-1]
                    result["solution"] = {
                        "patch": last_iteration.get("patch"),
                        "validation": last_iteration.get("test_validation", last_iteration.get("validation")),
                        "solution_text": last_iteration.get("solution"),
                        "bug_location": bug_location
                    }

        # Set the final depth score
        result["depth"] = current_depth

        return result

    def _create_reflection_prompt(
            self,
            issue_description: str,
            bug_location: Dict[str, Any],
            phase: str,
            context: str
    ) -> str:
        """
        Create a prompt for self-reflection based on the current phase.

        Args:
            issue_description: Issue description text.
            bug_location: Bug location information.
            phase: Current phase of the process ('INITIAL_REFLECTION' or 'SOLUTION_REFINEMENT').
            context: Additional context for the reflection.

        Returns:
            Formatted prompt string.
        """
        if phase == "INITIAL_REFLECTION":
            prompt = f"""
            You are an expert software engineer analyzing a bug in code. First, understand the bug thoroughly
            before attempting to fix it.

            ISSUE DESCRIPTION:
            {issue_description}

            BUG LOCATION:
            File: {bug_location.get('file', 'Unknown')}
            Function: {bug_location.get('function', 'Unknown')}
            Line Range: {bug_location.get('line_start')} to {bug_location.get('line_end')}
            Bug Type: {bug_location.get('bug_type', 'Unknown')}
            Bug Description: {bug_location.get('bug_description', 'Unknown')}

            CODE WITH BUG:
            ```python
            {bug_location.get('code_content', '')}
            ```

            ADDITIONAL CONTEXT:
            {context}

            YOUR TASK:
            1. Carefully analyze the code and the bug description
            2. Identify exactly what is causing the bug
            3. Explain why the bug is occurring (logical reasoning)
            4. Describe how the bug affects the program's behavior
            5. Discuss what a correct implementation should do instead

            Provide your DETAILED analysis before attempting any solution.
            """
        else:  # SOLUTION_REFINEMENT or any other phase
            prompt = f"""
            You are an expert software engineer refining a solution to a bug. Review the current solution
            and improve it based on your analysis and any test feedback.

            ISSUE DESCRIPTION:
            {issue_description}

            BUG LOCATION:
            File: {bug_location.get('file', 'Unknown')}
            Function: {bug_location.get('function', 'Unknown')}
            Line Range: {bug_location.get('line_start')} to {bug_location.get('line_end')}
            Bug Type: {bug_location.get('bug_type', 'Unknown')}
            Bug Description: {bug_location.get('bug_description', 'Unknown')}

            CODE WITH BUG:
            ```python
            {bug_location.get('code_content', '')}
            ```

            ADDITIONAL CONTEXT:
            {context}

            YOUR TASK:
            1. Critically review the current solution and identify any issues or areas for improvement
            2. Consider whether the solution correctly addresses the root cause of the bug
            3. Pay special attention to any test feedback
            4. Propose specific improvements or alternatives
            5. Generate an improved patch that will pass validation and tests

            Provide your detailed analysis followed by an improved solution.
            """

        return prompt

    def _format_solution_context(
            self,
            issue_description: str,
            bug_location: Dict[str, Any],
            issue: Dict[str, Any]
    ) -> str:
        """
        Format context for solution phase.

        Args:
            issue_description: Issue description.
            bug_location: Bug location information.
            issue: The full issue data.

        Returns:
            Formatted context string.
        """
        # Format code content safely
        code_content = bug_location.get("code_content", "")
        if not code_content and bug_location.get("file"):
            # Try to get code from issue
            code_content = issue.get("FAIL_TO_PASS", "")

        # Format bug line information
        bug_lines = bug_location.get("bug_lines", [])
        if not bug_lines:
            if bug_location.get("line_start") and bug_location.get("line_end"):
                bug_lines = list(range(bug_location.get("line_start"), bug_location.get("line_end") + 1))

        # Get the exact start and end lines for proper patch context
        line_start = bug_location.get("line_start")
        line_end = bug_location.get("line_end")

        # Create a clear indication of the bug lines
        bug_lines_str = ""
        if bug_lines:
            bug_lines_str = f"Bug Lines: {', '.join(map(str, bug_lines))}\n"
        elif line_start and line_end:
            bug_lines_str = f"Bug Line Range: {line_start} to {line_end}\n"

        context = f"""
        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        {bug_lines_str}
        Bug Type: {bug_location.get('bug_type', 'Unknown')}
        Bug Description: {bug_location.get('bug_description', 'Unknown')}

        CODE CONTENT:
        ```python
        {code_content}
        ```
        """

        # Add failing test if available
        test_file = bug_location.get("test_file", "")
        test_function = bug_location.get("test_function", "")
        if test_file and test_function:
            context += f"\n\nFAILING TEST:\nTest File: {test_file}\nTest Function: {test_function}\n"

        # Add failing test code if available
        failing_test_code = issue.get("FAIL_TO_PASS", "")
        if failing_test_code:
            context += f"\n\nFAILING TEST CODE:\n```python\n{failing_test_code}\n```\n"

        # Add passing test if available
        passing_test_code = issue.get("PASS_TO_PASS", "")
        if passing_test_code:
            context += f"\n\nPASSING TEST CODE:\n```python\n{passing_test_code}\n```\n"

        return context

    def validate_patch_with_tests(
            self,
            patch: str,
            issue_id: str,
            issue: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a patch by applying it and running the unit tests.

        Args:
            patch: The patch to validate.
            issue_id: The issue ID.
            issue: The issue data.

        Returns:
            Dictionary with validation results.
        """
        logger.info(f"Validating patch with unit tests for issue {issue_id}")

        # First, validate patch syntax
        basic_validation = self.patch_validator.validate_patch(patch, issue_id)
        if not basic_validation.get("success", False):
            logger.warning(f"Basic patch validation failed: {basic_validation.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": basic_validation.get("error", "Patch validation failed"),
                "test_run": False,
                "test_output": "",
                "basic_validation": basic_validation
            }

        # Get Python environment path
        python_env_path = self.data_loader.get_python_env_path(issue)
        if not python_env_path:
            logger.warning("Python environment path not found in issue data")
            return {
                "success": False,
                "error": "Python environment path not found",
                "test_run": False,
                "test_output": "",
                "basic_validation": basic_validation
            }

        # Extract test file path
        test_file = self.data_loader.find_failing_test(issue)
        if not test_file:
            logger.warning("Failed to determine test file path")
            return {
                "success": False,
                "error": "Test file path not found",
                "test_run": False,
                "test_output": "",
                "basic_validation": basic_validation
            }

        # Get the repository path
        repo_path = Path("/storage/homefs/jp22b083/SSI/S-R-1/data/repositories/astropy/astropy")
        if not repo_path:
            logger.warning("Repository path not found in issue data")
            return {
                "success": False,
                "error": "Repository path not found",
                "test_run": False,
                "test_output": "",
                "basic_validation": basic_validation
            }

        # Apply the patch to the repository
        try:
            # First, make sure we're on the right branch
            success = self.data_loader.checkout_issue_branch(issue)
            if not success:
                logger.warning("Failed to checkout issue branch")
                return {
                    "success": False,
                    "error": "Failed to checkout issue branch",
                    "test_run": False,
                    "test_output": "",
                    "basic_validation": basic_validation
                }

            # Apply the patch using git apply
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_patch_file:
                temp_patch_file.write(patch)
                temp_patch_path = temp_patch_file.name

            # Apply the patch
            apply_result = subprocess.run(
                ["git", "apply", "--check", temp_patch_path],
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if apply_result.returncode != 0:
                logger.warning(f"Patch check failed: {apply_result.stderr}")
                return {
                    "success": False,
                    "error": f"Patch cannot be applied: {apply_result.stderr}",
                    "test_run": False,
                    "test_output": "",
                    "basic_validation": basic_validation
                }

            # Actually apply the patch
            apply_result = subprocess.run(
                ["git", "apply", temp_patch_path],
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if apply_result.returncode != 0:
                logger.warning(f"Patch application failed: {apply_result.stderr}")
                return {
                    "success": False,
                    "error": f"Failed to apply patch: {apply_result.stderr}",
                    "test_run": False,
                    "test_output": "",
                    "basic_validation": basic_validation
                }

            # Run the tests
            logger.info(f"Running test: {test_file}")
            test_cmd = [python_env_path, "-m", "pytest", test_file, "-v"]

            try:
                test_result = subprocess.run(
                    test_cmd,
                    cwd=repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.test_timeout
                )

                # Determine if the test passed
                test_passed = test_result.returncode == 0

                # Get the output
                test_output = test_result.stdout + "\n" + test_result.stderr

                # Revert the changes to keep the repository clean
                revert_result = subprocess.run(
                    ["git", "checkout", "."],
                    cwd=repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if revert_result.returncode != 0:
                    logger.warning(f"Failed to revert changes: {revert_result.stderr}")

                # Clean up temp file
                os.unlink(temp_patch_path)

                # Return the result
                return {
                    "success": test_passed,
                    "test_run": True,
                    "test_output": test_output,
                    "basic_validation": basic_validation
                }

            except subprocess.TimeoutExpired:
                logger.warning(f"Test execution timed out after {self.test_timeout} seconds")

                # Revert the changes to keep the repository clean
                revert_result = subprocess.run(
                    ["git", "checkout", "."],
                    cwd=repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Clean up temp file
                os.unlink(temp_patch_path)

                return {
                    "success": False,
                    "error": f"Test execution timed out after {self.test_timeout} seconds",
                    "test_run": True,
                    "test_output": "Test timed out",
                    "basic_validation": basic_validation
                }

        except Exception as e:
            logger.error(f"Error validating patch with tests: {str(e)}", exc_info=True)

            # Try to revert changes if possible
            try:
                revert_result = subprocess.run(
                    ["git", "checkout", "."],
                    cwd=repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            except:
                pass

            # Clean up temp file if it exists
            try:
                os.unlink(temp_patch_path)
            except:
                pass

            return {
                "success": False,
                "error": f"Error validating patch: {str(e)}",
                "test_run": False,
                "test_output": "",
                "basic_validation": basic_validation
            }

    def _extract_patch_from_solution(self, solution: str) -> Optional[str]:
        """
        Extract a Git patch from the solution.

        Args:
            solution: Solution text.

        Returns:
            Extracted patch or None.
        """
        # Look for Git diff format
        diff_pattern = r'(diff --git.*?)(?=^```|\Z)'
        diff_match = re.search(diff_pattern, solution, re.MULTILINE | re.DOTALL)

        if diff_match:
            return diff_match.group(1).strip()

        # Look for code blocks
        code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
        code_match = re.search(code_block_pattern, solution, re.MULTILINE | re.DOTALL)

        if code_match:
            content = code_match.group(1).strip()
            if content.startswith('diff --git') or ('---' in content and '+++' in content):
                return content

        # Check if solution itself is a patch
        if solution.strip().startswith('diff --git') or ('---' in solution and '+++' in solution):
            return solution.strip()

        return None

    def _calculate_solution_completeness(self, solution: str, has_patch: bool) -> float:
        """
        Calculate solution completeness score.

        Args:
            solution: Solution text.
            has_patch: Whether a patch was extracted.

        Returns:
            Completeness score (0-1).
        """
        score = 0.5  # Start with base score for having a solution

        # Check for implementation details
        if '```' in solution:
            score += 0.2  # Contains code blocks

        # Check for patch format
        if has_patch:
            score += 0.2  # Contains a patch
        elif 'diff --git' in solution or ('---' in solution and '+++' in solution):
            score += 0.15  # Contains patch-like content

        # Check for explanation quality
        if len(solution) > 200 and solution.count('\n') > 5:
            score += 0.1  # Substantial solution with explanation

        return min(1.0, score)

    def _calculate_combined_depth(self, depth_scores: Dict[str, float]) -> float:
        """
        Calculate combined depth score.

        Args:
            depth_scores: Dictionary with individual depth scores.

        Returns:
            Combined depth score (0-1).
        """
        weights = {
            "location_specificity": 0.5,
            "solution_completeness": 0.5
        }

        combined = 0.0
        weight_sum = 0.0

        for metric, weight in weights.items():
            if metric in depth_scores:
                combined += depth_scores[metric] * weight
                weight_sum += weight

        if weight_sum == 0:
            return 0.0

        return combined / weight_sum

    def _finalize_result(self, result, issue, start_time):
        """
        Finalize the result by adding comprehensive information from the issue.

        Args:
            result: The current result dictionary
            issue: The issue dictionary
            start_time: Time when processing started

        Returns:
            Complete result dictionary with all relevant information
        """
        # Calculate processing time
        processing_time = time.time() - start_time

        # Get ground truth solution
        ground_truth = self.data_loader.get_solution_patch(issue)

        # Get hints information
        hints = self.data_loader.get_hints(issue)

        # Get failing and passing tests
        fail_to_pass = self.data_loader.get_failed_code(issue)
        pass_to_pass = self.data_loader.get_passing_code(issue)

        # Get issue description
        issue_description = self.data_loader.get_issue_description(issue)

        # Create complete output
        complete_result = {
            # Basic identification
            "issue_id": issue.get("instance_id") or issue.get("branch_name"),
            "repo": issue.get("repo", ""),
            "issue_number": issue.get("issue_number") or issue.get("number"),

            # Summary information
            "summary": {
                "total_iterations": result.get("total_iterations", 1),
                "success": result.get("success", False),
                "early_stopped": result.get("early_stopped", False),
                "best_solution_iteration": result.get("best_solution", {}).get("iteration", 0) if result.get(
                    "best_solution") else 0,
                "processing_time": processing_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },

            # Depth scores in one place
            "depth_scores": result.get("depth_scores", {}),

            # Issue information
            "issue_info": {
                "description": issue_description[:500] + "..." if len(issue_description) > 500 else issue_description,
                "ground_truth_patch_available": bool(ground_truth),
                "hints_available": hints is not None,
                "fail_to_pass_tests": fail_to_pass,
                "pass_to_pass_tests": pass_to_pass,
            },

            # Final solution
            "final_solution": self._extract_final_solution(result)
        }

        # Add error if present
        if result.get("error"):
            complete_result["error"] = result.get("error")

        return complete_result

    def _extract_final_solution(self, result):
        """
        Extract the final solution details.

        Args:
            result: Full result dictionary

        Returns:
            Final solution information
        """
        if not result.get("solution"):
            return {
                "patch": "",
                "success": False
            }

        solution = result.get("solution", {})
        if isinstance(solution, dict):
            return {
                "patch": solution.get("patch", ""),
                "validation": solution.get("validation", {}),
                "success": solution.get("validation", {}).get("success", False),
                "bug_location": solution.get("bug_location", {})
            }

        return {
            "patch": "",
            "success": False
        }
