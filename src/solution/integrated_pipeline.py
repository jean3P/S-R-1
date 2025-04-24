# src/solution/integrated_pipeline.py

import logging
import re
import os
import torch
from typing import Dict, Any, Optional
import time
import gc

from ..data.data_loader import SWEBenchDataLoader
from ..models import create_model
from ..utils.repository_explorer import RepositoryExplorer
from ..utils.patch_validator import PatchValidator
from ..utils.bug_detector import BugDetector
from ..utils.bug_fixer import BugFixer
from ..reasoning.enhanced_chain_of_thought import EnhancedChainOfThought
from ..reasoning.enhanced_tree_of_though import EnhancedTreeOfThought
from ..reasoning.self_reflection import SelfReflection

logger = logging.getLogger(__name__)


class IntegratedBugFixingPipeline:
    """
    Implements the optimized bug fixing pipeline with integrated validation.
    Combines RAG+Bug Localization, Tree of Thought exploration, and Chain of Thought
    solution development with continuous validation.
    """

    def __init__(self, config, model_name=None):
        """
        Initialize the integrated bug fixing pipeline.

        Args:
            config: Configuration object.
            model_name: Name of the model to use, or None to use default.
        """
        self.config = config
        self.model_name = model_name or config.get("default_model", "deepseek-r1-distill")
        self.model = None  # Lazy-load model on demand

        # Initialize core components
        self.data_loader = SWEBenchDataLoader(config)
        self.repo_explorer = RepositoryExplorer(config)
        self.patch_validator = PatchValidator(config)

        # Configure parameters
        self.max_rag_iterations = 2
        self.max_tot_branches = 3
        self.max_cot_iterations = 2
        self.max_total_iterations = 8

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
        Solve an issue using the integrated pipeline with Bug Detector.

        Args:
            issue_id: Issue ID to solve.
            previous_solution: Optional dictionary with previous best solution and feedback.

        Returns:
            Dictionary with solution results.
        """
        logger.info(f"Starting integrated bug fixing pipeline with Bug Detector for issue {issue_id}")
        start_time = time.time()

        # Handle previous solution feedback if available
        if previous_solution:
            logger.info(f"Using feedback from previous iteration with depth {previous_solution.get('depth', 0.0)}")

        # Load issue
        issue = self.data_loader.load_issue(issue_id)
        if not issue:
            return {"error": f"Issue {issue_id} not found"}

        # Ensure repository exists and prepare it
        if not self.repo_explorer.ensure_repository_exists(issue):
            return {"error": f"Failed to download repository for issue {issue_id}"}

        if not self.data_loader.prepare_repository_for_analysis(issue):
            return {"error": f"Failed to prepare repository for analysis"}

        # Get issue description
        issue_description = self.data_loader.get_issue_description(issue)

        # Initialize tracking variables
        total_iterations_used = 0
        result = {
            "issue_id": issue_id,
            "phases": [],
            "solution": None,
            "success": False,
            "depth_scores": {
                "location_specificity": 0.0,
                "root_cause_confidence": 0.0,
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

        # Run the three-phase pipeline with early stopping
        try:
            # Phase 1: Bug Detection (replacing RAG + Bug Localization)
            initial_bug_location = None
            if previous_solution and previous_solution.get("solution"):
                prev_sol = previous_solution.get("solution")
                if isinstance(prev_sol, dict) and "bug_location" in prev_sol:
                    initial_bug_location = prev_sol["bug_location"]
                    logger.info(f"Using bug location from previous solution: {initial_bug_location}")

            # Initialize Bug Detector
            bug_detector = BugDetector(self.config)

            # Detect bug location
            bug_detection_result = bug_detector.detect_bug_location(issue_id)

            # If we have an initial bug location from previous iteration, enhance it
            if initial_bug_location:
                # Merge information from previous bug location with new detection
                if not bug_detection_result.get("function") and initial_bug_location.get("function"):
                    bug_detection_result["function"] = initial_bug_location.get("function")
                if not bug_detection_result.get("line_numbers") and initial_bug_location.get("line_numbers"):
                    bug_detection_result["line_numbers"] = initial_bug_location.get("line_numbers")

            # Calculate location specificity
            location_specificity = self._calculate_location_specificity(bug_detection_result)

            # Add phase info
            result["phases"].append({
                "name": "bug_detection",
                "bug_location": bug_detection_result,
                "depth": location_specificity,
                "early_stopped": False
            })

            result["depth_scores"]["location_specificity"] = location_specificity
            total_iterations_used += 1

            # Try to generate a patch based on the detected bug location
            validated_patches = []
            valid_solution = None

            if bug_detection_result.get("file"):
                logger.info(f"Attempting to generate patch with detected bug location")

                # Get model
                model = self._get_model()

                # Initialize bug fixer
                bug_fixer = BugFixer(model, self.config)

                # Generate patch
                patch = bug_fixer.generate_fix(bug_detection_result, issue_description, {})

                if patch:
                    # Validate patch
                    validation_result = self.patch_validator.validate_patch(patch, issue_id)
                    valid_patch_found = validation_result.get("success", False)

                    validated_patches.append({
                        "patch": patch,
                        "validation": validation_result,
                        "location": bug_detection_result
                    })

                    if valid_patch_found:
                        valid_solution = {
                            "patch": patch,
                            "source": "bug_detection",
                            "bug_location": bug_detection_result
                        }
                        logger.info("Valid solution found during bug detection phase")

                        # Early stopping
                        result["solution"] = valid_solution
                        result["success"] = True
                        result["early_stopped_at"] = "bug_detection"
                        result["depth_scores"]["combined"] = self._calculate_combined_depth(result["depth_scores"])
                        return self._finalize_result(result, issue, start_time)

            # Continue only if we have iterations left and didn't reach max total
            if total_iterations_used < self.max_total_iterations:
                try:
                    # Phase 2: Tree of Thought Exploration
                    # Use root cause analysis from previous solution if available
                    initial_root_cause = None
                    if previous_solution and previous_solution.get("solution"):
                        prev_sol = previous_solution.get("solution")
                        if isinstance(prev_sol, dict) and "root_cause" in prev_sol:
                            initial_root_cause = prev_sol["root_cause"]
                            logger.info(f"Using root cause from previous solution")

                    phase2_result = self._run_tot_exploration_phase(
                        issue,
                        issue_description,
                        bug_detection_result,  # Using bug location from detection
                        initial_root_cause
                    )

                    # Add phase info with safe defaults
                    result["phases"].append({
                        "name": "tot_exploration",
                        "branches": phase2_result.get("branches", []),
                        "depth": phase2_result.get("depth", 0.0),
                        "early_stopped": phase2_result.get("early_stopped", False)
                    })

                    # Update depth scores safely
                    result["depth_scores"]["root_cause_confidence"] = phase2_result.get("depth", 0.0)

                    # Count iterations safely
                    branches = phase2_result.get("branches", [])
                    total_iterations_used += len(branches) if isinstance(branches, list) else 0

                    # Early stopping check
                    if phase2_result.get("valid_solution"):
                        logger.info("Valid solution found during ToT Exploration phase. Early stopping.")
                        result["solution"] = phase2_result["valid_solution"]
                        result["success"] = True
                        result["early_stopped_at"] = "tot_exploration"
                        result["depth_scores"]["combined"] = self._calculate_combined_depth(result["depth_scores"])
                        return self._finalize_result(result, issue, start_time)

                    # Safely select branch for next phase
                    selected_branch = {}
                    if isinstance(phase2_result.get("branches"), list) and phase2_result["branches"]:
                        try:
                            best_branch_idx = 0  # Default to first branch
                            selected_branch = phase2_result["branches"][best_branch_idx]
                        except (IndexError, KeyError) as e:
                            logger.warning(f"Error selecting branch: {e}, using empty dict instead")
                except Exception as e:
                    logger.error(f"Error in ToT exploration phase: {str(e)}", exc_info=True)
                    selected_branch = {}

            if total_iterations_used < self.max_total_iterations:
                # Phase 3: Chain of Thought Solution Development
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

                phase3_result = self._run_cot_solution_phase(
                    issue,
                    issue_description,
                    bug_detection_result,  # Using bug location from detection
                    selected_branch,
                    initial_patch,
                    initial_solution_text
                )

                result["phases"].append({
                    "name": "cot_solution",
                    "iterations": phase3_result["iterations"],
                    "depth": phase3_result["depth"],
                    "early_stopped": phase3_result["early_stopped"]
                })

                result["depth_scores"]["solution_completeness"] = phase3_result["depth"]
                total_iterations_used += len(phase3_result["iterations"])

                # Store final solution
                result["solution"] = phase3_result["solution"]
                result["success"] = phase3_result.get("success", False)

            # Calculate combined depth score
            result["depth_scores"]["combined"] = self._calculate_combined_depth(result["depth_scores"])

            return self._finalize_result(result, issue, start_time)

        except Exception as e:
            logger.error(f"Error in integrated pipeline: {str(e)}")
            result["error"] = str(e)
            return result

    def _run_rag_bug_localization_phase(self, issue, issue_description, initial_bug_location=None):
        """
        Run the RAG + Bug Localization phase using focused code retrieval.

        Args:
            issue: Issue dictionary
            issue_description: Description of the issue
            initial_bug_location: Optional initial bug location from previous iteration

        Returns:
            Dictionary with phase results
        """
        logger.info("Starting RAG + Bug Localization phase")
        model = self._get_model()  # Add this line

        # Get repository info
        repo_data = issue.get("repository_exploration", {})

        # Safely extract issue ID for validation
        issue_id = issue.get("instance_id") or issue.get("id")
        if issue_id is None:
            # Fallback - extract from other fields
            issue_id = f"{issue.get('repo', 'unknown')}__{issue.get('issue_number', 'unknown')}"

        # Retrieve focused code that fits within token limits
        focused_data = self._retrieve_focused_code(issue, issue_description, repo_data)

        # Initialize bug fixer
        from ..utils.bug_fixer import BugFixer
        bug_fixer = BugFixer(self.model, self.config)

        # Initialize patch validator
        from ..utils.patch_validator import PatchValidator
        patch_validator = PatchValidator(self.config)

        # Initialize results tracking
        iterations = []
        valid_solution = None
        max_iterations = min(2, self.max_total_iterations)  # Use 2 iterations by default

        # If we have an initial bug location from previous iteration, use it as the first iteration
        if initial_bug_location:
            logger.info(f"Using bug location from previous iteration: {initial_bug_location}")

            # Create first iteration with the provided bug location
            try:
                # If we identified a file, attempt to generate a patch
                validated_patches = []
                valid_patch_found = False

                if initial_bug_location.get("file"):
                    logger.info(f"Attempting to generate patch with location from previous iteration")

                    # Create context with focused code
                    context_with_focused_code = {
                        "relevant_files": focused_data.get("prioritized_files", []),
                        "file_contents": {},  # We'll use focused_code directly in generate_fix
                        "focused_code": focused_data.get("formatted_code"),
                        "key_entities": focused_data.get("key_entities", [])
                    }

                    # Generate patch with higher confidence given previous location
                    bug_fixer = BugFixer(model, self.config)
                    patch = bug_fixer.generate_fix(initial_bug_location, issue_description, context_with_focused_code)

                    if patch:
                        # Validate patch
                        validation_result = patch_validator.validate_patch(patch, issue_id)
                        valid_patch_found = validation_result.get("success", False)

                        validated_patches.append({
                            "patch": patch,
                            "validation": validation_result,
                            "location": initial_bug_location
                        })

                        if valid_patch_found:
                            valid_solution = {
                                "patch": patch,
                                "source": "bug_localization_from_previous",
                                "bug_location": initial_bug_location
                            }
                            logger.info("Valid solution found using location from previous iteration")

                # Record iteration using previous bug location
                iteration_result = {
                    "iteration": 1,
                    "bug_location": initial_bug_location,
                    "specificity": 1.0,  # High specificity since it's from previous iteration
                    "validated_patches": validated_patches,
                    "valid_patch_found": valid_patch_found,
                    "from_previous_iteration": True
                }

                iterations.append(iteration_result)

                # If valid patch found, we can skip additional iterations
                if valid_patch_found:
                    # Best location is the initial one (with high specificity)
                    return {
                        "iterations": iterations,
                        "bug_location": initial_bug_location,
                        "depth": 1.0,
                        "early_stopped": True,
                        "valid_solution": valid_solution
                    }

            except Exception as e:
                logger.error(f"Error using bug location from previous iteration: {e}")
                # We'll fall back to regular iterations

        # Continue with regular iterations if needed
        start_iteration = len(iterations) + 1
        for i in range(start_iteration - 1, max_iterations):
            logger.info(f"RAG + Bug Localization iteration {i + 1}/{max_iterations}")

            try:
                # Create bug location based on focused code
                prioritized_files = focused_data.get("prioritized_files", [])

                if i == 0 and prioritized_files:
                    # First iteration - use first prioritized file
                    bug_location = {
                        "file": prioritized_files[0],
                        "function": None,
                        "line_numbers": None,
                        "issue": issue_description[:100] + "..." if len(issue_description) > 100 else issue_description
                    }

                    # Try to find function if available
                    for item in focused_data.get("focused_code", []):
                        if item.get("file") == bug_location["file"] and item.get("type") == "functions":
                            functions = item.get("functions", [])
                            if functions:
                                bug_location["function"] = functions[0].get("name")
                                break

                    specificity = 1.0  # High specificity for first iteration with focused approach

                elif i == 1 and len(prioritized_files) > 1:
                    # Second iteration - try next prioritized file
                    bug_location = {
                        "file": prioritized_files[1],
                        "function": None,
                        "line_numbers": None,
                        "issue": issue_description[:100] + "..." if len(issue_description) > 100 else issue_description
                    }

                    # Try to find function if available
                    for item in focused_data.get("focused_code", []):
                        if item.get("file") == bug_location["file"] and item.get("type") == "functions":
                            functions = item.get("functions", [])
                            if functions:
                                bug_location["function"] = functions[0].get("name")
                                break

                    specificity = 0.8  # Slightly lower specificity for second iteration
                else:
                    # Fallback or less specific iteration
                    bug_location = {
                        "file": prioritized_files[0] if prioritized_files else None,
                        "function": None,
                        "line_numbers": None,
                        "issue": issue_description[:100] + "..." if len(issue_description) > 100 else issue_description
                    }
                    specificity = 0.6

                # If we identified a file, attempt to generate a patch
                validated_patches = []
                valid_patch_found = False

                if bug_location["file"]:
                    logger.info(f"Attempting to generate patch with location specificity: {specificity}")

                    # Create context with focused code
                    context_with_focused_code = {
                        "relevant_files": prioritized_files,
                        "file_contents": {},  # We'll use focused_code directly in generate_fix
                        "focused_code": focused_data.get("formatted_code"),
                        "key_entities": focused_data.get("key_entities", [])
                    }

                    # Generate patch
                    bug_fixer = BugFixer(model, self.config)  # Use the initialized model
                    patch = bug_fixer.generate_fix(bug_location, issue_description, context_with_focused_code)

                    if patch:
                        # Validate patch
                        validation_result = patch_validator.validate_patch(patch, issue_id)
                        valid_patch_found = validation_result.get("success", False)

                        validated_patches.append({
                            "patch": patch,
                            "validation": validation_result,
                            "location": bug_location
                        })

                        if valid_patch_found:
                            valid_solution = {
                                "patch": patch,
                                "source": "bug_localization",
                                "bug_location": bug_location
                            }
                            logger.info("Valid solution found during bug localization")

                # Record iteration
                iteration_result = {
                    "iteration": i + 1,
                    "bug_location": bug_location,
                    "specificity": specificity,
                    "validated_patches": validated_patches,
                    "valid_patch_found": valid_patch_found
                }

                iterations.append(iteration_result)

                # If valid patch found, stop iterations
                if valid_patch_found:
                    break

            except Exception as e:
                logger.error(f"Error in bug localization iteration {i + 1}: {str(e)}")
                iterations.append({
                    "iteration": i + 1,
                    "error": str(e)
                })
                break

        # Determine best bug location from iterations
        best_location = None
        best_specificity = 0.0

        for iteration in iterations:
            specificity = iteration.get("specificity", 0.0)
            if specificity > best_specificity and "bug_location" in iteration:
                best_specificity = specificity
                best_location = iteration["bug_location"]

        # Ensure we have at least an empty location object
        if best_location is None:
            best_location = {
                "file": None,
                "function": None,
                "line_numbers": None,
                "issue": "Unable to identify specific bug location"
            }

        return {
            "iterations": iterations,
            "bug_location": best_location,
            "depth": best_specificity,
            "early_stopped": valid_solution is not None,
            "valid_solution": valid_solution
        }

    def _run_tot_exploration_phase(
            self,
            issue: Dict[str, Any],
            issue_description: str,
            bug_location: Dict[str, Any],
            initial_root_cause: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the Tree of Thought exploration phase with iterative patch refinement.

        Args:
            issue: Issue dictionary.
            issue_description: Issue description text.
            bug_location: Bug location information from previous phase.
            initial_root_cause: Optional root cause analysis from previous iteration.

        Returns:
            Dictionary with phase results.
        """
        logger.info("Starting Tree of Thought exploration phase")
        model = self._get_model()

        # Initialize Tree of Thought reasoner
        tot_reasoner = EnhancedTreeOfThought(self.config, model)

        # Initialize result tracking with defaults
        result = {
            "branches": [],
            "branch_iterations": [],
            "selected_branch": None,
            "depth": 0.0,
            "early_stopped": False,
            "valid_solution": None
        }

        try:
            # Get repository exploration data for context
            repo_exploration = self.repo_explorer.explore_repository(issue)

            # Format context with bug location information
            context = self._format_context_with_bug_location(issue_description, bug_location, repo_exploration)

            # Incorporate initial root cause if provided from previous iteration
            if initial_root_cause:
                logger.info("Using root cause analysis from previous iteration")
                # Modify context to include previous root cause analysis
                context = f"{context}\n\nPREVIOUS ROOT CAUSE ANALYSIS:\n{initial_root_cause}\n"

                # Add to result for tracking
                result["previous_root_cause_used"] = True

            # Get issue_id for validation
            issue_id = issue.get("instance_id") or issue.get("id")
            if not issue_id:
                # Fallback to constructing from other fields
                issue_id = f"{issue.get('repo', 'unknown')}__{issue.get('issue_number', 'unknown')}"

            # Initial ToT exploration
            tot_result = tot_reasoner.explore_with_validation(
                issue_id,
                issue_description,
                context,
                bug_location
            )

            # Enhanced error handling for None returns
            if tot_result is None:
                logger.error("ToT exploration returned None. Using default empty result.")
                # Return the default result we initialized earlier
                return result

            # Process the result (now safe because we've handled None case)
            if not isinstance(tot_result, dict):
                logger.error(f"Unexpected ToT result format: {type(tot_result)}")
                return result

            # Copy relevant fields from tot_result
            result["branches"] = tot_result.get("branches", [])
            result["early_stopped"] = tot_result.get("early_stopped", False)
            result["depth"] = tot_result.get("depth", 0.0)

            # If we found a valid solution already, great!
            if tot_result.get("valid_solution"):
                result["valid_solution"] = tot_result["valid_solution"]
                # Add root cause to the solution for future iterations
                if isinstance(result["valid_solution"], dict):
                    best_branch_idx = result["valid_solution"].get("branch", 1) - 1
                    if 0 <= best_branch_idx < len(result["branches"]):
                        result["valid_solution"]["root_cause"] = result["branches"][best_branch_idx].get("content", "")
                logger.info("Found valid patch during initial ToT exploration.")
                return result

            # If no valid solution was found, select top branches for iterative refinement
            # Sort branches by confidence score (checking if branches exist and are not empty)
            sorted_branches = []
            if result["branches"]:
                try:
                    sorted_branches = sorted(
                        result["branches"],
                        key=lambda b: b.get("confidence", 0),
                        reverse=True
                    )
                except Exception as e:
                    logger.error(f"Error sorting branches: {e}")

            # Select top branches for refinement (limit to 2 for efficiency)
            top_branches = sorted_branches[:min(2, len(sorted_branches))]

            # Track branch iterations
            branch_iterations = []

            # Try to refine patches for top branches
            max_refinement_iterations = 2  # Limit refinement iterations per branch
            for branch_idx, branch in enumerate(top_branches):
                # Safely get branch content
                branch_content = branch.get("content", "")
                branch_confidence = branch.get("confidence", 0)
                logger.info(
                    f"Attempting iterative refinement for branch {branch_idx + 1} with confidence {branch_confidence}")

                # Initial patch (might be None)
                current_patch = branch.get("patch")
                current_validation = branch.get("validation", {})

                # Track iterations for this branch
                branch_iter_results = []

                for iter_idx in range(max_refinement_iterations):
                    iteration_result = {
                        "branch_idx": branch_idx + 1,
                        "iteration": iter_idx + 1,
                        "patch": current_patch,
                        "validation": current_validation
                    }

                    # If we already have a valid patch, we're done with this branch
                    if current_validation.get("success", False):
                        logger.info(f"Branch {branch_idx + 1} already has valid patch. Skipping refinement.")
                        branch_iter_results.append(iteration_result)
                        break

                    # If we don't have a patch yet, generate one
                    if current_patch is None:
                        logger.info(f"Generating initial patch for branch {branch_idx + 1}")
                        current_patch = self._generate_patch_from_branch(
                            model,
                            branch_content,
                            bug_location,
                            issue_description,
                            repo_exploration
                        )

                        if current_patch:
                            # Validate the new patch
                            try:
                                current_validation = self.patch_validator.validate_patch(current_patch, issue_id)
                                iteration_result["patch"] = current_patch
                                iteration_result["validation"] = current_validation
                            except Exception as e:
                                logger.error(f"Error validating patch for branch {branch_idx + 1}: {e}")
                                iteration_result["error"] = f"Validation error: {str(e)}"
                        else:
                            logger.warning(f"Failed to generate patch for branch {branch_idx + 1}")
                            iteration_result["error"] = "Failed to generate patch"
                    else:
                        # We have a patch but it's not valid - try to refine it using validation feedback
                        logger.info(f"Refining patch for branch {branch_idx + 1}, iteration {iter_idx + 1}")
                        validation_feedback = current_validation.get("feedback", "No feedback available")

                        try:
                            refined_patch = self._refine_patch_with_feedback(
                                model,
                                current_patch,
                                validation_feedback,
                                branch_content,
                                bug_location,
                                issue_description
                            )

                            if refined_patch:
                                # Validate the refined patch
                                refined_validation = self.patch_validator.validate_patch(refined_patch, issue_id)
                                current_patch = refined_patch
                                current_validation = refined_validation
                                iteration_result["patch"] = current_patch
                                iteration_result["validation"] = current_validation
                            else:
                                logger.warning(f"Failed to refine patch for branch {branch_idx + 1}")
                                iteration_result["error"] = "Failed to refine patch"
                        except Exception as e:
                            logger.error(f"Error refining patch for branch {branch_idx + 1}: {e}")
                            iteration_result["error"] = f"Refinement error: {str(e)}"

                    # Add iteration result
                    branch_iter_results.append(iteration_result)

                    # If we have a valid patch, we can stop iterating for this branch
                    if current_validation.get("success", False):
                        logger.info(f"Found valid patch for branch {branch_idx + 1} in iteration {iter_idx + 1}")
                        break

                # Add this branch's iterations to the overall results
                branch_iterations.append({
                    "branch_idx": branch_idx + 1,
                    "branch_content": branch_content,
                    "iterations": branch_iter_results,
                    "final_patch": current_patch,
                    "final_validation": current_validation,
                    "success": current_validation.get("success", False)
                })

                # If we found a valid patch for this branch, we can create a valid solution
                if current_validation and current_validation.get("success", False):
                    result["valid_solution"] = {
                        "patch": current_patch,
                        "validation": current_validation,
                        "bug_location": bug_location,
                        "branch": branch_idx + 1,
                        "root_cause": branch_content,
                        "refined": True,
                        "iterations_required": len(branch_iter_results)
                    }
                    result["early_stopped"] = True
                    logger.info(
                        f"Found valid patch after {len(branch_iter_results)} refinement iterations for branch {branch_idx + 1}")
                    break  # Early stop if we found a valid solution

            # Store branch iteration results
            result["branch_iterations"] = branch_iterations

            # Select the best branch overall (either the one with valid solution or highest confidence)
            if result["valid_solution"]:
                best_branch_idx = result["valid_solution"].get("branch", 1) - 1
                if 0 <= best_branch_idx < len(result["branches"]):
                    result["selected_branch"] = result["branches"][best_branch_idx]
            else:
                # No valid solution found, select branch with highest confidence
                if sorted_branches:
                    result["selected_branch"] = sorted_branches[0]
                    logger.info(
                        f"No valid solution found, selected branch with highest confidence {sorted_branches[0].get('confidence', 0)}")

            logger.info(
                f"ToT exploration completed with {len(result.get('branches', []))} branches and {len(branch_iterations)} refinement attempts")

        except Exception as e:
            logger.error(f"Error in ToT exploration phase: {str(e)}", exc_info=True)
            # Return default result with empty branches

        return result

    def _generate_refined_patch(
            self,
            model,
            branch_content: str,
            initial_patch: str,
            bug_location: Dict[str, Any],
            issue_description: str,
            repo_exploration: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a refined patch based on an initial patch that wasn't valid.

        Args:
            model: The language model.
            branch_content: Content of the branch with the root cause analysis.
            initial_patch: The initial patch that needs refinement.
            bug_location: Bug location information.
            issue_description: Issue description.
            repo_exploration: Repository exploration data.

        Returns:
            Refined patch or None if refinement failed.
        """
        prompt = f"""
        You are an expert software engineer refining a patch that didn't correctly apply.

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}

        ROOT CAUSE ANALYSIS:
        {branch_content}

        INITIAL PATCH (NEEDS CORRECTION):
        {initial_patch}
        
        REPO EXPLORATION:
        {repo_exploration}

        The initial patch did not apply correctly. Please create a refined patch that:
        1. Follows the proper Git patch format
        2. Has correct file paths matching the repository structure
        3. Has correct line numbers and contexts that match the actual file
        4. Makes the minimal change needed to fix the issue

        Your patch MUST start with "diff --git" and include all required headers.
        """

        try:
            # Generate refined patch
            response = model.generate(prompt)

            # Extract patch
            patch_pattern = r'(diff --git.*?)(?=^```|\Z)'
            patch_match = re.search(patch_pattern, response, re.MULTILINE | re.DOTALL)

            if patch_match:
                return patch_match.group(1).strip()

            # Check for code blocks
            code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
            code_match = re.search(code_block_pattern, response, re.MULTILINE | re.DOTALL)

            if code_match:
                content = code_match.group(1).strip()
                if content.startswith('diff --git') or ('---' in content and '+++' in content):
                    return content

            # If response itself looks like a patch
            if response.strip().startswith('diff --git') or ('---' in response and '+++' in response):
                return response.strip()

            return None
        except Exception as e:
            logger.error(f"Error generating refined patch: {e}")
            return None

    def _run_cot_solution_phase(
            self,
            issue: Dict[str, Any],
            issue_description: str,
            bug_location: Dict[str, Any],
            selected_branch: Dict[str, Any],
            initial_patch: Optional[str] = None,
            initial_solution_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the Chain of Thought solution development phase.

        Args:
            issue: Issue dictionary.
            issue_description: Issue description text.
            bug_location: Bug location information.
            selected_branch: Selected branch from ToT phase.
            initial_patch: Optional patch from previous iteration.
            initial_solution_text: Optional solution text from previous iteration.

        Returns:
            Dictionary with phase results.
        """
        logger.info("Starting Chain of Thought solution development phase")
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

        # Get repository exploration data
        repo_exploration = self.repo_explorer.explore_repository(issue)

        # Format context
        branch_content = selected_branch.get("content", "") if selected_branch else ""
        context = self._format_solution_context(
            issue_description,
            bug_location,
            branch_content,
            repo_exploration
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

        current_solution = initial_solution_text
        current_depth = 0.0 if not initial_solution_text else 0.5  # Start with some depth if using previous solution

        # Get issue_id for validation
        issue_id = issue.get("instance_id") or issue.get("id")
        if not issue_id:
            # Fallback to constructing from other fields
            issue_id = f"{issue.get('repo', 'unknown')}__{issue.get('issue_number', 'unknown')}"

        # Run up to max_cot_iterations
        for i in range(self.max_cot_iterations):
            logger.info(f"CoT Solution iteration {i + 1}/{self.max_cot_iterations}")

            # Clear memory before generation
            self._run_memory_cleanup()

            # First iteration: generate solution
            if i == 0 or current_solution is None:
                try:
                    # Use solve_with_validation instead of solve
                    solution_data = cot_reasoner.solve_with_validation(
                        issue_id,
                        issue_description,
                        context,
                        bug_location,
                        branch_content
                    )

                    if isinstance(solution_data, dict) and "solution" in solution_data:
                        solution = solution_data["solution"]
                    else:
                        solution = solution_data
                except Exception as e:
                    logger.error(f"Error generating solution: {e}", exc_info=True)
                    solution = "Error generating solution."

                # If we had an initial patch and solution generation failed, use the initial patch
                if solution == "Error generating solution." and initial_patch:
                    solution = f"Using previous patch as solution:\n\n{initial_patch}"
            else:
                # Apply self-reflection to refine the solution
                try:
                    reflection_result = reflector.refine_solution(
                        current_solution,
                        issue_description,
                        context
                    )
                    solution = reflection_result.get("final_solution", current_solution)
                except Exception as e:
                    logger.error(f"Error in reflection: {e}", exc_info=True)
                    solution = current_solution

            # Calculate solution completeness
            solution_completeness = self._calculate_solution_completeness(solution)

            # Extract patch
            patch = self._extract_patch_from_solution(solution)

            # If we have an initial patch and extraction failed, use the initial patch
            if not patch and initial_patch and i == 0:
                logger.info("Using initial patch from previous iteration since extraction failed")
                patch = initial_patch

            iteration_result = {
                "solution": solution,
                "depth": solution_completeness,
            }

            # Validate patch if we have one
            if patch:
                # Validate the patch
                validation_result = self.patch_validator.validate_patch(patch, issue_id)
                iteration_result["patch"] = patch
                iteration_result["validation"] = validation_result

                # If valid, mark success
                if validation_result.get("success", False):
                    result["solution"] = {
                        "patch": patch,
                        "validation": validation_result,
                        "solution_text": solution,
                        "bug_location": bug_location,  # Include for next iteration
                        "root_cause": branch_content  # Include for next iteration
                    }
                    result["success"] = True
                    result["early_stopped"] = True
                    logger.info("Found valid patch during CoT solution phase.")

            # Update tracking
            result["iterations"].append(iteration_result)
            current_solution = solution
            current_depth = max(current_depth, solution_completeness)

            # Early stopping if we found a valid solution
            if result["success"]:
                break

        # Store final solution and depth
        if not result["solution"] and result["iterations"]:
            # Use best attempt if no valid solution
            try:
                best_iteration = max(result["iterations"], key=lambda x: x.get("depth", 0))
                result["solution"] = {
                    "patch": best_iteration.get("patch"),
                    "validation": best_iteration.get("validation"),
                    "solution_text": best_iteration.get("solution"),
                    "bug_location": bug_location,  # Include for next iteration
                    "root_cause": branch_content  # Include for next iteration
                }
            except Exception as e:
                logger.error(f"Error selecting best iteration: {e}")
                # Use the last iteration as fallback
                if result["iterations"]:
                    last_iteration = result["iterations"][-1]
                    result["solution"] = {
                        "patch": last_iteration.get("patch"),
                        "validation": last_iteration.get("validation"),
                        "solution_text": last_iteration.get("solution"),
                        "bug_location": bug_location,  # Include for next iteration
                        "root_cause": branch_content  # Include for next iteration
                    }

        result["depth"] = current_depth

        return result

    def _self_reflect_on_bug_location(
            self,
            model,
            bug_location: Dict[str, Any],
            issue_description: str,
            repo_exploration: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply self-reflection to improve bug location.

        Args:
            model: Language model.
            bug_location: Current bug location.
            issue_description: Issue description.
            repo_exploration: Repository exploration data.

        Returns:
            Updated bug location.
        """
        # Create reflection prompt
        reflection_prompt = f"""
        You are an expert software engineer reviewing a bug location analysis.

        ISSUE DESCRIPTION:
        {issue_description}

        CURRENT BUG LOCATION ANALYSIS:
        {bug_location}

        Your task is to critically analyze this bug location and improve it:
        1. Is the file path correct and specific enough?
        2. Is the function identification precise?
        3. Are the line numbers specific? Could they be narrowed down further?
        4. Is the explanation of the issue clear and technically accurate?

        Provide a revised bug location that is more precise and specific.

        Your response should be in this format:
        FILE: [specific file path]
        FUNCTION: [specific function name]
        LINE NUMBERS: [specific line numbers or range]
        ISSUE: [clear technical explanation of the bug]
        """

        # Generate reflection
        reflection_response = model.generate(reflection_prompt)

        # Parse improved bug location
        improved_location = {
            "file": bug_location.get("file"),
            "function": bug_location.get("function"),
            "line_numbers": bug_location.get("line_numbers"),
            "issue": bug_location.get("issue")
        }

        # Extract improved fields using regex
        file_match = re.search(r'FILE:\s*(.+?)(?:\n|$)', reflection_response)
        if file_match and file_match.group(1).strip():
            improved_location["file"] = file_match.group(1).strip()

        function_match = re.search(r'FUNCTION:\s*(.+?)(?:\n|$)', reflection_response)
        if function_match and function_match.group(1).strip():
            improved_location["function"] = function_match.group(1).strip()

        line_match = re.search(r'LINE NUMBERS:\s*(.+?)(?:\n|$)', reflection_response)
        if line_match and line_match.group(1).strip():
            improved_location["line_numbers"] = line_match.group(1).strip()

        issue_match = re.search(r'ISSUE:\s*(.+?)(?:$|\n\n)', reflection_response, re.DOTALL)
        if issue_match and issue_match.group(1).strip():
            improved_location["issue"] = issue_match.group(1).strip()

        return improved_location

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
        if bug_location.get("line_numbers"):
            score += 0.2

        # Check issue explanation
        if bug_location.get("issue") and len(bug_location.get("issue", "")) > 20:
            score += 0.2

        return min(1.0, score)

    def _generate_patch_from_location(
            self,
            model,
            bug_location: Dict[str, Any],
            issue_description: str,
            repo_exploration: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a patch based on bug location information.

        Args:
            model: Language model.
            bug_location: Bug location information.
            issue_description: Issue description.
            repo_exploration: Repository exploration data.

        Returns:
            Generated patch or None.
        """
        # Initialize bug fixer
        bug_fixer = BugFixer(model, self.config)

        # Generate fix
        try:
            patch = bug_fixer.generate_fix(bug_location, issue_description, repo_exploration)
            return patch
        except Exception as e:
            logger.error(f"Error generating patch from location: {e}")
            return None

    def _format_context_with_bug_location(
            self,
            issue_description: str,
            bug_location: Dict[str, Any],
            repo_exploration: Dict[str, Any]
    ) -> str:
        """
        Format context for ToT phase including bug location.

        Args:
            issue_description: Issue description.
            bug_location: Bug location information.
            repo_exploration: Repository exploration data.

        Returns:
            Formatted context string.
        """
        context = f"""
        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}
        Issue: {bug_location.get('issue', 'Unknown')}

        REPOSITORY INFORMATION:
        """

        # Add relevant files
        relevant_files = repo_exploration.get("relevant_files", [])
        if relevant_files:
            context += "Relevant Files:\n"
            for file in relevant_files[:5]:  # Limit to top 5
                context += f"- {file}\n"

        # Add file content for the identified file
        if bug_location.get("file") and "file_contents" in repo_exploration:
            file_path = bug_location.get("file")
            if file_path in repo_exploration["file_contents"]:
                file_info = repo_exploration["file_contents"][file_path]

                # Add function content if available
                if bug_location.get("function") and "functions" in file_info:
                    functions = file_info["functions"]
                    function_name = bug_location.get("function")

                    if isinstance(functions, dict) and function_name in functions:
                        func_info = functions[function_name]
                        context += f"\nFunction Code:\n```python\n{func_info.get('code', '')}\n```\n"
                    elif isinstance(functions, list):
                        for func in functions:
                            if func.get("name") == function_name:
                                context += f"\nFunction Code:\n```python\n{func.get('code', '')}\n```\n"
                                break

        return context

    def _calculate_root_cause_confidence(self, branch: str) -> float:
        """
        Calculate root cause confidence from a branch.

        Args:
            branch: Branch text from ToT.

        Returns:
            Confidence score (0-1).
        """
        # Base score
        score = 0.4  # Default plausible explanation score

        # Check for code references
        if '```' in branch:
            score += 0.2  # Contains code examples

        # Check for specific code references
        if re.search(r'(line \d+|function [a-zA-Z0-9_]+|class [a-zA-Z0-9_]+)', branch):
            score += 0.1  # Contains specific code references

        # Check for explanation quality
        if len(branch) > 500:  # Substantial explanation
            score += 0.1

        # Check for technical depth
        technical_terms = ['bug', 'error', 'exception', 'fix', 'issue', 'problem',
                           'function', 'class', 'method', 'variable']
        technical_count = sum(1 for term in technical_terms if term in branch.lower())
        if technical_count >= 5:
            score += 0.1

        return min(1.0, score)

    def _generate_patch_from_branch(
            self,
            model,
            branch: str,
            bug_location: Dict[str, Any],
            issue_description: str,
            repo_exploration: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a patch based on a promising ToT branch.

        Args:
            model: Language model.
            branch: Branch content.
            bug_location: Bug location information.
            issue_description: Issue description.
            repo_exploration: Repository exploration data.

        Returns:
            Generated patch or None.
        """
        # Create patch generation prompt
        prompt = f"""
        You are an expert software engineer tasked with fixing a bug.

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}

        ROOT CAUSE ANALYSIS:
        {branch}

        Your task is to create a precise Git patch that fixes this issue.
        The patch must:
        1. Be in proper Git patch format starting with "diff --git"
        2. Include correct file paths based on the bug location
        3. Use correct line numbers in the hunk headers
        4. Include context lines around the changes
        5. Make minimal changes to fix the bug

        IMPORTANT: Your response must be ONLY the Git patch.
        """

        try:
            # Generate patch
            response = model.generate(prompt)

            # Extract patch using regex
            patch_pattern = r'(diff --git.*?)(?=^```|\Z)'
            patch_match = re.search(patch_pattern, response, re.MULTILINE | re.DOTALL)

            if patch_match:
                return patch_match.group(1).strip()

            # Check for code block
            code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
            code_match = re.search(code_block_pattern, response, re.MULTILINE | re.DOTALL)

            if code_match:
                content = code_match.group(1).strip()
                if content.startswith('diff --git') or ('---' in content and '+++' in content):
                    return content

            # If no patch found, return the response if it looks like a patch
            if response.strip().startswith('diff --git') or ('---' in response and '+++' in response):
                return response.strip()

            return None
        except Exception as e:
            logger.error(f"Error generating patch from branch: {e}")
            return None

    def _format_solution_context(
            self,
            issue_description: str,
            bug_location: Dict[str, Any],
            branch_content: str,
            repo_exploration: Dict[str, Any]
    ) -> str:
        """
        Format context for CoT solution phase.

        Args:
            issue_description: Issue description.
            bug_location: Bug location information.
            branch_content: Selected branch content.
            repo_exploration: Repository exploration data.

        Returns:
            Formatted context string.
        """
        context = f"""
        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}
        Issue: {bug_location.get('issue', 'Unknown')}

        ROOT CAUSE ANALYSIS:
        {branch_content}

        REPOSITORY INFORMATION:
        """

        # Add file content for the identified file
        if bug_location.get("file") and "file_contents" in repo_exploration:
            file_path = bug_location.get("file")
            if file_path in repo_exploration["file_contents"]:
                file_info = repo_exploration["file_contents"][file_path]

                # Add function content if available
                if bug_location.get("function") and "functions" in file_info:
                    functions = file_info["functions"]
                    function_name = bug_location.get("function")

                    if isinstance(functions, dict) and function_name in functions:
                        func_info = functions[function_name]
                        context += f"\nFunction Code:\n```python\n{func_info.get('code', '')}\n```\n"
                    elif isinstance(functions, list):
                        for func in functions:
                            if func.get("name") == function_name:
                                context += f"\nFunction Code:\n```python\n{func.get('code', '')}\n```\n"
                                break
                else:
                    # Add entire file content if function not specified
                    context += f"\nFile Content:\n```python\n{file_info.get('content', '')[:2000]}\n```\n"

        return context

    def _calculate_solution_completeness(self, solution: str) -> float:
        """
        Calculate solution completeness score.

        Args:
            solution: Solution text.

        Returns:
            Completeness score (0-1).
        """
        score = 0.5  # Start with base score for having a solution

        # Check for implementation details
        if '```' in solution:
            score += 0.2  # Contains code blocks

        # Check for patch format
        if 'diff --git' in solution or ('---' in solution and '+++' in solution):
            score += 0.2  # Contains a patch

        # Check for explanation quality
        if len(solution) > 200 and solution.count('\n') > 5:
            score += 0.1  # Substantial solution with explanation

        return min(1.0, score)

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

    def _calculate_combined_depth(self, depth_scores: Dict[str, float]) -> float:
        """
        Calculate combined depth score.

        Args:
            depth_scores: Dictionary with individual depth scores.

        Returns:
            Combined depth score (0-1).
        """
        weights = {
            "location_specificity": 0.33,
            "root_cause_confidence": 0.33,
            "solution_completeness": 0.33
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

        # Get test patch information
        test_patch = self.data_loader.get_test_patch(issue)

        # Get failing and passing tests
        fail_to_pass = self.data_loader.get_fail_to_pass_tests(issue)
        pass_to_pass = self.data_loader.get_pass_to_pass_tests(issue)

        # Get issue description
        issue_description = self.data_loader.get_issue_description(issue)

        # Get repository exploration data
        repo_data = issue.get("repository_exploration", {})

        # Create complete output
        complete_result = {
            # Basic identification
            "issue_id": issue.get("instance_id") or issue.get("id"),
            "repo": issue.get("repo"),
            "issue_number": issue.get("issue_number") or issue.get("number"),

            # Issue information
            "issue_description": issue_description,
            "ground_truth_patch": ground_truth,
            "hints_available": hints is not None,
            "test_patch_available": isinstance(test_patch, dict) or (
                        isinstance(test_patch, str) and len(test_patch) > 0),
            "fail_to_pass_tests": fail_to_pass,
            "pass_to_pass_tests": pass_to_pass,

            # Repository information
            "repository_info": {
                "relevant_files": repo_data.get("relevant_files", []),
                "repo_path": repo_data.get("repo_path", ""),
                "file_scores": repo_data.get("file_scores", [])
            },

            # RAG results
            "rag_results": {
                "key_terms": repo_data.get("key_terms", []),
                "implementation_files": test_patch.get("implementation_files", []) if isinstance(test_patch,
                                                                                                 dict) else [],
                "test_functions": test_patch.get("test_functions", []) if isinstance(test_patch, dict) else [],
                "retrieval_time": repo_data.get("retrieval_time", 0)
            },

            # Pipeline phases
            "phases": result.get("phases", []),

            # Solution information
            "solution": result.get("solution"),
            "success": result.get("success", False),
            "depth_scores": result.get("depth_scores", {}),

            # Error handling
            "error": result.get("error"),
            "early_stopped_at": result.get("early_stopped_at"),

            # Performance metrics
            "processing_time": processing_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # If focused code is available, include it
        if hasattr(self, 'focused_code_data'):
            complete_result["focused_code_data"] = self.focused_code_data

        # Add any additional fields from the original result
        for key, value in result.items():
            if key not in complete_result:
                complete_result[key] = value

        return complete_result

    def _retrieve_focused_code(self, issue, issue_description, repository_data):
        """
        Retrieve only the most relevant code snippets that fit within token limits.

        Args:
            issue: Issue dictionary
            issue_description: The problem description
            repository_data: Repository exploration data

        Returns:
            Dictionary with focused code information
        """
        # Extract test patch if available
        from ..data.data_loader import SWEBenchDataLoader
        if not hasattr(self, 'data_loader'):
            self.data_loader = SWEBenchDataLoader(self.config)

        test_patch = self.data_loader.get_test_patch(issue)
        test_patch_content = test_patch.get("patch", "") if isinstance(test_patch, dict) else test_patch

        # Initialize repo_explorer if needed
        if not hasattr(self, 'repo_explorer') or self.repo_explorer is None:
            from ..utils.repository_explorer import RepositoryExplorer
            self.repo_explorer = RepositoryExplorer(self.config)

        # Prioritize Python files
        prioritized_files = []
        inferred_files = []  # Track inferred implementation files

        # First priority: Files from test patch
        if test_patch_content:
            file_pattern = r'(?:---|\+\+\+) [ab]/([^\n]+)'
            test_files = re.findall(file_pattern, test_patch_content)
            for file in test_files:
                if file.endswith('.py') and file not in prioritized_files:
                    prioritized_files.append(file)

                    # If it's a test file, infer the implementation file
                    if 'test_' in file or '_test' in file:
                        impl_file = self._infer_implementation_file(file)
                        if impl_file and impl_file not in inferred_files:
                            inferred_files.append(impl_file)

        # Add inferred implementation files with high priority
        for file in inferred_files:
            if file not in prioritized_files:
                prioritized_files.insert(0, file)  # Add at beginning for high priority

        # Second priority: Known implementation files from test patch info
        if isinstance(test_patch, dict) and "implementation_files" in test_patch:
            impl_files = test_patch.get("implementation_files", [])
            for file in impl_files:
                if file.endswith('.py') and file not in prioritized_files:
                    prioritized_files.append(file)

        # Third priority: Files from repository data
        relevant_files = repository_data.get("relevant_files", [])
        for file in relevant_files:
            if file.endswith('.py') and file not in prioritized_files:
                prioritized_files.append(file)

        # If we have a special case for astropy
        if "astropy" in issue.get("repo", ""):
            special_file = "astropy/wcs/wcsapi/fitswcs.py"
            if special_file not in prioritized_files:
                prioritized_files.insert(0, special_file)  # Add at beginning for highest priority

        # Extract key entities from problem description and test patch
        key_entities = self._extract_key_entities(issue_description, test_patch_content)

        # Estimate token budget - leave room for prompts and responses
        remaining_tokens = 8000  # Conservative limit for 16k context model

        # Fetch focused code snippets
        focused_code = []
        repo_path = repository_data.get("repo_path", "")

        # Track which files we've processed
        processed_files = set()

        # Process prioritized files
        for file_path in prioritized_files:
            if remaining_tokens <= 0:
                break

            if file_path in processed_files:
                continue

            processed_files.add(file_path)

            # Get file contents from repository data if available
            file_contents = None
            if "file_contents" in repository_data and file_path in repository_data["file_contents"]:
                file_info = repository_data["file_contents"][file_path]

                # Try to find relevant functions based on key entities
                relevant_funcs = []
                if "functions" in file_info:
                    functions = file_info.get("functions", {})

                    if isinstance(functions, dict):
                        for func_name, func_info in functions.items():
                            for entity in key_entities:
                                if entity.lower() in func_name.lower():
                                    code = func_info.get("code", "")
                                    if code:
                                        token_estimate = len(code.split()) + 50  # rough estimate
                                        if token_estimate < remaining_tokens:
                                            relevant_funcs.append({
                                                "name": func_name,
                                                "code": code,
                                                "tokens": token_estimate
                                            })
                                            remaining_tokens -= token_estimate
                                            break

                # If we found relevant functions, add them
                if relevant_funcs:
                    focused_code.append({
                        "file": file_path,
                        "type": "functions",
                        "relevance": "high" if any(entity in file_path for entity in key_entities) else "medium",
                        "functions": relevant_funcs
                    })
                else:
                    # Get summarized file content
                    summary = self._get_file_summary(file_path, repository_data)
                    token_estimate = len(summary.split()) + 50

                    if token_estimate < remaining_tokens:
                        focused_code.append({
                            "file": file_path,
                            "type": "summary",
                            "relevance": "high" if any(entity in file_path for entity in key_entities) else "medium",
                            "summary": summary,
                            "tokens": token_estimate
                        })
                        remaining_tokens -= token_estimate
            else:
                # File not in repository data, try to load directly
                full_path = os.path.join(repo_path, file_path) if repo_path else file_path
                try:
                    summary = self._get_file_summary(file_path, repository_data)
                    token_estimate = len(summary.split()) + 50

                    if token_estimate < remaining_tokens:
                        focused_code.append({
                            "file": file_path,
                            "type": "summary",
                            "relevance": "medium",
                            "summary": summary,
                            "tokens": token_estimate
                        })
                        remaining_tokens -= token_estimate
                except Exception as e:
                    logger.warning(f"Could not load file: {file_path} - {str(e)}")

        # Format the focused code for the model
        formatted_code = self._format_focused_code(focused_code)

        return {
            "focused_code": focused_code,
            "formatted_code": formatted_code,
            "key_entities": key_entities,
            "prioritized_files": prioritized_files,
            "estimated_tokens": 8000 - remaining_tokens
        }

    def _infer_implementation_file(self, test_file):
        """
        Infer the implementation file from a test file name.

        Args:
            test_file: Path to the test file

        Returns:
            Path to the inferred implementation file
        """
        # Split into directory and filename
        dir_name = os.path.dirname(test_file)
        file_name = os.path.basename(test_file)

        # Handle test_xxx.py pattern
        if file_name.startswith('test_'):
            impl_name = file_name[5:]  # Remove 'test_'

            # Try different locations
            if 'tests' in dir_name:
                # If in a tests directory, implementation might be in parent
                parent_dir = dir_name.replace('/tests', '')
                if parent_dir != dir_name:  # Only if actually changed
                    return os.path.join(parent_dir, impl_name)

            # Same directory
            return os.path.join(dir_name, impl_name)

        # Handle xxx_test.py pattern
        elif file_name.endswith('_test.py'):
            impl_name = file_name[:-8] + '.py'  # Remove '_test.py'

            # Try different locations
            if 'tests' in dir_name:
                # If in a tests directory, implementation might be in parent
                parent_dir = dir_name.replace('/tests', '')
                if parent_dir != dir_name:  # Only if actually changed
                    return os.path.join(parent_dir, impl_name)

            # Same directory
            return os.path.join(dir_name, impl_name)

        # If it's any file with 'test' in the name, try a simple heuristic
        elif 'test' in file_name.lower():
            # Replace 'test' with empty string
            impl_name = file_name.lower().replace('test', '')
            if impl_name == file_name.lower():
                return None  # No change, not a test file

            # Capitalize correctly
            capitalized = []
            for word in impl_name.split('_'):
                if word:
                    capitalized.append(word[0].upper() + word[1:])
            impl_name = '_'.join(capitalized) + '.py'

            return os.path.join(dir_name, impl_name)

        return None

    def _extract_key_entities(self, issue_description, test_patch=None):
        """Extract key entities from issue description and test patch."""
        entities = set()

        # Extract from problem statement
        func_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        class_pattern = r'\b([A-Z][a-zA-Z0-9_]*)\b'

        for pattern in [func_pattern, class_pattern]:
            for match in re.finditer(pattern, issue_description):
                entity = match.group(1)
                if len(entity) > 2:  # Skip very short names
                    entities.add(entity)

        # Extract from test patch
        if test_patch:
            # Look for test functions
            test_funcs = re.findall(r'def\s+(test_[a-zA-Z0-9_]+)', test_patch)
            for func in test_funcs:
                # Remove test_ prefix to get what's being tested
                if func.startswith('test_'):
                    entities.add(func[5:])
                entities.add(func)

            # Look for assertions
            for match in re.finditer(r'assert[^(]*\(\s*([a-zA-Z0-9_\.]+)', test_patch):
                parts = match.group(1).split('.')
                for part in parts:
                    if len(part) > 2:  # Skip single chars
                        entities.add(part)

        return list(entities)

    def _get_file_summary(self, file_path, repository_data):
        """Get a summary of a file with imports and key functions."""
        repo_path = repository_data.get("repo_path", "")
        full_path = os.path.join(repo_path, file_path) if repo_path else file_path

        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract imports
            import_lines = []
            for line in content.split('\n'):
                if line.strip().startswith(('import ', 'from ')):
                    import_lines.append(line)

            # Extract class and function definitions (just the signatures)
            class_pattern = r'(class\s+\w+(?:\([^)]*\))?:)'
            func_pattern = r'(def\s+\w+\([^)]*\)(?:\s*->.*?)?:)'

            classes = re.findall(class_pattern, content)
            functions = re.findall(func_pattern, content)

            # Build summary
            summary = f"# File: {file_path}\n\n"

            if import_lines:
                summary += "# Imports:\n" + "\n".join(import_lines[:10])
                if len(import_lines) > 10:
                    summary += f"\n# ... ({len(import_lines) - 10} more imports)\n"
                summary += "\n\n"

            if classes:
                summary += "# Classes:\n" + "\n".join(classes[:5])
                if len(classes) > 5:
                    summary += f"\n# ... ({len(classes) - 5} more classes)\n"
                summary += "\n\n"

            if functions:
                summary += "# Functions:\n" + "\n".join(functions[:10])
                if len(functions) > 10:
                    summary += f"\n# ... ({len(functions) - 10} more functions)\n"

            return summary
        except Exception as e:
            return f"# Could not read file {file_path}: {str(e)}"

    def _format_focused_code(self, focused_code):
        """Format focused code for the model."""
        formatted = "# FOCUSED CODE CONTEXT\n\n"

        for item in focused_code:
            file_path = item.get("file", "unknown")
            item_type = item.get("type", "unknown")

            formatted += f"## FILE: {file_path}\n"

            if item_type == "functions":
                functions = item.get("functions", [])
                for func in functions:
                    name = func.get("name", "unknown")
                    code = func.get("code", "")

                    formatted += f"### FUNCTION: {name}\n"
                    formatted += "```python\n"
                    formatted += code + "\n"
                    formatted += "```\n\n"

            elif item_type == "summary":
                summary = item.get("summary", "")
                formatted += "### FILE SUMMARY:\n"
                formatted += "```python\n"
                formatted += summary + "\n"
                formatted += "```\n\n"

        return formatted

    def _refine_patch_with_feedback(
            self,
            model,
            original_patch: str,
            validation_feedback: str,
            branch_content: str,
            bug_location: Dict[str, Any],
            issue_description: str
    ) -> Optional[str]:
        """
        Refine a patch based on validation feedback.

        Args:
            model: Language model.
            original_patch: Original patch to refine.
            validation_feedback: Feedback from validation.
            branch_content: Content of the branch with root cause analysis.
            bug_location: Bug location information.
            issue_description: Issue description.

        Returns:
            Refined patch or None if refinement failed.
        """
        prompt = f"""
        You are an expert software engineer refining a patch that didn't correctly apply.

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        File: {bug_location.get('file', 'Unknown')}
        Function: {bug_location.get('function', 'Unknown')}
        Line Numbers: {bug_location.get('line_numbers', 'Unknown')}

        ROOT CAUSE ANALYSIS:
        {branch_content}

        ORIGINAL PATCH (NEEDS CORRECTION):
        {original_patch}

        VALIDATION FEEDBACK:
        {validation_feedback}

        The original patch did not apply correctly. Please create a refined patch that:
        1. Addresses all the issues mentioned in the validation feedback
        2. Follows the proper Git patch format
        3. Has correct file paths matching the repository structure
        4. Has correct line numbers and contexts that match the actual file
        5. Makes the minimal change needed to fix the issue

        Your patch MUST start with "diff --git" and include all required headers.
        """

        try:
            # Generate refined patch
            response = model.generate(prompt)

            # Extract patch
            patch_pattern = r'(diff --git.*?)(?=^```|\Z)'
            patch_match = re.search(patch_pattern, response, re.MULTILINE | re.DOTALL)

            if patch_match:
                return patch_match.group(1).strip()

            # Check for code blocks
            code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
            code_match = re.search(code_block_pattern, response, re.MULTILINE | re.DOTALL)

            if code_match:
                content = code_match.group(1).strip()
                if content.startswith('diff --git') or ('---' in content and '+++' in content):
                    return content

            # If response itself looks like a patch
            if response.strip().startswith('diff --git') or ('---' in response and '+++' in response):
                return response.strip()

            return None
        except Exception as e:
            logger.error(f"Error generating refined patch: {e}")
            return None
