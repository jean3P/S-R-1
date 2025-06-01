import inspect
import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
import gc

from .adaptive_exploration import AdaptiveExplorationStrategy, BranchHistory
from ..utils.solution_diversity_analyzer import SolutionDiversityAnalyzer
from ..models import create_model
from ..utils.leetcode_test_runner import LeetCodeTestRunner
from ..evaluation.code_evaluator import CodeEvaluator

logger = logging.getLogger(__name__)


class LeetCodeSolutionPipeline:
    """
    Pipeline for generating LeetCode solutions with multi-round self-reflection.
    Generates multiple candidate solutions and iteratively improves them.
    """

    def __init__(self, config, model_name=None):
        """
        Initialize the LeetCode solution pipeline.

        Args:
            config: Configuration object.
            model_name: Name of the model to use, or None to use default.
        """
        self.num_candidates = None
        self.config = config
        self.model_name = model_name or config.get("default_model", "deepseek-r1-distill")
        self.model = None  # Lazy-load model on demand

        # Initialize test runner
        self.test_runner = LeetCodeTestRunner(config)

        # Configure tree-based parameters
        self.initial_k = config.get("leetcode", {}).get("initial_solutions", 3)
        self.branch_factor = config.get("leetcode", {}).get("branch_factor", 3)
        self.max_depth = config.get("leetcode", {}).get("max_depth", 3)
        self.early_stopping = config.get("leetcode", {}).get("early_stopping", True)

        # Add adaptive parameters
        self.adaptive_mode = config.get("leetcode", {}).get("adaptive_mode", False)
        self.consecutive_failures_threshold = config.get("leetcode", {}).get("consecutive_failures", 3)

        # Cache for tested solutions
        self.solution_cache = {}

        # Separate tracker for import failures (not mixed with solution cache)
        self.global_import_failures = set()

        # Directory for storing results
        self.results_dir = Path(config.get("evaluation", {}).get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Memory optimization
        self.use_memory_optimization = config.get("memory_efficient", True)

        # Initialize code evaluator if enabled
        self.use_code_eval = config.get("evaluation", {}).get("use_code_eval", False)
        if self.use_code_eval:
            self.code_evaluator = CodeEvaluator(config)
            logger.info("Initialized code evaluator with HuggingFace code_eval")
        else:
            self.code_evaluator = None

        # Initialize adaptive exploration strategy if enabled
        if self.adaptive_mode:
            self.adaptive_strategy = AdaptiveExplorationStrategy(config.get("adaptive", {}))
            logger.info("Initialized adaptive exploration strategy")
        else:
            self.adaptive_strategy = None

        logger.info(f"Initialized LeetCodeSolutionPipeline with model {self.model_name}")
        logger.info(
            f"Tree parameters: initial_k={self.initial_k}, branch_factor={self.branch_factor}, max_depth={self.max_depth}")
        logger.info(f"Adaptive mode: {self.adaptive_mode}")

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

    def solve_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve a LeetCode problem using tree-based branching approach.

        Args:
            problem_data: Problem data dictionary.

        Returns:
            Dictionary with solution results.
        """
        logger.info(f"Starting tree-based solution generation for problem {problem_data['problem_id']}")
        start_time = time.time()

        # Initialize tracking
        solution_tree = []
        all_solutions = []
        best_solution = None
        solution_found = False

        # Initialize branch histories for adaptive mode
        branch_histories = {}

        # Track statistics - coherent initialization
        stats = {
            # Basic counters
            "nodes_explored": 0,
            "candidates_generated": 0,

            # Test results breakdown
            "tests_passed": 0,
            "tests_failed": 0,
            "test_timeouts": 0,
            "test_errors": 0,  # Other errors not covered by specific categories

            # Import-specific tracking
            "import_errors": 0,  # Number of solutions that failed due to import errors
            "import_terminated_branches": 0,  # Number of branches terminated due to repeated import failures
            "unique_import_failures": set(),  # Set of unique modules that failed to import

            # Performance metrics
            "execution_times": [],
            "tree_depth": 0,

            # Termination tracking
            "termination_reasons": {
                "depth_limit": 0,
                "adaptive_threshold": 0,
                "import_failures": 0,
                "early_stopping": 0,
                "iteration_limit": 0
            },

            # Solution diversity
            "solution_diversity": {
                "unique_solutions": 0,
                "similarity_score": 0.0,
                "solution_lengths": {"min": 0, "max": 0, "avg": 0.0}
            },

            # Test case analysis
            "test_case_analysis": {
                "hardest_cases": {},  # Test cases with highest failure rates
                "first_failing_tests": {}  # Which tests fail first most often
            },

            # Feedback effectiveness tracking
            "feedback_impact": {
                "depths": {},
                "error_types": {},
                "test_case_improvements": {},
                "error_transitions": {}
            }
        }

        try:
            # Generate initial k solutions
            initial_candidates = self._generate_initial_candidates(problem_data)

            for i, candidate in enumerate(initial_candidates):
                stats["candidates_generated"] += 1
                stats["nodes_explored"] += 1

                # Evaluate immediately
                candidate_hash = self._calculate_solution_hash(candidate)
                test_result = self._test_solution(problem_data, candidate, stats)

                if "execution_time" in test_result:
                    stats["execution_times"].append(test_result["execution_time"])

                solution_node = {
                    "node_id": f"0_{i}",
                    "solution": candidate,
                    "solution_hash": candidate_hash,
                    "test_result": test_result,
                    "depth": 0,
                    "parent_id": None,
                    "children": [],
                    "passed": test_result.get("status") == "pass"
                }

                # Initialize branch history for adaptive mode
                if self.adaptive_mode:
                    branch_history = BranchHistory(node_id=solution_node["node_id"])
                    branch_histories[solution_node["node_id"]] = branch_history

                    # Track initial error
                    error_type = self._categorize_error(test_result.get("error_message", ""))
                    if error_type != "unknown":
                        branch_history.add_error(error_type)

                # Track failing tests coherently
                self._track_test_failures(test_result, stats)

                all_solutions.append(solution_node)

                # Update statistics based on test result
                self._update_test_stats(test_result, stats)

                # Update best solution if this one passed
                if test_result.get("status") == "pass":
                    solution_found = True
                    if best_solution is None:
                        best_solution = solution_node

                    # Early stopping check
                    if self.early_stopping:
                        logger.info(
                            f"Solution found in initial generation (node {solution_node['node_id']}), stopping early")
                        solution_tree.append(solution_node)
                        break

                # Branch from failed solutions
                if test_result.get("status") != "pass":
                    # Pass branch_histories to _branch_from_failure
                    self._branch_from_failure(
                        solution_node,
                        problem_data,
                        all_solutions,
                        stats,
                        depth=1,
                        branch_histories=branch_histories if self.adaptive_mode else None
                    )

                solution_tree.append(solution_node)

                # Check if we found a solution during branching
                if self.early_stopping and self._check_tree_for_solution(solution_node, all_solutions):
                    passing_solutions = [n for n in all_solutions if n.get("passed", False)]
                    if passing_solutions:
                        logger.info(
                            f"Solution found during branching (node {passing_solutions[0]['node_id']}), stopping early")
                    else:
                        logger.warning("Early stopping triggered but no passing solution found - this is a bug!")
                    break

            if not solution_found:
                stats["termination_reasons"]["iteration_limit"] += 1

            # Collect all solutions for final evaluation
            final_solutions = [s for s in all_solutions if s["passed"]]

            # Update best solution from all explored nodes
            for node in all_solutions:
                if (node["passed"]
                        and (
                                best_solution is None
                                or node["test_result"].get("execution_time", float("inf"))
                                < best_solution["test_result"].get("execution_time", float("inf")))
                ):
                    best_solution = node

        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}", exc_info=True)
            return {
                "problem_id": problem_data["problem_id"],
                "status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time
            }

        # Add adaptive exploration summary if applicable
        if self.adaptive_mode and self.adaptive_strategy:
            stats["adaptive_exploration_summary"] = self.adaptive_strategy.get_exploration_summary()

        # Perform batch evaluation with code_eval if enabled
        code_eval_results = None
        if self.code_evaluator is not None and all_solutions:
            logger.info("Performing batch evaluation with code_eval")
            solution_codes = [node["solution"] for node in all_solutions]

            # Add reference solution if available
            if problem_data.get("reference_solution"):
                solution_codes.append(problem_data["reference_solution"])

            code_eval_results = self.code_evaluator.evaluate_solutions(
                problem_data,
                solution_codes
            )

        if code_eval_results and "pass_at_k" in code_eval_results:
            logger.info("Adding code_eval depth correlation analysis")

            # Group solutions by depth
            solutions_by_depth = {}
            for node in all_solutions:
                depth = node.get("depth", 0)
                if depth not in solutions_by_depth:
                    solutions_by_depth[depth] = []
                solutions_by_depth[depth].append(node)

            # Calculate statistics for all solutions
            all_depths = [node.get("depth", 0) for node in all_solutions]

            depth_stats = {
                "min_depth": min(all_depths) if all_depths else 0,
                "max_depth": max(all_depths) if all_depths else 0,
                "avg_depth": sum(all_depths) / len(all_depths) if all_depths else 0,
                "solutions_per_depth": {depth: len(nodes) for depth, nodes in solutions_by_depth.items()},
                "passing_solutions_per_depth": {
                    depth: sum(1 for node in nodes if node.get("passed", False))
                    for depth, nodes in solutions_by_depth.items()
                }
            }

            # Add to stats
            stats["code_eval_metrics"] = {
                "depth_statistics": depth_stats,
                "pass_at_k": code_eval_results["pass_at_k"],
                "solutions_evaluated": code_eval_results.get("solutions_evaluated", len(all_solutions))
            }

            # Add success rate by depth if we have passing solutions
            if final_solutions:
                passing_depths = [node.get("depth", 0) for node in final_solutions]
                stats["code_eval_metrics"]["passing_solution_depths"] = {
                    "min": min(passing_depths) if passing_depths else 0,
                    "max": max(passing_depths) if passing_depths else 0,
                    "avg": sum(passing_depths) / len(passing_depths) if passing_depths else 0
                }

        # Calculate summary statistics with proper error handling
        stats["summary"] = self._calculate_summary_stats(stats, all_solutions)

        stats_serializable = self._prepare_stats_for_serialization(stats)

        # Prepare final result
        total_time = time.time() - start_time

        result = {
            "problem_id": problem_data["problem_id"],
            "problem_title": problem_data["problem_title"],
            "difficulty": problem_data.get("difficulty", ""),
            "status": "solved" if solution_found else "unsolved",
            "best_solution": best_solution["solution"] if best_solution else None,
            "passed_solutions": [s["solution"] for s in final_solutions],
            "all_solutions": [s["solution"] for s in all_solutions],
            "total_candidates": stats["candidates_generated"],
            "nodes_explored": stats["nodes_explored"],
            "tree_depth": self._calculate_tree_depth(solution_tree, all_solutions),
            "solution_tree": solution_tree,
            "stats": stats_serializable,
            "processing_time": total_time
        }

        # Add code_eval results if available
        if code_eval_results:
            result["code_eval_results"] = code_eval_results

        # Calculate solution diversity
        diversity_metrics = self._measure_solution_diversity(all_solutions)
        result["stats"]["solution_diversity"] = diversity_metrics

        # Track adaptive statistics if enabled
        if self.adaptive_mode:
            from ..utils.adaptive_statistics import AdaptiveStatisticsTracker
            tracker = AdaptiveStatisticsTracker(self.results_dir)
            adaptive_stats = tracker.track_problem_exploration(
                problem_data["problem_id"],
                self.model_name,
                result,
                self.adaptive_strategy,
                branch_histories
            )
            result["adaptive_stats"] = adaptive_stats

        # Save result to file
        self._save_results(problem_data["problem_id"], result)

        return result

    def _prepare_stats_for_serialization(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare stats dictionary for JSON serialization by converting sets to lists.
        """
        import copy

        # Deep copy to avoid modifying the original
        stats_copy = copy.deepcopy(stats)

        # Convert unique_import_failures set to list
        if "unique_import_failures" in stats_copy and isinstance(stats_copy["unique_import_failures"], set):
            stats_copy["unique_import_failures"] = list(stats_copy["unique_import_failures"])

        # Also check in summary if it exists
        if "summary" in stats_copy and "test_results" in stats_copy["summary"]:
            if "unique_import_failures" in stats_copy["summary"]["test_results"] and isinstance(
                    stats_copy["summary"]["test_results"]["unique_import_failures"], set):
                stats_copy["summary"]["test_results"]["unique_import_failures"] = list(
                    stats_copy["summary"]["test_results"]["unique_import_failures"])

        # Check import analysis in summary
        if "summary" in stats_copy and "import_analysis" in stats_copy["summary"]:
            if "missing_modules" in stats_copy["summary"]["import_analysis"] and isinstance(
                    stats_copy["summary"]["import_analysis"]["missing_modules"], set):
                stats_copy["summary"]["import_analysis"]["missing_modules"] = list(
                    stats_copy["summary"]["import_analysis"]["missing_modules"])

        return stats_copy

    def _branch_from_failure(
            self,
            parent_node: Dict[str, Any],
            problem_data: Dict[str, Any],
            all_solutions: List[Dict[str, Any]],
            stats: Dict[str, Any],
            depth: int,
            failure_counts: Dict[str, int] = None,
            branch_import_failures: Dict[str, set] = None,
            branch_histories: Dict[str, Any] = None
    ) -> bool:
        """
        Generate new solutions from a failed solution using tree branching.

        Args:
            parent_node: The failed solution node to branch from
            problem_data: Problem data dictionary
            all_solutions: List to append all generated solutions
            stats: Statistics dictionary to update
            depth: Current depth in the tree
            failure_counts: Dictionary tracking consecutive failures per branch (for adaptive mode)
            branch_import_failures: Dictionary tracking import failures in this branch
            branch_histories: Dictionary of BranchHistory objects for adaptive mode

        Returns:
            bool: True if a passing solution was found in this branch
        """
        # Initialize failure counts dictionary if this is the first call
        if self.adaptive_mode and failure_counts is None:
            failure_counts = {}

        # Initialize import failures tracking if this is the first call
        if branch_import_failures is None:
            branch_import_failures = {}

        parent_id = parent_node["node_id"]

        # Get parent error type for feedback tracking
        parent_result = parent_node.get("test_result", {})
        parent_error_type = self._categorize_error(parent_result.get("error_message", ""))
        parent_failed_tests = len(parent_result.get("failed_tests", []))

        # Update parent branch history if in adaptive mode
        if self.adaptive_mode and branch_histories and parent_id in branch_histories:
            parent_history = branch_histories[parent_id]
            parent_history.failures += 1
        else:
            parent_history = None

        # Initialize depth tracking if needed
        if str(depth) not in stats["feedback_impact"]["depths"]:
            stats["feedback_impact"]["depths"][str(depth)] = {
                "attempts": 0,
                "improvements": 0,
                "solved": 0
            }

        # Track error type statistics
        if parent_error_type:
            if parent_error_type not in stats["feedback_impact"]["error_types"]:
                stats["feedback_impact"]["error_types"][parent_error_type] = {
                    "attempts": 0,
                    "improvements": 0
                }
            stats["feedback_impact"]["error_types"][parent_error_type]["attempts"] += 1

        # Check if the parent has import errors
        if parent_node["test_result"].get("status") == "import_error":
            # Get the list of imports that failed
            parent_failed_imports = parent_node["test_result"].get("import_failures", [])

            # Create a branch key for tracking import failures
            branch_key = f"branch:{parent_id}"

            # Initialize this branch's failures if needed
            if branch_key not in branch_import_failures:
                branch_import_failures[branch_key] = set()

            # Check for repeated failures
            repeated_failures = set(parent_failed_imports) & branch_import_failures[branch_key]

            # If any imports have already failed in this branch, stop exploring
            if repeated_failures:
                logger.info(f"Stopping branch {parent_id} due to repeated import failures: {repeated_failures}")

                stats["import_terminated_branches"] += 1
                # Add repeated failures to unique import failures
                for module in repeated_failures:
                    stats["unique_import_failures"].add(module)

                stats["termination_reasons"]["import_failures"] += 1
                return False

            # Add current failures to the tracking set
            branch_import_failures[branch_key].update(parent_failed_imports)

        # ADAPTIVE MODE: Use adaptive termination logic
        if self.adaptive_mode and self.adaptive_strategy and branch_histories and parent_id in branch_histories:
            parent_history = branch_histories[parent_id]
            should_terminate, confidence, reason = self.adaptive_strategy.should_terminate_branch(
                parent_node, parent_history, depth, stats
            )

            if should_terminate:
                logger.info(
                    f"Adaptive termination: Stopping branch {parent_id} at depth {depth} (confidence={confidence:.2f}, reason={reason})")
                stats["termination_reasons"]["adaptive_threshold"] += 1
                return False
        # FIXED MODE: Check max depth limit
        elif not self.adaptive_mode and depth >= self.max_depth:
            logger.debug(f"Max depth {self.max_depth} reached, stopping branching")
            stats["termination_reasons"]["depth_limit"] += 1
            return False

        # ADAPTIVE MODE (fallback): Check consecutive failures threshold
        if self.adaptive_mode and not (self.adaptive_strategy and branch_histories):
            current_failures = failure_counts.get(parent_id, 0)
            effective_threshold = max(1, self.consecutive_failures_threshold - depth // 3)
            if current_failures >= effective_threshold:
                # Add tracking for adaptive termination analysis
                if "adaptive_stats" not in stats:
                    stats["adaptive_stats"] = {
                        "branches_terminated": 0,
                        "termination_depths": [],
                        "effective_thresholds": []
                    }
                stats["adaptive_stats"]["branches_terminated"] += 1
                stats["adaptive_stats"]["termination_depths"].append(depth)
                stats["adaptive_stats"]["effective_thresholds"].append(effective_threshold)

                logger.info(
                    f"Stopping branch {parent_id} at depth {depth} (effective threshold: {effective_threshold})")
                stats["termination_reasons"]["adaptive_threshold"] += 1
                return False

        logger.info(f"Branching from failed solution {parent_id} at depth {depth}")

        # Generate improved candidates based on the failure
        improved_candidates = self._generate_improved_candidates_for_node(
            problem_data,
            parent_node,
            self.branch_factor
        )

        solution_found = False
        any_improvement = False  # Track if any child improved over parent (for adaptive mode)

        # Update depth attempt count at this point
        stats["feedback_impact"]["depths"][str(depth)]["attempts"] += len(improved_candidates)

        # Update parent history for candidates generated
        if parent_history:
            parent_history.candidates_generated += len(improved_candidates)

        for i, candidate in enumerate(improved_candidates):
            stats["candidates_generated"] += 1
            stats["nodes_explored"] += 1

            # Calculate hash and check cache
            candidate_hash = self._calculate_solution_hash(candidate)
            if candidate_hash in self.solution_cache:
                test_result = self.solution_cache[candidate_hash]
                logger.debug(f"Using cached result for solution hash {candidate_hash[:8]}")
            else:
                test_result = self._test_solution(problem_data, candidate, stats)
                self.solution_cache[candidate_hash] = test_result

            if "execution_time" in test_result:
                stats["execution_times"].append(test_result["execution_time"])

            # Create child node
            child_node = {
                "node_id": f"{depth}_{len(all_solutions)}",
                "solution": candidate,
                "solution_hash": candidate_hash,
                "test_result": test_result,
                "depth": depth,
                "parent_id": parent_id,
                "children": [],
                "passed": test_result.get("status") == "pass"
            }

            # Create branch history for child if in adaptive mode
            if self.adaptive_mode and branch_histories:
                child_history = BranchHistory(node_id=child_node["node_id"])
                if parent_history:
                    child_history.start_time = parent_history.start_time
                branch_histories[child_node["node_id"]] = child_history

            # Track failing tests
            self._track_test_failures(test_result, stats)

            # Add to parent and solution list
            parent_node["children"].append(child_node["node_id"])
            all_solutions.append(child_node)

            # Track feedback effectiveness
            child_error_type = self._categorize_error(test_result.get("error_message", ""))
            child_failed_tests = len(test_result.get("failed_tests", []))

            # Track error in child history
            if self.adaptive_mode and branch_histories and child_node["node_id"] in branch_histories:
                child_history = branch_histories[child_node["node_id"]]
                if child_error_type != "unknown":
                    child_history.add_error(child_error_type)

            transition_key = f"{parent_error_type}->{child_error_type}"
            if transition_key not in stats["feedback_impact"]["error_transitions"]:
                stats["feedback_impact"]["error_transitions"][transition_key] = 0
            stats["feedback_impact"]["error_transitions"][transition_key] += 1

            # Check for improvement from parent to child
            improved = False
            tests_fixed = 0

            # Update test statistics
            self._update_test_stats(test_result, stats)

            # Process test result
            if test_result.get("status") == "pass":
                # Solution passed
                solution_found = True
                any_improvement = True
                improved = True
                tests_fixed = parent_failed_tests

                # Record solution at this depth
                stats["feedback_impact"]["depths"][str(depth)]["solved"] += 1

                # Record error type improvement
                if parent_error_type:
                    stats["feedback_impact"]["error_types"][parent_error_type]["improvements"] += 1

                logger.info(f"Solution found at node {child_node['node_id']} (depth {depth})")

                # For adaptive mode, reset failure count for this branch
                if self.adaptive_mode:
                    failure_counts[parent_id] = 0

                # Early stopping: don't branch further from passing solutions
                if self.early_stopping:
                    continue

            elif test_result.get("status") == "fail":
                # Check for partial improvement (fewer failed tests)
                if child_failed_tests < parent_failed_tests and parent_failed_tests > 0:
                    improved = True
                    tests_fixed = parent_failed_tests - child_failed_tests
                    stats["feedback_impact"]["depths"][str(depth)]["improvements"] += 1
                    if parent_error_type:
                        stats["feedback_impact"]["error_types"][parent_error_type]["improvements"] += 1

                # For adaptive mode, track failures
                if self.adaptive_mode:
                    # Check if this solution is "better" than parent (e.g., passes more tests)
                    if child_failed_tests < parent_failed_tests:
                        # Child is better than parent, reset failure count
                        failure_counts[parent_id] = 0
                        any_improvement = True
                    else:
                        # Child is not better, increment failure count
                        failure_counts[parent_id] = failure_counts.get(parent_id, 0) + 1

                    # Set child's initial failure count
                    failure_counts[child_node["node_id"]] = 0

            elif test_result.get("status") == "import_error":
                # If previous was also import error but this one has different imports,
                # consider it partial improvement
                if parent_result.get("status") == "import_error":
                    parent_imports = set(parent_result.get("import_failures", []))
                    child_imports = set(test_result.get("import_failures", []))
                    if parent_imports and child_imports and len(child_imports) < len(parent_imports):
                        improved = True
                        stats["feedback_impact"]["depths"][str(depth)]["improvements"] += 1
                        if parent_error_type:
                            stats["feedback_impact"]["error_types"][parent_error_type]["improvements"] += 1

                if self.adaptive_mode:
                    # Count import errors as failures for adaptive mode
                    failure_counts[parent_id] = failure_counts.get(parent_id, 0) + 1
                    failure_counts[child_node["node_id"]] = 0
            else:
                # Other error in testing
                if self.adaptive_mode:
                    # Count errors as failures for adaptive mode
                    failure_counts[parent_id] = failure_counts.get(parent_id, 0) + 1
                    failure_counts[child_node["node_id"]] = 0

            # Update overall improvements tracking
            if improved:
                any_improvement = True

                # Update branch histories with improvement
                if parent_history and parent_failed_tests > 0:
                    improvement_rate = tests_fixed / parent_failed_tests
                    parent_history.add_improvement(improvement_rate, tests_fixed)

                if self.adaptive_mode and branch_histories and child_node["node_id"] in branch_histories:
                    child_history = branch_histories[child_node["node_id"]]
                    if parent_failed_tests > 0:
                        improvement_rate = tests_fixed / parent_failed_tests
                        child_history.add_improvement(improvement_rate, tests_fixed)

                # Track test case improvements
                if "failed_tests" in parent_result and "failed_tests" in test_result:
                    # Track which specific test cases were fixed
                    for parent_test in parent_result["failed_tests"]:
                        # Get input signature as a string
                        if isinstance(parent_test.get("input"), dict):
                            test_key = str(sorted(parent_test["input"].items()))
                        else:
                            test_key = str(parent_test.get("input"))

                        # Check if this test case was fixed
                        fixed = True
                        for child_test in test_result["failed_tests"]:
                            if isinstance(child_test.get("input"), dict):
                                child_key = str(sorted(child_test["input"].items()))
                            else:
                                child_key = str(child_test.get("input"))

                            if child_key == test_key:
                                fixed = False
                                break

                        if fixed and test_key:
                            # This test case was fixed
                            if test_key not in stats["feedback_impact"]["test_case_improvements"]:
                                stats["feedback_impact"]["test_case_improvements"][test_key] = 0
                            stats["feedback_impact"]["test_case_improvements"][test_key] += 1

            # Decide whether to branch further from this child
            should_branch = (
                    test_result.get("status") != "pass" and  # Not already solved
                    (not solution_found or not self.early_stopping)  # Not stopping early or solution not found yet
            )

            if should_branch:
                # FIXED MODE: branch if not at max depth yet
                if not self.adaptive_mode and depth < self.max_depth - 1:
                    child_success = self._branch_from_failure(
                        child_node, problem_data, all_solutions, stats, depth + 1,
                        failure_counts, branch_import_failures, branch_histories
                    )
                    solution_found = solution_found or child_success

                # ADAPTIVE MODE: branch if failures are below threshold
                elif self.adaptive_mode and failure_counts.get(child_node["node_id"],
                                                               0) < self.consecutive_failures_threshold:
                    child_success = self._branch_from_failure(
                        child_node, problem_data, all_solutions, stats, depth + 1,
                        failure_counts, branch_import_failures, branch_histories
                    )
                    solution_found = solution_found or child_success
                    any_improvement = any_improvement or child_success

                # If found solution in deeper branch and using early stopping, break
                if solution_found and self.early_stopping:
                    stats["termination_reasons"]["early_stopping"] += 1
                    break

        # Update parent history timing
        if parent_history:
            parent_history.total_time = time.time() - parent_history.start_time

        # In adaptive mode - if we've made some improvements but haven't found a solution,
        # reset the failure counter to give this branch more chances
        if self.adaptive_mode and any_improvement and not solution_found:
            failure_counts[parent_id] = 0
            if "adaptive_events" not in stats:
                stats["adaptive_events"] = {"branch_extensions": 0, "improvement_resets": 0}
            stats["adaptive_events"]["improvement_resets"] += 1

        if not solution_found and not self.early_stopping:
            stats["termination_reasons"]["iteration_limit"] += 1

        return solution_found

    def _update_test_stats(self, test_result: Dict[str, Any], stats: Dict[str, Any]):
        """
        Update statistics based on test result in a coherent way.
        """
        status = test_result.get("status", "error")

        if status == "pass":
            stats["tests_passed"] += 1
        elif status == "fail":
            stats["tests_failed"] += 1
        elif status == "timeout":
            stats["test_timeouts"] += 1
        elif status == "import_error":
            stats["import_errors"] += 1
            # Track unique import failures
            for failed_import in test_result.get("import_failures", []):
                stats["unique_import_failures"].add(failed_import)
        else:
            # Any other error type
            stats["test_errors"] += 1

    def _track_test_failures(self, test_result: Dict[str, Any], stats: Dict[str, Any]):
        """
        Track failing test cases for analysis.
        """
        if "failed_tests" in test_result and test_result["failed_tests"]:
            # Get first failing test
            first_test = test_result["failed_tests"][0]
            first_test_key = str(first_test.get("input", "unknown"))

            # Track first failing test
            if first_test_key not in stats["test_case_analysis"]["first_failing_tests"]:
                stats["test_case_analysis"]["first_failing_tests"][first_test_key] = 0
            stats["test_case_analysis"]["first_failing_tests"][first_test_key] += 1

            # Track all failing tests
            for test in test_result["failed_tests"]:
                test_key = str(test.get("input", "unknown"))
                if test_key not in stats["test_case_analysis"]["hardest_cases"]:
                    stats["test_case_analysis"]["hardest_cases"][test_key] = 0
                stats["test_case_analysis"]["hardest_cases"][test_key] += 1

    def _calculate_summary_stats(self, stats: Dict[str, Any], all_solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics with proper division by zero handling.
        """
        summary = {}

        # Basic metrics
        total_tests = stats["tests_passed"] + stats["tests_failed"] + stats["test_timeouts"] + stats["import_errors"] + \
                      stats["test_errors"]
        total_candidates = max(stats["candidates_generated"], 1)
        total_explored = max(stats["nodes_explored"], 1)

        summary["efficiency"] = {
            "solving_rate": stats["tests_passed"] / total_candidates,
            "branch_success_rate": len([s for s in all_solutions if s["passed"]]) / total_explored,
            "test_success_rate": stats["tests_passed"] / max(total_tests, 1)
        }

        # Test result breakdown
        summary["test_results"] = {
            "total": total_tests,
            "passed": stats["tests_passed"],
            "failed": stats["tests_failed"],
            "timeouts": stats["test_timeouts"],
            "import_errors": stats["import_errors"],
            "other_errors": stats["test_errors"],
            "unique_import_failures": list(stats["unique_import_failures"])
        }

        # Error recovery metrics
        if "feedback_impact" in stats and "error_types" in stats["feedback_impact"]:
            error_types = stats["feedback_impact"]["error_types"]

            total_attempts = 0
            total_improvements = 0

            for error_type, data in error_types.items():
                total_attempts += data.get("attempts", 0)
                total_improvements += data.get("improvements", 0)

            recovery_rate = total_improvements / total_attempts if total_attempts > 0 else 0

            summary["error_recovery"] = {
                "total_attempts": total_attempts,
                "total_improvements": total_improvements,
                "recovery_rate": recovery_rate
            }

            # Get top 5 most common errors
            if error_types:
                top_errors = sorted(
                    [(et, data["attempts"]) for et, data in error_types.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                summary["top_errors"] = top_errors

        # Test case analysis
        if stats["test_case_analysis"]["hardest_cases"]:
            top_hardest = sorted(
                stats["test_case_analysis"]["hardest_cases"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            summary["hardest_test_cases"] = top_hardest

        # Termination reasons
        if "termination_reasons" in stats:
            summary["termination_reasons"] = stats["termination_reasons"]

        # Import failure analysis
        if stats["import_errors"] > 0:
            summary["import_analysis"] = {
                "total_import_errors": stats["import_errors"],
                "branches_terminated_by_imports": stats["import_terminated_branches"],
                "unique_missing_modules": len(stats["unique_import_failures"]),
                "missing_modules": list(stats["unique_import_failures"])
            }

        return summary

    def _generate_improved_candidates_for_node(
            self,
            problem_data: Dict[str, Any],
            failed_node: Dict[str, Any],
            num_candidates: int
    ) -> List[str]:
        """
        Generate improved candidates based on a specific failed node.

        Args:
            problem_data: Problem data dictionary
            failed_node: The specific failed node to improve from
            num_candidates: Number of candidates to generate

        Returns:
            List of improved solution candidates
        """
        logger.info(f"Generating {num_candidates} improved candidates for node {failed_node['node_id']}")
        model = self._get_model()
        candidates = []

        for i in range(num_candidates):
            self._run_memory_cleanup()

            # Create a targeted improvement prompt
            prompt = self._create_targeted_improvement_prompt(
                problem_data,
                failed_node,
                i + 1
            )

            # Generate improved solution
            response = model.generate(prompt)

            # Extract solution code
            solution_code = self._extract_solution_code(response)

            if solution_code:
                candidates.append(solution_code)
                logger.info(f"Generated improved candidate {i + 1}/{num_candidates} ({len(solution_code)} chars)")
            else:
                logger.warning(f"Failed to extract solution code for improved candidate {i + 1}")

        return candidates

    def _create_targeted_improvement_prompt(
            self,
            problem_data: Dict[str, Any],
            failed_node: Dict[str, Any],
            candidate_num: int
    ) -> str:
        """
        Create a prompt specifically targeting the failures of a particular node.
        """
        test_result = failed_node["test_result"]
        error_msg = test_result.get("error_message", "Unknown error")

        # Build specific feedback
        feedback_parts = [f"Previous solution failed: {error_msg}"]

        if "failed_tests" in test_result:
            for i, test in enumerate(test_result["failed_tests"][:3]):  # Show up to 3 failed tests
                feedback_parts.append(
                    f"Failed Test {i + 1}: Input={test.get('input')}, "
                    f"Expected={test.get('expected')}, Got={test.get('actual')}"
                )

        problem_desc = problem_data['query']
        import_pr = problem_data['prompt']

        prompt = f"""
        You are an expert Python programmer solving a LeetCode problem. Follow my instructions precisely.

        # Problem Statement
        Problem ID: {problem_data['problem_id']}  
        Title: {problem_data['problem_title']}
        Difficulty: {problem_data['difficulty']}

        # Problem Description        
        {problem_desc}
        
        # Available Imports and initial code
        {import_pr}
        
        # CRITICAL: DO NOT use any imports other than those listed above!

        # Required Function Signature
        Entry Point: {problem_data['entry_point']}

        # Previous Failed Solution
        ```python
        {failed_node['solution']}
        ```

        # Test Feedback
        {chr(10).join(feedback_parts)}

        # Task
        Generate improved candidate solution #{candidate_num} that fixes the specific issues above.

        # REQUIRED RESPONSE FORMAT
        You must follow this exact format:

        1. Start with "## Issue Analysis"
        2. Analyze what went wrong (2-3 sentences)
        3. Then write "## Improved Solution"
        4. Describe your fix (2-3 sentences)
        5. Then write EXACTLY "## Code Solution"
        6. Provide your Python solution in a code block

        IMPORTANT: Make sure your solution handles the failed test cases correctly.
        """

        return prompt

    def _check_tree_for_solution(self, node: Dict[str, Any], all_solutions: List[Dict[str, Any]] = None) -> bool:
        """
        Recursively check if any node in the tree has a passing solution.

        Args:
            node: The current node to check
            all_solutions: List of all solution nodes (used for ID lookups)

        Returns:
            True if a passing solution is found, False otherwise
        """
        # Double-check the passed status based on the actual test result
        node_passed = False
        if "test_result" in node:
            test_status = node["test_result"].get("status", "")
            node_passed = test_status == "pass"
            # Update the node's passed flag if it's incorrect
            if node_passed != node.get("passed", False):
                logger.warning(f"Fixed inconsistent pass status for node {node.get('node_id', 'unknown')}")
                node["passed"] = node_passed
        else:
            node_passed = node.get("passed", False)

        if node_passed:
            logger.debug(f"Found passing solution in node {node.get('node_id', 'unknown')}")
            return True

        # If this is the first call, extract all_solutions from the node's context
        if all_solutions is None:
            frame = inspect.currentframe().f_back
            all_solutions = frame.f_locals.get('all_solutions', [])
            if not all_solutions:
                logger.warning("No solutions found in context when checking tree")
                return False

        # Check each child node by ID
        for child_id in node.get("children", []):
            # Find the child node by ID
            child_node = next((n for n in all_solutions if n["node_id"] == child_id), None)
            if child_node:
                if self._check_tree_for_solution(child_node, all_solutions):
                    return True

        return False

    def _calculate_tree_depth(self, solution_tree: List[Dict[str, Any]], all_solutions: List[Dict[str, Any]]) -> int:
        """Calculate the maximum depth of the solution tree."""
        max_depth = 0

        def traverse(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)

            # Get all children by looking up their IDs in all_solutions
            for child_id in node.get("children", []):
                child_node = next((n for n in all_solutions if n["node_id"] == child_id), None)
                if child_node:
                    traverse(child_node, depth + 1)

        # Start traversal from root nodes in solution_tree
        for root_node in solution_tree:
            traverse(root_node)

        return max_depth

    # Keep all other existing methods unchanged
    def _generate_initial_candidates(self, problem_data: Dict[str, Any]) -> List[str]:
        """Generate initial solution candidates."""
        logger.info(f"Generating {self.initial_k} initial solution candidates")
        model = self._get_model()
        candidates = []

        for i in range(self.initial_k):
            self._run_memory_cleanup()
            prompt = self._create_cot_prompt(problem_data, i + 1)
            response = model.generate(prompt)
            solution_code = self._extract_solution_code(response)

            if solution_code:
                candidates.append(solution_code)
                logger.info(f"Generated initial candidate {i + 1}/{self.initial_k} ({len(solution_code)} chars)")
            else:
                logger.warning(f"Failed to extract solution code for candidate {i + 1}")

        return candidates

    def _generate_improved_candidates(
            self,
            problem_data: Dict[str, Any],
            previous_candidates: List[Dict[str, Any]],
            reflection_logs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate improved candidates through self-reflection.

        Args:
            problem_data: Problem data dictionary.
            previous_candidates: Previous solution candidates with test results.
            reflection_logs: Logs from previous reflection rounds.

        Returns:
            List of improved solution code candidates.
        """
        logger.info(f"Generating {self.num_candidates} improved solution candidates")
        model = self._get_model()
        candidates = []

        # Filter recent candidates (from last round)
        last_round = reflection_logs[-1]["round"]
        recent_candidates = [c for c in previous_candidates if c["candidate_id"].startswith(f"{last_round}_")]

        for i in range(self.num_candidates):
            self._run_memory_cleanup()

            # Create a prompt with self-reflection
            prompt = self._create_reflection_prompt(problem_data, recent_candidates, reflection_logs, i + 1)

            # Log the prompt for debugging (truncate if too long)
            logger.debug(f"Improved candidate {i + 1} prompt: {prompt}")

            # Generate improved solution
            response = model.generate(prompt)

            # Log the full response for debugging
            logger.debug(f"Improved candidate {i + 1} full response: {response}")

            # Extract solution code
            solution_code = self._extract_solution_code(response)

            if solution_code:
                candidates.append(solution_code)
                logger.info(f"Generated improved candidate {i + 1}/{self.num_candidates} ({len(solution_code)} chars)")
            else:
                # Log more details about the failure
                logger.warning(f"Failed to extract solution code for improved candidate {i + 1}")
                logger.warning(f"Response excerpt: {response[:500]}...{response[-500:] if len(response) > 500 else ''}")

        return candidates

    def _create_cot_prompt(self, problem_data: Dict[str, Any], candidate_num: int) -> str:
        """
        Create a Chain of Thought prompt for initial solution generation with strict formatting.
        """
        # Clean up the problem description if needed
        imports_desc = problem_data['prompt']
        problem_desc = problem_data['query']

        prompt = f"""
        You are an expert Python programmer solving a LeetCode problem. Follow my instructions precisely.

        # Problem Statement
        Problem ID: {problem_data['problem_id']}  
        Title: {problem_data['problem_title']}
        Difficulty: {problem_data['difficulty']}
        Tags: {', '.join(problem_data['tags'])}

        # Query Description
        {problem_desc}

        # Imports and initial code
        {imports_desc}

        # Required Function Signature
        Entry Point: {problem_data['entry_point']}

        # Task
        You are generating candidate solution #{candidate_num}. Your solution will be tested against the test cases.

        # CRITICAL IMPORT RESTRICTION
        You MUST ONLY use the imports provided above. DO NOT use any additional imports like:
        - sortedcontainers, etc
        - Any other external libraries not shown above
        If you need a data structure, implement it yourself or use the standard library imports provided.


        # REQUIRED RESPONSE FORMAT
        You must follow this exact format:

        1. Start with "## Problem Analysis"
        2. Briefly analyze the problem (2-3 sentences)
        3. Then write "## Solution Approach" 
        4. Briefly describe your approach (2-3 sentences)
        5. Then write EXACTLY "## Code Solution"
        6. Provide your Python solution in a code block that starts with ```python and ends with ```
        7. The *very last* characters in your reply **must** be the closing ``` fence. Anything after that will be 
        ignored and may cause automatic failure.

        EXAMPLE OUTPUT FORMAT:
        ## Problem Analysis
        [Brief analysis]

        ## Solution Approach
        [Brief approach description]

        ## Code Solution
        ```python
        class Solution:
            def method(self, params):
                # Your solution here
                return result
        ```

        IMPORTANT: 
        - **Strict grader**: Replies not in the format above will be discarded without scoring.
        - Wrap code in a fence (```python` … ```), nothing else.  
        - Follow signature *exactly*: `{problem_data['entry_point']}`  
        - Handle edge cases.
        - Avoid indentation problems.
        """

        return prompt

    def _create_reflection_prompt(
            self,
            problem_data: Dict[str, Any],
            previous_candidates: List[Dict[str, Any]],
            reflection_logs: List[Dict[str, Any]],
            candidate_num: int
    ) -> str:
        """
        Create a self-reflection prompt with strict output formatting.
        """
        # Extract the test results from previous candidates
        test_feedback = []

        for candidate in previous_candidates:
            result = candidate["test_result"]
            if result.get("status") == "pass":
                test_feedback.append(f"Candidate {candidate['candidate_id']} PASSED all tests.")
            else:
                error_msg = result.get("error_message", "Unknown error")
                test_feedback.append(f"Candidate {candidate['candidate_id']} FAILED tests: {error_msg}")
                # Include failed test cases if available
                if "failed_tests" in result:
                    for i, test in enumerate(result["failed_tests"][:3]):  # Limit to 3 for brevity
                        test_feedback.append(
                            f"  Failed Test {i + 1}: Input={test.get('input')}, Expected={test.get('expected')}, Got={test.get('actual')}")

        # Include code examples from previous candidates
        code_examples = []
        passing_examples = [c for c in previous_candidates if c["test_result"].get("status") == "pass"]
        failing_examples = [c for c in previous_candidates if c["test_result"].get("status") != "pass"]

        # Add up to 1 passing example if available
        if passing_examples:
            example = passing_examples[0]
            code_examples.append(
                f"### Passing Example (Candidate {example['candidate_id']})\n```python\n{example['solution']}\n```")

        # Add up to 2 failing examples if available
        for i, example in enumerate(failing_examples[:2]):
            code_examples.append(
                f"### Failing Example (Candidate {example['candidate_id']})\n```python\n{example['solution']}\n```")

        # Get imports from problem data
        imports_desc = problem_data['prompt']
        problem_desc = problem_data['query']

        # Construct the prompt
        prompt = f"""
        You are an expert Python programmer solving a LeetCode problem. Follow my instructions precisely.

        # Problem Statement
        Problem ID: {problem_data['problem_id']}  
        Title: {problem_data['problem_title']}
        Difficulty: {problem_data['difficulty']}
        Tags: {', '.join(problem_data['tags'])}

        # Problem Description
        {problem_desc}

        # Available Imports and initial code
        {imports_desc}

        # CRITICAL IMPORT RESTRICTION
        DO NOT use any imports beyond what's provided above. Common mistakes to avoid:
        - No additional imports even from standard library unless shown above
        - If you need a data structure, implement it yourself

        If you see import errors in the test feedback, you MUST rewrite without those imports!

        # Required Function Signature
        Entry Point: {problem_data['entry_point']}

        # Test Feedback 
        {chr(10).join(test_feedback)}

        # Previous Solutions
        {chr(10).join(code_examples)}

        # Task
        You are generating improved candidate solution #{candidate_num}. Fix the issues in previous solutions.
        If any solution failed with import errors, rewrite it using ONLY the available imports listed above.

        # REQUIRED RESPONSE FORMAT
        You must follow this exact format:

        1. Start with "## Issue Analysis"
        2. Briefly analyze what went wrong in previous solutions (2-3 sentences)
        3. Then write "## Improved Solution" 
        4. Briefly describe your improved approach (2-3 sentences)
        5. Then write EXACTLY "## Code Solution"
        6. Provide your Python solution in a code block that starts with ```python and ends with ```
        7. Do not include any other text or explanations after the code block

        EXAMPLE OUTPUT FORMAT:
        ## Issue Analysis
        [Brief analysis of issues]

        ## Improved Solution
        [Brief improved approach description]

        ## Code Solution
        ```python
        class Solution:
            def method(self, params):
                # Your improved solution here
                return result
        ```

        IMPORTANT: 
        - Your solution code MUST be wrapped in ```python and ``` tags exactly as shown
        - ONLY include the actual code between these tags
        - Use ONLY the imports from "Available Imports" section
        - Make your code handle all edge cases
        - Follow the required function signature exactly: {problem_data['entry_point']}
        - Avoid indentation problems.
        """

        return prompt

    def _extract_solution_code(self, response: str) -> str:
        """
        Robustly extract the candidate solution code from a model response.
        Strategy:
        1. Slice the response at the first "## Code Solution" header (case-insensitive).
           If the header is missing we fallback to the whole response.
        2. Search that slice for all fenced code blocks whose opening fence starts
           with ``` or ```python / ```py (language tag optional).
           - We take the last fenced block - this is almost always the fresh answer,
             because previous reflection rounds add earlier blocks above.
        3. If no closed fence exists but an opening fence is found, treat everything
           after the opening fence as the code (handles missing closing fence).
        4. Fallback: locate a `class Solution` definition and grab everything until the
           next blank line or EOF.
        5. Final strip and return an empty string on total failure.
        """
        import re
        # Focus on the region after the dedicated header, if present.
        header_regex = re.compile(r'##\s*Code Solution', re.IGNORECASE)
        header_match = header_regex.search(response)
        chunk = response[header_match.end():] if header_match else response

        # Grab the last well-formed fenced code block (language tag optional).
        fence_regex = re.compile(
            r'```(?:python|py)?\s*([\s\S]*?)```',  # non-greedy
            flags=re.IGNORECASE
        )
        fenced_blocks = fence_regex.findall(chunk)
        if fenced_blocks:
            return fenced_blocks[-1].strip()

        # Handle an unterminated fence - take everything after the opening fence.
        open_fence = re.search(r'```(?:python|py)?\s*([\s\S]*)$', chunk, flags=re.IGNORECASE)
        if open_fence:
            return open_fence.group(1).strip()

        # Fallback: extract from a `class Solution` definition onward.
        class_match = re.search(r'class\s+Solution[\s\S]+', chunk)
        if class_match:
            return class_match.group(0).strip()

        # Nothing found.
        logger.warning("Could not extract solution code from response")
        return ""

    def _calculate_solution_hash(self, solution: str) -> str:
        """
        Calculate a hash for the solution to detect duplicates.

        Args:
            solution: Solution code string.

        Returns:
            Solution hash.
        """
        # Normalize the solution to ignore whitespace and comments
        normalized_solution = self._normalize_solution(solution)

        # Calculate hash
        return hashlib.sha256(normalized_solution.encode('utf-8')).hexdigest()

    def _normalize_solution(self, solution: str) -> str:
        """
        Normalize a solution to ignore insignificant differences.

        Args:
            solution: Original solution string.

        Returns:
            Normalized solution string.
        """
        import re

        # Remove comments
        solution = re.sub(r'#.*$', '', solution, flags=re.MULTILINE)

        # Remove empty lines and leading/trailing whitespace
        lines = [line.strip() for line in solution.splitlines() if line.strip()]

        # Join lines with a single space
        return ' '.join(lines)

    def _test_solution(self, problem_data: Dict[str, Any], solution: str, stats: Dict[str, Any] = None) -> Dict[
        str, Any]:
        """
        Test a solution against the test cases.
        Now accepts stats to update import failures coherently.
        """
        # Log the complete solution being tested
        logger.debug(f"TESTING SOLUTION:\n{solution}")
        # Log the test code
        logger.debug(f"TEST CODE:\n{problem_data.get('test', 'No test found')}")
        # Log the entry point
        logger.debug(f"ENTRY POINT: {problem_data.get('entry_point', 'No entry point found')}")

        # Fix any indentation issues in the solution code
        solution = self._fix_indentation(solution)

        # Calculate a hash for solution caching
        solution_hash = self._calculate_solution_hash(solution)

        # Check if we've already tested this exact solution
        if solution_hash in self.solution_cache:
            logger.debug(f"Using cached result for solution hash {solution_hash[:8]}")
            return self.solution_cache[solution_hash]

        # Run the test using the test runner (which handles environment management)
        result = self.test_runner.run_tests(problem_data, solution)

        # Track import failures globally and in stats if provided
        if result.get("status") == "import_error":
            logger.warning(f"Solution cannot be tested due to import errors: {result.get('error_message')}")

            for failed_import in result.get("import_failures", []):
                self.global_import_failures.add(failed_import)
                if stats:
                    stats["unique_import_failures"].add(failed_import)

        # Cache the result
        self.solution_cache[solution_hash] = result

        logger.debug(f"TEST RESULT: {json.dumps(result, indent=2)}")
        return result

    def _fix_indentation(self, code: str) -> str:
        """
        Fix indentation issues in Python code.
        - Ensures module-level code has zero indentation
        - Properly handles class and function definitions
        - Correctly handles import statements

        Args:
            code: The Python code string to fix

        Returns:
            The code with fixed indentation
        """
        if not code or code.isspace():
            return code

        # Split the code into lines
        lines = code.split('\n')

        # First, detect minimum indentation level for imports and class definitions
        min_indent = float('inf')
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):  # Skip empty lines and comments
                indent = len(line) - len(line.lstrip())
                if line.lstrip().startswith(('import ', 'from ', 'class ', 'def ')):
                    min_indent = min(min_indent, indent)

        # If no valid lines found, default to 0
        if min_indent == float('inf'):
            min_indent = 0

        # Process each line to ensure proper indentation
        fixed_lines = []
        current_block_indent = 0
        in_class_or_function = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue

            # Get current indentation
            current_indent = len(line) - len(line.lstrip())

            # Check for class/function definitions or module-level imports
            if stripped.startswith(('class ', 'def ')):
                # Start of new class or function
                if current_indent == min_indent:
                    # This is a top-level class or function
                    in_class_or_function = True
                    current_block_indent = current_indent
                    fixed_lines.append(stripped)  # No indentation for class/function definitions
                else:
                    # This is a nested class or function, preserve relative indentation
                    relative_indent = current_indent - min_indent
                    fixed_lines.append(' ' * relative_indent + stripped)
                continue

            # Check if line is an import or module-level statement
            if current_indent == min_indent and stripped.startswith(('import ', 'from ')):
                fixed_lines.append(stripped)  # No indentation for imports
                continue

            # For other lines, adjust indentation relative to the min_indent
            if current_indent >= min_indent:
                relative_indent = current_indent - min_indent
                fixed_lines.append(' ' * relative_indent + stripped)
            else:
                # If indentation is less than min_indent, treat as module level
                fixed_lines.append(stripped)

        # Reassemble the code
        fixed_code = '\n'.join(fixed_lines)
        return fixed_code

    def _measure_solution_diversity(self, all_solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Measure diversity among generated solutions using the SolutionDiversityAnalyzer.

        Args:
            all_solutions: List of solution nodes with solution code and hashes

        Returns:
            Dictionary with diversity metrics
        """
        if not all_solutions:
            return {
                "unique_solutions": 0,
                "similarity_score": 0.0,
                "solution_lengths": {"min": 0, "max": 0, "avg": 0.0}
            }

        # Extract solution codes and hashes
        solution_codes = [s["solution"] for s in all_solutions]
        solution_hashes = [s["solution_hash"] for s in all_solutions]

        # Initialize and use the diversity analyzer
        try:
            analyzer = SolutionDiversityAnalyzer()
            diversity_metrics = analyzer.analyze_diversity(solution_codes, solution_hashes)

            logger.info(f"Analyzed diversity across {len(solution_codes)} solutions: "
                        f"similarity_score={diversity_metrics['similarity_score']:.2f}, "
                        f"feature_diversity={diversity_metrics['feature_diversity']:.2f}")

            return diversity_metrics
        except Exception as e:
            # Fallback to simple hash-based diversity if analyzer fails
            logger.error(f"Error in diversity analysis: {str(e)}")

            # Count unique solution hashes
            unique_hashes = set(solution_hashes)

            # Calculate solution length statistics
            solution_lengths = [len(s["solution"]) for s in all_solutions]
            min_length = min(solution_lengths) if solution_lengths else 0
            max_length = max(solution_lengths) if solution_lengths else 0
            avg_length = sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0

            return {
                "unique_solutions": len(unique_hashes),
                "unique_ratio": len(unique_hashes) / len(all_solutions) if all_solutions else 0.0,
                "similarity_score": 0.0,  # Cannot calculate without analyzer
                "solution_lengths": {
                    "min": min_length,
                    "max": max_length,
                    "avg": avg_length
                }
            }

    def _save_results(self, problem_id: str, result: Dict[str, Any]) -> None:
        """
        Save the results to a file.

        Args:
            problem_id: Problem ID.
            result: Result dictionary.
        """
        # Create result directory if it doesn't exist
        result_dir = self.results_dir / "leetcode_solutions"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        result_file = result_dir / f"{problem_id}.json"

        try:
            # Ensure all sets are converted to lists before saving
            import json

            # Custom JSON encoder to handle sets
            class SetEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, set):
                        return list(obj)
                    return json.JSONEncoder.default(self, obj)

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, cls=SetEncoder)

            logger.info(f"Saved results to {result_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            # Try to identify what caused the serialization error
            try:
                # Attempt to find non-serializable objects
                import json
                json.dumps(result)  # This will raise an error with details
            except TypeError as te:
                logger.error(f"Serialization error details: {te}")

    def evaluate_by_round(self, problem_data: Dict[str, Any], reflection_logs: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate solutions round by round to see improvement over iterations.

        Args:
            problem_data: Problem data dictionary
            reflection_logs: Logs from all reflection rounds

        Returns:
            Dictionary with round-by-round evaluation results
        """
        if self.code_evaluator is None:
            return {"error": "code_eval not enabled"}

        round_evaluations = []
        cumulative_solutions = []

        for round_log in reflection_logs:
            # Add this round's solutions to cumulative list
            round_solutions = [c["solution"] for c in round_log["candidates"]]
            cumulative_solutions.extend(round_solutions)

            # Evaluate solutions up to this round
            round_eval = self.code_evaluator.evaluate_solutions(
                problem_data,
                cumulative_solutions.copy()
            )

            round_evaluations.append({
                "round": round_log["round"],
                "solutions_count": len(cumulative_solutions),
                "pass_at_k": round_eval.get("pass_at_k", {}),
                "error": round_eval.get("error", None)
            })

        return {
            "round_evaluations": round_evaluations,
            "final_pass_at_k": round_evaluations[-1]["pass_at_k"] if round_evaluations else {}
        }

    def _categorize_error(self, error_message):
        """
        Categorize error messages to track feedback effectiveness.

        Args:
            error_message: The error message string from test results

        Returns:
            String representing the error category
        """
        if not error_message:
            return "unknown"

        # Import related errors
        if "ModuleNotFoundError" in error_message or "ImportError" in error_message:
            return "import_error"
        elif "No module named" in error_message:
            return "missing_module"

        # Common assertion and runtime errors
        elif "AssertionError" in error_message:
            return "assertion_failure"
        elif "IndexError" in error_message:
            return "index_error"
        elif "TypeError" in error_message:
            return "type_error"
        elif "ValueError" in error_message:
            return "value_error"
        elif "KeyError" in error_message:
            return "key_error"
        elif "AttributeError" in error_message:
            return "attribute_error"
        elif "ZeroDivisionError" in error_message:
            return "zero_division_error"
        elif "NameError" in error_message:
            return "name_error"
        elif "SyntaxError" in error_message:
            return "syntax_error"
        elif "RecursionError" in error_message or "maximum recursion depth" in error_message:
            return "recursion_error"
        elif "MemoryError" in error_message or "out of memory" in error_message:
            return "memory_error"
        elif "RuntimeError" in error_message:
            return "runtime_error"
        # Catch-all for other errors
        else:
            return "other_error"
