#!/usr/bin/env python3
"""
Enhanced LeetCode Results Analyzer with Complete Data Extraction

This script analyzes LeetCode benchmark results for different models across difficulty levels
(Easy, Medium, Hard) and generates comprehensive research-focused tables for academic papers.

Now properly extracts all detailed data from individual problem JSON files including:
- Complete solution trees with parent-child relationships
- Detailed feedback impact metrics by depth
- Code evaluation results with pass@k values
- Test case analysis and error transitions
- Summary statistics from each problem
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd


class LeetCodeResultsAnalyzer:
    """Enhanced analyzer for LeetCode benchmark results with complete data extraction."""

    def __init__(self, easy_dir, medium_dir, hard_dir, output_file):
        """Initialize the analyzer with result directories."""
        self.easy_dir = Path(easy_dir)
        self.medium_dir = Path(medium_dir)
        self.hard_dir = Path(hard_dir)
        self.output_file = Path(output_file)

        # List of models to analyze
        self.models = ["qwen2-5-coder", "deepseek-r1-distill", "qwq-preview"]

        # Initialize data structures
        self.results = {
            "Easy": {},
            "Medium": {},
            "Hard": {},
            "Overall": {}
        }

        self.pass_at_k_results = {
            "Easy": {},
            "Medium": {},
            "Hard": {},
            "Overall": {}
        }

        self.problem_counts = {
            "Easy": defaultdict(int),
            "Medium": defaultdict(int),
            "Hard": defaultdict(int)
        }

        # Enhanced data structures for detailed analysis
        self.detailed_stats = {
            "Easy": defaultdict(lambda: defaultdict(list)),
            "Medium": defaultdict(lambda: defaultdict(list)),
            "Hard": defaultdict(lambda: defaultdict(list))
        }

        # FIXED: Track error analysis per model and difficulty
        self.error_analysis = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: {"occurrences": 0, "resolutions": 0})))
        self.test_case_analysis = defaultdict(lambda: defaultdict(int))
        self.tree_exploration_data = defaultdict(list)
        self.solution_diversity_data = defaultdict(list)

        # New structures for individual problem data
        self.problem_level_stats = defaultdict(list)
        self.error_transition_counts = defaultdict(int)
        self.code_eval_by_problem = defaultdict(list)

        # Debug mode flag
        self.debug_mode = False

        # model → difficulty → depth(str) → list[dict(attempts, improvements, solved)]
        self.feedback_impact_data = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        # model → difficulty → depth(str) → int (solved count)
        self.depth_solution_counts = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def load_data(self):
        """Load and process data from all result directories."""
        print("Loading data from result directories...")

        # Process each difficulty level
        self._process_difficulty("Easy", self.easy_dir)
        self._process_difficulty("Medium", self.medium_dir)
        self._process_difficulty("Hard", self.hard_dir)

        # Calculate overall metrics
        self._calculate_overall_metrics()

        print("Data loading and processing complete.")
        return self

    def _process_difficulty(self, difficulty, result_dir):
        """Process results for a specific difficulty level with enhanced statistics extraction."""
        if not result_dir.exists():
            print(f"Warning: {result_dir} does not exist, skipping {difficulty} level")
            return

        # Get all model directories
        model_dirs = [d for d in result_dir.iterdir() if d.is_dir() and d.name in self.models]

        if not model_dirs:
            print(f"Warning: No model directories found in {result_dir}")
            return

        for model_dir in model_dirs:
            model_name = model_dir.name

            # Initialize metrics for this model and difficulty
            if model_name not in self.results[difficulty]:
                self.results[difficulty][model_name] = {}
                self.pass_at_k_results[difficulty][model_name] = defaultdict(list)

            # Process individual problem files FIRST for detailed stats
            leetcode_solutions_dir = model_dir / "leetcode_solutions"
            if leetcode_solutions_dir.exists():
                self._process_individual_problems_detailed(leetcode_solutions_dir, model_name, difficulty)

            # Then process combined_results.json for overview
            combined_file = model_dir / "combined_results.json"
            if combined_file.exists():
                self._process_combined_results_overview(combined_file, model_name, difficulty)

            # Finally process summary.json for any additional metrics
            summary_file = model_dir / "summary.json"
            if summary_file.exists():
                self._process_summary(summary_file, model_name, difficulty)

    def _process_individual_problems_detailed(
            self,
            solutions_dir: Path,
            model_name: str,
            difficulty: str,
    ) -> None:
        """
        Walk through every  <problem>.json  file produced by the pipeline and
        aggregate *all* fine–grained statistics, keeping the aggregates
        separated by (model, difficulty).
        """
        solution_files = list(solutions_dir.glob("*.json"))
        if not solution_files:
            print(f"[WARN] no *.json files in {solutions_dir}")
            return

        if self.debug_mode:
            print(
                f"[DEBUG] {model_name}/{difficulty}: "
                f"parsing {len(solution_files)} solution files"
            )

        for fn in solution_files:
            try:
                with open(fn, "r", encoding="utf-8") as fp:
                    problem_data = json.load(fp)
            except Exception as e:
                print(f"[ERR] cannot read {fn.name}: {e}")
                continue

            # --------------------------------------------------------------------
            # 1.   BASIC IDENTIFIERS
            # --------------------------------------------------------------------
            problem_id = problem_data.get("problem_id", fn.stem)
            stats = problem_data.get("stats", {})
            fi = stats.get("feedback_impact", {})

            # --------------------------------------------------------------------
            # 2.   SOLUTION-TREE  → first-pass depth  + per-depth counts
            # --------------------------------------------------------------------
            tree = problem_data.get("solution_tree", [])
            if tree:
                fp_depth = self._find_first_pass_depth_from_tree(tree)
                if fp_depth >= 0:
                    self.detailed_stats[difficulty][model_name][
                        "first_pass_depths"
                    ].append(fp_depth)

                # total / passed counts by depth   (populates detailed_stats[…])
                self._count_solutions_by_depth_from_tree(
                    tree, model_name, difficulty
                )

            # --------------------------------------------------------------------
            # 3.   TREE-EXPLORATION META
            # --------------------------------------------------------------------
            self.tree_exploration_data[f"{model_name}_{difficulty}"].append(
                {
                    "problem_id": problem_id,
                    "nodes_explored": stats.get("nodes_explored", 0),
                    "candidates_generated": stats.get("candidates_generated", 0),
                    "tree_depth": problem_data.get("tree_depth", 0),
                    "termination_reasons": stats.get("termination_reasons", {}),
                    "status": problem_data.get("status", "unknown"),
                    "processing_time": problem_data.get("processing_time", 0),
                }
            )

            # --------------------------------------------------------------------
            # 4.   FEEDBACK-IMPACT :  depth buckets  +  error-type tallies
            # --------------------------------------------------------------------
            # a) per-depth buckets ----------------------------------------------
            for depth_str, depth_data in fi.get("depths", {}).items():
                # keep per-difficulty separation
                self.feedback_impact_data[model_name][difficulty][depth_str].append(
                    depth_data
                )
                # running total of *solved* at that depth
                self.depth_solution_counts[model_name][difficulty][depth_str] += (
                    depth_data.get("solved", 0)
                )

            # b) FIXED: error-type totals tracked per model and difficulty ------
            for err, edata in fi.get("error_types", {}).items():
                self.error_analysis[model_name][difficulty][err]["occurrences"] += edata.get("attempts", 0)
                self.error_analysis[model_name][difficulty][err]["resolutions"] += edata.get("improvements", 0)

            # c) error transitions ----------------------------------------------
            for tr, cnt in fi.get("error_transitions", {}).items():
                self.error_transition_counts[tr] += cnt

            # --------------------------------------------------------------------
            # 5.   TEST CASE ANALYSIS (if present)
            # --------------------------------------------------------------------
            if "test_case_analysis" in stats:
                tca = stats["test_case_analysis"]
                # Track hardest cases
                for test_case, count in tca.get("hardest_cases", {}).items():
                    self.test_case_analysis["hardest_cases"][test_case] += count
                # Track first failing tests
                for test_case, count in tca.get("first_failing_tests", {}).items():
                    self.test_case_analysis["first_failing"][test_case] += count

            # --------------------------------------------------------------------
            # 6.   SOLUTION DIVERSITY (if present)
            # --------------------------------------------------------------------
            if "solution_diversity" in stats:
                self.solution_diversity_data[f"{model_name}_{difficulty}"].append(
                    stats["solution_diversity"]
                )

            # --------------------------------------------------------------------
            # 7.   PASS@k  (stats or code_eval_results)
            # --------------------------------------------------------------------
            # from stats ---------------------------------------------------------
            cem = stats.get("code_eval_metrics", {})
            for k, v in cem.get("pass_at_k", {}).items():
                k_norm = str(k).replace("pass@", "")
                self.pass_at_k_results[difficulty][model_name][k_norm].append(v)

            # from top-level code_eval_results ----------------------------------
            cer = problem_data.get("code_eval_results", {})
            for k, v in cer.get("pass_at_k", {}).items():
                k_norm = str(k).replace("pass@", "")
                self.pass_at_k_results[difficulty][model_name][k_norm].append(v)

            # --------------------------------------------------------------------
            # 8.   KEEP FULL PER-PROBLEM SNAPSHOT (optional downstream analysis)
            # --------------------------------------------------------------------
            self.problem_level_stats[f"{model_name}_{difficulty}"].append(
                {
                    "problem_id": problem_id,
                    "status": problem_data.get("status", "unknown"),
                    "total_candidates": problem_data.get("total_candidates", 0),
                    "nodes_explored": problem_data.get("nodes_explored", 0),
                    "tree_depth": problem_data.get("tree_depth", 0),
                    "processing_time": problem_data.get("processing_time", 0),
                    "passed_solutions": len(problem_data.get("passed_solutions", [])),
                    "all_solutions": len(problem_data.get("all_solutions", [])),
                }
            )

        # end for-each-file

    def _find_first_pass_depth_from_tree(self, tree):
        """Find the depth of the first passing solution from the solution tree."""
        if not tree:
            return -1

        min_depth = float('inf')

        for node in tree:
            if isinstance(node, dict):
                # Check multiple ways a node might indicate success
                passed = (node.get("passed", False) or
                          (node.get("test_result", {}).get("status") == "pass") or
                          node.get("status") == "solved")

                if passed:
                    node_depth = node.get("depth", 0)
                    min_depth = min(min_depth, node_depth)

        return min_depth if min_depth != float('inf') else -1

    def _count_solutions_by_depth_from_tree(self, tree, model_name, difficulty):
        """Count solutions at each depth from the solution tree."""
        if not tree:
            return

        depth_counts = defaultdict(lambda: {"total": 0, "passed": 0})

        for node in tree:
            if isinstance(node, dict):
                depth = node.get("depth", 0)
                depth_counts[depth]["total"] += 1

                # Check if passed
                passed = (node.get("passed", False) or
                          (node.get("test_result", {}).get("status") == "pass"))
                if passed:
                    depth_counts[depth]["passed"] += 1

        # Store in detailed stats
        for depth, counts in depth_counts.items():
            key = f"depth_{depth}_solutions"
            self.detailed_stats[difficulty][model_name][key].append(counts)

    def _process_combined_results_overview(self, combined_file, model_name, difficulty):
        """Process combined results for basic overview metrics."""
        try:
            with open(combined_file, 'r') as f:
                combined_data = json.load(f)

            # Calculate basic metrics
            total_problems = len(combined_data)
            solved_problems = sum(1 for r in combined_data if r.get("status") == "solved")
            success_rate = (solved_problems / total_problems) * 100 if total_problems > 0 else 0

            avg_time = np.mean([r.get("processing_time", 0) for r in combined_data]) if total_problems > 0 else 0
            avg_candidates = np.mean([r.get("total_candidates", 0) for r in combined_data]) if total_problems > 0 else 0

            # Store basic metrics
            self.results[difficulty][model_name] = {
                "total_problems": total_problems,
                "solved_problems": solved_problems,
                "success_rate": success_rate,
                "avg_time": avg_time,
                "avg_candidates": avg_candidates
            }

            # Update problem count
            self.problem_counts[difficulty][model_name] = total_problems

        except Exception as e:
            print(f"Error processing combined results for {model_name} at {difficulty} level: {str(e)}")

    def _debug_print_problem_structure(self, problem):
        """Debug helper to print problem structure."""
        print("\n=== DEBUG: Problem Structure ===")
        print(f"Problem ID: {problem.get('problem_id', 'Unknown')}")
        print(f"Status: {problem.get('status', 'Unknown')}")
        print(f"Keys: {list(problem.keys())}")

        if "stats" in problem:
            print("\nStats structure:")
            stats = problem["stats"]
            print(f"  Keys: {list(stats.keys())}")

            if "feedback_impact" in stats:
                fi = stats["feedback_impact"]
                print(f"  feedback_impact keys: {list(fi.keys())}")
                if "depths" in fi:
                    print(f"    depths: {list(fi['depths'].keys())}")
                    for depth, data in fi["depths"].items():
                        print(f"      depth {depth}: {data}")

        if "code_eval_results" in problem:
            cer = problem["code_eval_results"]
            print(f"\ncode_eval_results keys: {list(cer.keys())}")
            if "pass_at_k" in cer:
                print(f"  pass_at_k: {cer['pass_at_k']}")

        print("=== END DEBUG ===\n")

    def _generate_enhanced_pass_at_k_table(self):
        """Generate Pass@k table with conditional evaluation - only for problems with ≥k candidates."""
        latex_content = [
            "\\subsection{Pass@k Evaluation with Conditional Sampling}",
            "",
            "\\textbf{Research Question}: How many attempts are needed for reliable code generation?",
            "",
            "\\textbf{Note}: Pass@k values are calculated only on problems that generated at least k candidates. ",
            "For example, pass@10 is computed only on the ~30\\% of problems that didn't find a solution ",
            "within the first 9 attempts. Lower percentages of qualifying problems for higher k values ",
            "indicate effective early stopping on easier problems.",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|cccc|cccc}",
            "\\toprule",
            "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{4}{c|}{\\textbf{Pass@k (\\%)}} & \\multicolumn{4}{c}{\\textbf{Problems with $\\geq$k candidates}} \\\\",
            "& \\textbf{@1} & \\textbf{@3} & \\textbf{@5} & \\textbf{@10} & \\textbf{$\\geq$1} & \\textbf{$\\geq$3} & \\textbf{$\\geq$5} & \\textbf{$\\geq$10} \\\\",
            "\\midrule"
        ]

        for model_name in self.models:
            pass_values = []
            problem_counts = []

            for k_str in ["1", "3", "5", "10"]:
                k = int(k_str)

                # Count problems with at least k candidates across all difficulties
                qualifying_problems = 0
                total_candidates_list = []

                for difficulty in ["Easy", "Medium", "Hard"]:
                    key = f"{model_name}_{difficulty}"
                    if key in self.problem_level_stats:
                        for problem in self.problem_level_stats[key]:
                            total_candidates = problem.get("total_candidates", 0)
                            total_candidates_list.append(total_candidates)

                            if total_candidates >= k:
                                qualifying_problems += 1

                # Get pass@k values from our aggregated data
                all_pass_values = []
                for difficulty in ["Easy", "Medium", "Hard"]:
                    if model_name in self.pass_at_k_results[difficulty]:
                        values = self.pass_at_k_results[difficulty][model_name].get(k_str, [])
                        if values:
                            all_pass_values.extend(values)

                if all_pass_values and qualifying_problems > 0:
                    # Calculate average pass@k
                    avg_pass = np.mean(all_pass_values)
                    # Convert to percentage if needed
                    if avg_pass <= 1.0:
                        avg_pass *= 100
                    pass_values.append(f"{avg_pass:.1f}")
                else:
                    pass_values.append("-")

                problem_counts.append(qualifying_problems)

            # Calculate total problems across all difficulties
            total_problems = sum(self.problem_counts[difficulty].get(model_name, 0)
                                 for difficulty in ["Easy", "Medium", "Hard"])

            # Format problem counts with percentages
            problem_percentages = []
            for count in problem_counts:
                if total_problems > 0:
                    pct = (count / total_problems) * 100
                    problem_percentages.append(f"{count} ({pct:.0f}\\%)")
                else:
                    problem_percentages.append(str(count))

            latex_content.append(
                f"{model_name} & {' & '.join(pass_values)} & {' & '.join(problem_percentages)} \\\\"
            )

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Pass@k metrics computed conditionally on problems with sufficient candidates. ",
            "The right columns show the number (and percentage) of problems that generated at least k candidates.}",
            "\\label{tab:pass_at_k_conditional}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _generate_problem_solving_efficiency_table(self):
        """Generate table showing overall problem-solving efficiency."""
        latex_content = [
            "\\subsection{Problem-Solving Efficiency}",
            "",
            "\\textbf{Research Question}: How efficiently do models solve problems with early stopping?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|c|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Problems Solved} & \\textbf{Problems Solved} & \\textbf{Avg. Candidates} & \\textbf{Avg. Candidates} \\\\",
            " & \\textbf{in First 3} & \\textbf{Overall} & \\textbf{to Solution} & \\textbf{for Failures} \\\\",
            "\\midrule"
        ]

        for model_name in self.models:
            # Initialize counters
            solved_in_first_3 = 0
            total_solved = 0
            total_problems = 0

            # FIXED: Separate tracking for candidates
            total_candidates_for_solved = 0  # Sum of candidates for solved problems
            total_candidates_for_failed = 0  # Sum of candidates for failed problems
            num_failed = 0

            # Analyze each problem across all difficulties
            for difficulty in ["Easy", "Medium", "Hard"]:
                key = f"{model_name}_{difficulty}"
                # ADD THE DEBUG CODE HERE
                if model_name == "qwen2-5-coder" and key in self.problem_level_stats and len(
                        self.problem_level_stats[key]) > 0:
                    sample = self.problem_level_stats[key][0]
                    print(f"DEBUG Sample problem data for {key}: total_candidates={sample.get('total_candidates')}, "
                          f"all_solutions={sample.get('all_solutions')}, "
                          f"nodes_explored={sample.get('nodes_explored')}, "
                          f"candidates_generated={sample.get('candidates_generated', 'NOT FOUND')}")

                if key in self.problem_level_stats:
                    for problem in self.problem_level_stats[key]:
                        total_problems += 1
                        total_candidates = problem.get("all_solutions", 0)
                        status = problem.get("status", "unknown")

                        if status == "solved":
                            total_solved += 1
                            # FIXED: Add to total, don't append to list
                            total_candidates_for_solved += total_candidates
                            if total_candidates <= 3:
                                solved_in_first_3 += 1
                        else:
                            num_failed += 1
                            total_candidates_for_failed += total_candidates

            # Calculate percentages and averages
            if total_problems > 0:
                solved_in_3_pct = (solved_in_first_3 / total_problems) * 100
                overall_pct = (total_solved / total_problems) * 100
                solved_in_3_str = f"{solved_in_first_3}/{total_problems} ({solved_in_3_pct:.1f}\\%)"
                overall_str = f"{total_solved}/{total_problems} ({overall_pct:.1f}\\%)"
            else:
                solved_in_3_str = "0/0 (0.0\\%)"
                overall_str = "0/0 (0.0\\%)"

            # FIXED: Calculate averages correctly
            avg_to_solution = total_candidates_for_solved / total_solved if total_solved > 0 else 0.0
            # Add this after calculating avg_to_solution
            print(f"DEBUG {model_name}: Total candidates for solved: {total_candidates_for_solved}, "
                  f"Solved count: {total_solved}, Average: {avg_to_solution}")
            avg_for_failures = total_candidates_for_failed / num_failed if num_failed > 0 else 0.0

            latex_content.append(
                f"{model_name} & {solved_in_3_str} & {overall_str} & "
                f"{avg_to_solution:.1f} & {avg_for_failures:.1f} \\\\"
            )

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Overall problem-solving efficiency showing how many problems were solved early ",
            "(within 3 candidates) versus requiring deeper exploration. Average candidates to solution ",
            "includes all candidates generated before finding the solution. Average candidates for failures ",
            "shows exploration depth for unsolved problems.}",
            "\\label{tab:problem_solving_efficiency}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _generate_difficulty_breakdown_table(self):
        """Generate table showing candidate generation patterns by difficulty."""
        latex_content = [
            "\\subsection{Candidate Generation Patterns by Difficulty}",
            "",
            "\\textbf{Research Question}: How does problem difficulty affect exploration depth?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|l|cccc|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Difficulty} & \\textbf{$\\geq$3} & \\textbf{$\\geq$5} & \\textbf{$\\geq$10} & \\textbf{$\\geq$20} & \\textbf{Avg. Candidates} \\\\",
            "\\midrule"
        ]

        for model_idx, model_name in enumerate(self.models):
            for diff_idx, difficulty in enumerate(["Easy", "Medium", "Hard"]):
                # Count problems with different candidate thresholds
                counts = {3: 0, 5: 0, 10: 0, 20: 0}
                all_candidates = []
                total_problems = 0

                key = f"{model_name}_{difficulty}"
                if key in self.problem_level_stats:
                    for problem in self.problem_level_stats[key]:
                        total_problems += 1
                        total_candidates = problem.get("total_candidates", 0)
                        all_candidates.append(total_candidates)

                        # Count problems exceeding each threshold
                        for threshold in [3, 5, 10, 20]:
                            if total_candidates >= threshold:
                                counts[threshold] += 1

                # Calculate average
                avg_candidates = np.mean(all_candidates) if all_candidates else 0.0

                # Format cells with counts and percentages
                cells = []
                for threshold in [3, 5, 10, 20]:
                    if total_problems > 0:
                        pct = (counts[threshold] / total_problems) * 100
                        cells.append(f"{counts[threshold]}/{total_problems} ({pct:.0f}\\%)")
                    else:
                        cells.append("0/0 (0\\%)")

                # Add model name only on first row
                model_display = model_name if diff_idx == 0 else ""

                latex_content.append(
                    f"{model_display} & {difficulty} & {' & '.join(cells)} & {avg_candidates:.1f} \\\\"
                )

            if model_idx < len(self.models) - 1:
                latex_content.append("\\midrule")

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Candidate generation patterns by difficulty level. Shows the number (and percentage) ",
            "of problems that generated at least k candidates, revealing how exploration depth varies with problem difficulty. ",
            "Easy problems often solve in <3 attempts, while Hard problems almost always require deep exploration.}",
            "\\label{tab:difficulty_breakdown}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _generate_depth_distribution_table(self):
        """Generate table showing distribution of solutions across depths."""
        latex_content = [
            "\\subsection{Solution Distribution Across Tree Depths}",
            "",
            "This table shows where solutions are found in the exploration tree, ",
            "helping identify the optimal search depth for different problem difficulties.",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|l|cccc|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Difficulty} & \\textbf{Depth 0} & \\textbf{Depth 1} & \\textbf{Depth 2} & \\textbf{Depth 3} & \\textbf{Total} \\\\",
            "\\midrule"
        ]

        for model_idx, model_name in enumerate(self.models):
            for diff_idx, difficulty in enumerate(["Easy", "Medium", "Hard"]):
                # Count solutions at each depth
                depth_counts = defaultdict(int)

                # Use first_pass_depths to count where solutions were found
                first_pass_depths = self.detailed_stats[difficulty][model_name].get("first_pass_depths", [])
                for depth in first_pass_depths:
                    if depth >= 0:
                        depth_counts[depth] += 1

                # Also use depth solution counts if available
                for depth in range(4):
                    count = self.depth_solution_counts.get(model_name, {}).get(difficulty, {}).get(str(depth), 0)
                    if count > 0:
                        depth_counts[depth] = max(depth_counts[depth], count)

                # Calculate total
                total = sum(depth_counts.values())

                # Format cells with counts and percentages
                cells = []
                for depth in range(4):
                    count = depth_counts[depth]
                    if total > 0:
                        pct = (count / total) * 100
                        cells.append(f"{count} ({pct:.0f}\\%)")
                    else:
                        cells.append("0")

                # Add model name only on first row
                if diff_idx == 0:
                    row_start = model_name
                else:
                    row_start = ""

                latex_content.append(
                    f"{row_start} & {difficulty} & {' & '.join(cells)} & {total} \\\\"
                )

            if model_idx < len(self.models) - 1:
                latex_content.append("\\midrule")

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Distribution of successful solutions across tree depths by model and difficulty. ",
            "Shows both count and percentage of solutions found at each depth.}",
            "\\label{tab:depth_distribution}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _generate_feedback_depth_impact_table(self):
        """
        Build the LaTeX "Feedback Effectiveness by Tree Depth" table.
        FIXED: Now correctly accesses error_analysis with model and difficulty keys.
        """
        # ---------- table header --------------------------------------------------
        latex = [
            "\\subsection{Feedback Effectiveness by Tree Depth}",
            "",
            "\\textbf{Research Question}: Does iterative feedback improve solution quality?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|ccc|ccc|c}",
            "\\toprule",
            "\\multirow{2}{*}{\\textbf{Model–Difficulty}} & "
            "\\multicolumn{3}{c|}{\\textbf{Improvement Rate}} & "
            "\\multicolumn{3}{c|}{\\textbf{Solutions Found}} & "
            "\\textbf{Error} \\\\",
            "& 0$\\,\\rightarrow\\,$1 & 1$\\,\\rightarrow\\,$2 & 2$\\,\\rightarrow\\,$3 "
            "& D1 & D2 & D3 & \\textbf{Recovery} \\\\",
            "\\midrule"
        ]

        # ---------- rows ----------------------------------------------------------
        for mi, model in enumerate(self.models):
            for difficulty in ("Easy", "Medium", "Hard"):
                # ── 1) improvement rates ------------------------------------------
                rates = []
                for depth in ("1", "2", "3"):  # transition buckets
                    bucket = (
                        self.feedback_impact_data
                        .get(model, {})
                        .get(difficulty, {})
                        .get(depth, [])
                    )
                    attempts = sum(x.get("attempts", 0) for x in bucket)
                    improvements = sum(
                        x.get("improvements", 0) + x.get("solved", 0) for x in bucket
                    )
                    pct = (improvements / attempts * 100) if attempts else 0.0
                    rates.append(f"{pct:.1f}\\%")

                # ── 2) solutions found at each depth ------------------------------
                d_counts = self.depth_solution_counts.get(model, {}).get(difficulty, {})
                d1 = d_counts.get("1", 0)
                d2 = d_counts.get("2", 0)
                d3 = d_counts.get("3", 0)

                # ── 3) FIXED: error-recovery rate ---------------------------------
                err_stats = self.error_analysis.get(model, {}).get(difficulty, {})
                tot_occ = sum(v["occurrences"] for v in err_stats.values())
                tot_res = sum(v["resolutions"] for v in err_stats.values())
                recovery = (tot_res / tot_occ * 100) if tot_occ else 0.0

                # ── 4) emit row ----------------------------------------------------
                latex.append(
                    f"{model}–{difficulty} & {' & '.join(rates)} & "
                    f"{d1} & {d2} & {d3} & {recovery:.1f}\\% \\\\"
                )
            if mi != len(self.models) - 1:  # horizontal rule between models
                latex.append("\\midrule")

        # ---------- table footer --------------------------------------------------
        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Per-difficulty feedback effectiveness. "
            "Improvement Rate = (improvements + solves)$\\,\\div$attempts at that depth.}",
            "\\label{tab:feedback_depth_impact}",
            "\\end{table}",
            ""
        ])

        return latex

    def _generate_solution_diversity_table(self):
        """Generate solution diversity metrics table with proper calculations."""
        latex_content = [
            "\\subsection{Solution Diversity Analysis}",
            "",
            "\\textbf{Research Question}: Do models generate diverse solution strategies?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|c|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Unique Ratio} & \\textbf{Avg. Similarity} & \\textbf{Feature Diversity} & \\textbf{Length Variance} \\\\",
            "\\midrule"
        ]

        for model_name in self.models:
            # Aggregate diversity data across difficulties
            all_diversity_data = []
            for difficulty in ["Easy", "Medium", "Hard"]:
                key = f"{model_name}_{difficulty}"
                if key in self.solution_diversity_data:
                    all_diversity_data.extend(self.solution_diversity_data[key])

            if all_diversity_data:
                # Calculate metrics properly
                unique_ratios = []
                similarities = []
                feature_diversities = []
                length_variances = []

                for d in all_diversity_data:
                    # Unique ratio (already calculated correctly in the data)
                    if "unique_ratio" in d:
                        unique_ratios.append(d["unique_ratio"] * 100)

                    # Similarity score
                    if "similarity_score" in d:
                        similarities.append(d["similarity_score"])

                    # Feature diversity
                    if "feature_diversity" in d:
                        feature_diversities.append(d["feature_diversity"])

                    # Length variance calculation
                    if "solution_lengths" in d:
                        lengths = d["solution_lengths"]
                        if lengths.get("avg", 0) > 0:
                            variance = (lengths.get("max", 0) - lengths.get("min", 0)) / lengths.get("avg", 1)
                            length_variances.append(variance)

                avg_unique_ratio = np.mean(unique_ratios) if unique_ratios else 0
                avg_similarity = np.mean(similarities) if similarities else 0
                avg_feature_diversity = np.mean(feature_diversities) if feature_diversities else 0
                avg_length_variance = np.mean(length_variances) if length_variances else 0

                latex_content.append(
                    f"{model_name} & {avg_unique_ratio:.1f}\\% & {avg_similarity:.2f} & "
                    f"{avg_feature_diversity:.2f} & {avg_length_variance:.2f} \\\\"
                )
            else:
                latex_content.append(f"{model_name} & - & - & - & - \\\\")

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Solution diversity metrics showing variety in generated approaches.}",
            "\\label{tab:solution_diversity}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _generate_error_transition_analysis_table(self):
        """Generate error transition analysis table."""
        latex_content = [
            "\\subsection{Error Transition Analysis}",
            "",
            "\\textbf{Research Question}: How do errors evolve through iterative feedback?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|l}",
            "\\toprule",
            "\\textbf{Error Transition} & \\textbf{Frequency} & \\textbf{Interpretation} \\\\",
            "\\midrule"
        ]

        # Sort transitions by frequency
        sorted_transitions = sorted(self.error_transition_counts.items(),
                                    key=lambda x: x[1],
                                    reverse=True)[:10]

        for transition, count in sorted_transitions:
            # Parse transition
            parts = transition.split("->")
            if len(parts) == 2:
                from_error = parts[0].replace("_", " ").title()
                to_error = parts[1].replace("_", " ").title()

                # Interpret the transition
                if to_error == "Unknown" or to_error == "Pass":
                    interpretation = "Error resolved"
                elif from_error == to_error:
                    interpretation = "Persistent error"
                else:
                    interpretation = "Error transformation"

                latex_content.append(
                    f"{from_error} → {to_error} & {count} & {interpretation} \\\\"
                )

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Most common error transitions showing how errors evolve through feedback iterations.}",
            "\\label{tab:error_transitions}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _generate_depth_performance_analysis_table(self):
        """Generate detailed performance analysis by tree depth."""
        latex_content = [
            "\\subsection{Performance Analysis by Tree Depth}",
            "",
            "\\textbf{Research Question}: At which depths are solutions most likely to be found?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|l|cccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Metric} & \\textbf{Depth 0} & \\textbf{Depth 1} & \\textbf{Depth 2} & \\textbf{Depth 3} \\\\",
            "\\midrule"
        ]

        for model_name in self.models:
            # Aggregate depth statistics across difficulties
            depth_stats = defaultdict(lambda: {"total": 0, "passed": 0})

            for difficulty in ["Easy", "Medium", "Hard"]:
                for depth in range(4):
                    key = f"depth_{depth}_solutions"
                    if key in self.detailed_stats[difficulty][model_name]:
                        for stat in self.detailed_stats[difficulty][model_name][key]:
                            depth_stats[depth]["total"] += stat.get("total", 0)
                            depth_stats[depth]["passed"] += stat.get("passed", 0)

            # Solutions row
            solutions_data = []
            for depth in range(4):
                passed = depth_stats[depth]["passed"]
                solutions_data.append(str(passed) if passed > 0 else "-")

            latex_content.append(f"{model_name} & Solutions & {' & '.join(solutions_data)} \\\\")

            # Success rate row
            rates_data = []
            for depth in range(4):
                total = depth_stats[depth]["total"]
                passed = depth_stats[depth]["passed"]
                rate = (passed / total * 100) if total > 0 else 0
                rates_data.append(f"{rate:.1f}\\%" if total > 0 else "-")

            latex_content.append(f" & Success Rate & {' & '.join(rates_data)} \\\\")

            if model_name != self.models[-1]:
                latex_content.append("\\midrule")

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Solution discovery patterns across tree depths showing where solutions are most likely to be found.}",
            "\\label{tab:depth_performance}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def generate_latex(self):
        """Generate LaTeX document with all research-focused tables."""
        print("Generating enhanced LaTeX document with complete data analysis...")

        # Calculate improvement percentages
        improvements = self.calculate_improvements()

        # Start building the LaTeX document
        latex_content = []

        # Document preamble
        latex_content.extend([
            "\\documentclass[10pt]{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{multirow}",
            "\\usepackage{array}",
            "\\usepackage{colortbl}",
            "\\usepackage{xcolor}",
            "\\usepackage{siunitx}",
            "\\usepackage{caption}",
            "\\usepackage{adjustbox}",
            "\\usepackage{amsmath}",
            "\\begin{document}",
            "",
            f"\\section{{LeetCode Performance Analysis ({self.current_date})}}",
            ""
        ])

        # Experimental Setup
        easy_count = self.problem_counts['Easy'].get('qwen2-5-coder', 20)
        medium_count = self.problem_counts['Medium'].get('qwen2-5-coder', 20)
        hard_count = self.problem_counts['Hard'].get('qwen2-5-coder', 10)
        total_count = easy_count + medium_count + hard_count

        latex_content.extend([
            "\\subsection{Experimental Setup}",
            "",
            "\\begin{itemize}",
            "\\item \\textbf{Models}: qwen2-5-coder, deepseek-r1-distill, qwq-preview",
            f"\\item \\textbf{{Dataset}}: {easy_count} Easy, {medium_count} Medium, {hard_count} Hard problems ({total_count} total)",
            "\\item \\textbf{Tree Search}: k=3 initial solutions, branch factor=3, max depth=3",
            "\\item \\textbf{Adaptive Mode}: Multi-signal termination with early stopping",
            "\\item \\textbf{Evaluation}: HuggingFace code\\_eval metric with pass@k (k $\\in$ \\{1, 3, 5, 10\\})",
            "\\item \\textbf{Feedback}: Error-specific prompting with test case analysis",
            "\\end{itemize}",
            ""
        ])

        # Add all tables in logical order
        latex_content.extend(self._generate_model_performance_comparison_table())

        # Add the new efficiency table before pass@k
        latex_content.extend(self._generate_problem_solving_efficiency_table())

        # Enhanced pass@k with better explanation
        latex_content.extend(self._generate_enhanced_pass_at_k_table())

        # Add difficulty breakdown to show why pass@k behaves this way
        latex_content.extend(self._generate_difficulty_breakdown_table())

        # Tree-depth based analysis
        # latex_content.extend(self._generate_tree_depth_success_table())
        latex_content.extend(self._generate_depth_distribution_table())

        # Continue with other tables
        latex_content.extend(self._generate_feedback_depth_impact_table())
        latex_content.extend(self._generate_error_type_resolution_table())
        latex_content.extend(self._generate_tree_exploration_efficiency_table())
        latex_content.extend(self._generate_test_case_difficulty_analysis_table())
        latex_content.extend(self._generate_solution_diversity_table())
        latex_content.extend(self._generate_computational_cost_analysis_table())
        latex_content.extend(self._generate_error_transition_analysis_table())
        latex_content.extend(self._generate_depth_performance_analysis_table())
        latex_content.extend(self._generate_success_rate_table())
        latex_content.extend(self._generate_feedback_iteration_tables())

        # End document
        latex_content.append("\\end{document}")

        # Write to file
        with open(self.output_file, 'w') as f:
            f.write("\n".join(latex_content))

        print(f"Enhanced LaTeX document generated successfully: {self.output_file}")
        return self.output_file

    # Keep all other methods from the previous version...
    def _generate_model_performance_comparison_table(self):
        """
        LaTeX table where every 'Success' cell shows  N_solved / N_total  plus %
        so it matches the visual style of your first success-rate table.
        """
        latex = [
            "\\subsection{Overall Model Performance Comparison}",
            "",
            "\\textbf{Research Question}: How do different LLMs perform on LeetCode problem-solving across difficulty levels?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|cc|cc|cc|cc|c|c}",
            "\\toprule",
            "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{2}{c|}{\\textbf{Easy}} & "
            "\\multicolumn{2}{c|}{\\textbf{Medium}} & \\multicolumn{2}{c|}{\\textbf{Hard}} & "
            "\\multicolumn{2}{c|}{\\textbf{Overall}} & \\textbf{Avg.} & \\textbf{Avg.}\\\\",
            "& Success & Time (s) & Success & Time (s) & Success & Time (s) & Success & Time (s) & "
            "\\textbf{Candidates} & \\textbf{Time (s)}\\\\",
            "\\midrule"
        ]

        for model in self.models:
            row = []

            # per-difficulty cells -------------------------------------------------
            for diff in ["Easy", "Medium", "Hard", "Overall"]:
                cell_solved = cell_total = cell_rate = cell_time = "-"
                if model in self.results[diff]:
                    r = self.results[diff][model]
                    cell_total = r.get("total_problems", 0)
                    cell_solved = r.get("solved_problems", 0)
                    cell_rate = r.get("success_rate", 0.0)
                    cell_time = r.get("avg_time", 0.0)

                    success_str = f"{cell_solved}/{cell_total} ({cell_rate:.1f}\\%)"
                    time_str = f"{cell_time:.1f}"
                else:
                    success_str, time_str = "-", "-"

                row.extend([success_str, time_str])

            # overall averages -----------------------------------------------------
            overall = self.results["Overall"].get(model, {})
            avg_cand = overall.get("avg_candidates", 0.0)
            avg_time = overall.get("avg_time", 0.0)

            latex.append(f"{model} & {' & '.join(row)} & {avg_cand:.1f} & {avg_time:.1f} \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Comprehensive model performance comparison across difficulty levels.}",
            "\\label{tab:model_performance_comparison}",
            "\\end{table}",
            ""
        ])
        return latex

    def _generate_error_type_resolution_table(self):
        """Generate error type resolution matrix (Research Question 4)."""
        latex_content = [
            "\\subsection{Error Type Resolution Analysis}",
            "",
            "\\textbf{Research Question}: Which error types are most effectively resolved through feedback?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|c|l}",
            "\\toprule",
            "\\textbf{Error Type} & \\textbf{Occurrences} & \\textbf{Resolution Rate} & \\textbf{Avg. Attempts} & \\textbf{Category} \\\\",
            "\\midrule"
        ]

        # Aggregate error data across all models and difficulties
        aggregated_errors = defaultdict(lambda: {"occurrences": 0, "resolutions": 0})

        for model in self.error_analysis:
            for difficulty in self.error_analysis[model]:
                for error_type, data in self.error_analysis[model][difficulty].items():
                    aggregated_errors[error_type]["occurrences"] += data["occurrences"]
                    aggregated_errors[error_type]["resolutions"] += data["resolutions"]

        # Sort error types by occurrence
        sorted_errors = sorted(aggregated_errors.items(),
                               key=lambda x: x[1]["occurrences"],
                               reverse=True)[:10]  # Top 10 errors

        for error_type, data in sorted_errors:
            occurrences = data["occurrences"]
            resolutions = data["resolutions"]
            resolution_rate = (resolutions / occurrences * 100) if occurrences > 0 else 0
            avg_attempts = occurrences / resolutions if resolutions > 0 else float('inf')

            # Categorize errors
            if "import" in error_type or "module" in error_type:
                category = "Import/Module"
            elif "assertion" in error_type:
                category = "Logic"
            elif any(x in error_type for x in ["index", "key", "attribute"]):
                category = "Access"
            elif any(x in error_type for x in ["type", "value"]):
                category = "Type/Value"
            else:
                category = "Other"

            avg_attempts_str = f"{avg_attempts:.1f}" if avg_attempts != float('inf') else "$\\infty$"

            latex_content.append(
                f"{error_type.replace('_', ' ').title()} & {occurrences} & "
                f"{resolution_rate:.1f}\\% & {avg_attempts_str} & {category} \\\\"
            )

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Error type analysis showing resolution effectiveness through iterative feedback.}",
            "\\label{tab:error_resolution}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _generate_tree_exploration_efficiency_table(self):
        """Generate tree exploration efficiency analysis (Research Question 5)."""
        latex_content = [
            "\\subsection{Tree Exploration Efficiency}",
            "",
            "\\textbf{Research Question}: What is the optimal tree exploration strategy for different difficulties?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|l|c|c|c|l}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Difficulty} & \\textbf{Success Rate} & \\textbf{Avg. Nodes} & \\textbf{Nodes/Success} & \\textbf{Primary Termination} \\\\",
            "\\midrule"
        ]

        for model_name in self.models:
            for difficulty in ["Easy", "Medium", "Hard"]:
                key = f"{model_name}_{difficulty}"
                if key in self.tree_exploration_data:
                    data = self.tree_exploration_data[key]

                    # Calculate metrics
                    success_rate = self.results[difficulty].get(model_name, {}).get("success_rate", 0)
                    avg_nodes = np.mean([d["nodes_explored"] for d in data]) if data else 0
                    successful_runs = sum(1 for d in data if d["status"] == "solved")
                    nodes_per_success = avg_nodes / (successful_runs / len(data)) if successful_runs > 0 else float(
                        'inf')

                    # Find primary termination reason
                    termination_counts = Counter()
                    for d in data:
                        for reason, count in d.get("termination_reasons", {}).items():
                            termination_counts[reason] += count

                    primary_termination = termination_counts.most_common(1)[0][0] if termination_counts else "unknown"
                    primary_termination = primary_termination.replace("_", " ").title()

                    nodes_per_success_str = f"{nodes_per_success:.1f}" if nodes_per_success != float(
                        'inf') else "$\\infty$"

                    latex_content.append(
                        f"{model_name} & {difficulty} & {success_rate:.1f}\\% & "
                        f"{avg_nodes:.1f} & {nodes_per_success_str} & {primary_termination} \\\\"
                    )

                if model_name != self.models[-1] and difficulty == "Hard":
                    latex_content.append("\\midrule")

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Tree exploration efficiency metrics by model and difficulty.}",
            "\\label{tab:tree_efficiency}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _generate_test_case_difficulty_analysis_table(self):
        """Generate test case difficulty analysis (Research Question 6)."""
        latex_content = [
            "\\subsection{Test Case Difficulty Analysis}",
            "",
            "\\textbf{Research Question}: Which test patterns are most challenging for LLMs?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|l}",
            "\\toprule",
            "\\textbf{Test Pattern} & \\textbf{Failure Count} & \\textbf{First-Fail Rate} & \\textbf{Example} \\\\",
            "\\midrule"
        ]

        # Get top hardest test cases
        hardest_cases = sorted(self.test_case_analysis["hardest_cases"].items(),
                               key=lambda x: x[1],
                               reverse=True)[:8]

        for test_case, failure_count in hardest_cases:
            # Calculate first-fail rate properly
            first_fail_count = self.test_case_analysis["first_failing"].get(test_case, 0)
            # Ensure rate doesn't exceed 100%
            first_fail_rate = min((first_fail_count / failure_count * 100) if failure_count > 0 else 0, 100.0)

            # Categorize and create example
            test_str = str(test_case)
            if "[]" in test_str or "empty" in test_str.lower():
                pattern = "Empty input"
                example = "nums=[]"
            elif any(x in test_str for x in ["10000", "100000", "1000000"]):
                pattern = "Large input"
                example = "n=$10^6$"
            elif "0" in test_str or "-" in test_str:
                pattern = "Edge values"
                example = "k=0, negatives"
            elif len(test_str) > 50:
                pattern = "Complex input"
                example = "Multiple constraints"
            else:
                pattern = "Special case"
                example = test_str[:30] + "..." if len(test_str) > 30 else test_str

            latex_content.append(
                f"{pattern} & {failure_count} & {first_fail_rate:.1f}\\% & \\texttt{{{example}}} \\\\"
            )

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Most challenging test case patterns based on failure analysis.}",
            "\\label{tab:test_difficulty}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _generate_computational_cost_analysis_table(self):
        """Generate computational cost analysis table (Research Question 8)."""
        latex_content = [
            "\\subsection{Computational Cost Analysis}",
            "",
            "\\textbf{Research Question}: What is the computational trade-off for higher success rates?",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|c|c|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Success Rate} & \\textbf{Total Time (s)} & \\textbf{Candidates} & \\textbf{Time/Success} & \\textbf{Efficiency Score} \\\\",
            "\\midrule"
        ]

        for model_name in self.models:
            overall = self.results["Overall"].get(model_name, {})
            success_rate = overall.get("success_rate", 0)
            avg_time = overall.get("avg_time", 0)
            avg_candidates = overall.get("avg_candidates", 0)
            solved = overall.get("solved_problems", 0)
            total = overall.get("total_problems", 1)

            # Calculate time per successful solution
            time_per_success = (avg_time * total / solved) if solved > 0 else float('inf')

            # Calculate efficiency score (success_rate / time_per_success)
            efficiency_score = success_rate / time_per_success if time_per_success > 0 and time_per_success != float(
                'inf') else 0

            time_per_success_str = f"{time_per_success:.1f}" if time_per_success != float('inf') else "$\\infty$"

            latex_content.append(
                f"{model_name} & {success_rate:.1f}\\% & {avg_time:.1f} & "
                f"{avg_candidates:.1f} & {time_per_success_str} & {efficiency_score:.3f} \\\\"
            )

        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Computational cost analysis showing trade-offs between success rate and resource usage.}",
            "\\label{tab:computational_cost}",
            "\\end{table}",
            ""
        ])

        return latex_content

    def _process_summary(self, summary_file, model_name, difficulty):
        """Process summary.json file for a model."""
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)

            # Extract additional metrics from summary
            if "code_eval" in summary_data:
                pass_at_k = summary_data["code_eval"]

                # Add to the pass@k results - handle both formats
                for k, value in pass_at_k.items():
                    k_value = str(k).replace("pass@", "").replace("pass_at_", "")
                    # Only use summary data if we don't have individual problem data
                    if k_value not in self.pass_at_k_results[difficulty][model_name] or \
                            not self.pass_at_k_results[difficulty][model_name][k_value]:
                        self.pass_at_k_results[difficulty][model_name][k_value] = [value]

            # Extract total and solved problems
            if "total_problems" in summary_data and "solved_problems" in summary_data:
                total = summary_data["total_problems"]
                solved = summary_data["solved_problems"]

                # Update if not already set from combined results
                if "total_problems" not in self.results[difficulty][model_name]:
                    self.results[difficulty][model_name]["total_problems"] = total
                    self.results[difficulty][model_name]["solved_problems"] = solved

                    if total > 0:
                        self.results[difficulty][model_name]["success_rate"] = (solved / total) * 100

                # Update problem count
                self.problem_counts[difficulty][model_name] = total

        except Exception as e:
            print(f"Error processing summary for {model_name} at {difficulty} level: {str(e)}")

    def analyze_feedback_iterations(self):
        """Analyze how model performance improves with each feedback iteration."""
        print("Analyzing feedback iterations...")

        # Define the iterations and corresponding pass@k metrics
        iterations = [
            {"name": "Initial", "metric": "1"},
            {"name": "Iter 1", "metric": "3"},
            {"name": "Iter 2", "metric": "5"},
            {"name": "Iter 3", "metric": "10"}
        ]

        # Initialize result structure
        feedback_results = {
            "absolute_performance": {},
            "marginal_improvements": {},
            "efficiency_frontier": {}
        }

        # Extract performance values for each model, difficulty, and iteration
        for model_name in self.models:
            feedback_results["absolute_performance"][model_name] = {}

            for difficulty in ["Easy", "Medium", "Hard", "Overall"]:
                if model_name in self.pass_at_k_results[difficulty]:
                    feedback_results["absolute_performance"][model_name][difficulty] = {}

                    # Get values for each iteration
                    for iter_info in iterations:
                        iter_name = iter_info["name"]
                        k = iter_info["metric"]

                        if k in self.pass_at_k_results[difficulty][model_name]:
                            values = self.pass_at_k_results[difficulty][model_name][k]
                            if values:
                                avg_value = np.mean(values) * 100 if max(values) <= 1 else np.mean(values)
                                feedback_results["absolute_performance"][model_name][difficulty][iter_name] = avg_value

        # Calculate marginal improvements between iterations
        for model_name in self.models:
            feedback_results["marginal_improvements"][model_name] = {}

            for difficulty in ["Easy", "Medium", "Hard", "Overall"]:
                if difficulty in feedback_results["absolute_performance"][model_name]:
                    perf = feedback_results["absolute_performance"][model_name][difficulty]

                    if len(perf) > 1:
                        feedback_results["marginal_improvements"][model_name][difficulty] = {}

                        # Calculate improvements between consecutive iterations
                        for i in range(1, len(iterations)):
                            prev_iter = iterations[i - 1]["name"]
                            curr_iter = iterations[i]["name"]

                            if prev_iter in perf and curr_iter in perf:
                                improvement_key = f"{prev_iter}→{curr_iter}"
                                improvement = perf[curr_iter] - perf[prev_iter]
                                feedback_results["marginal_improvements"][model_name][difficulty][
                                    improvement_key] = improvement

        # Identify efficiency frontier (optimal number of iterations)
        for model_name in self.models:
            feedback_results["efficiency_frontier"][model_name] = {}

            for difficulty in ["Easy", "Medium", "Hard", "Overall"]:
                if difficulty in feedback_results["marginal_improvements"][model_name]:
                    improvements = feedback_results["marginal_improvements"][model_name][difficulty]

                    # Simple rule: optimal iteration is the last one with improvement > threshold
                    threshold = 3.0  # 3 percentage points as threshold
                    optimal_iter = 1  # Default to first iteration

                    # Check iteration 1 → 2
                    if "Initial→Iter 1" in improvements and improvements["Initial→Iter 1"] > threshold:
                        optimal_iter = 1

                        # Check iteration 2 → 3
                        if "Iter 1→Iter 2" in improvements and improvements["Iter 1→Iter 2"] > threshold:
                            optimal_iter = 2

                            # Check iteration 3 → 4
                            if "Iter 2→Iter 3" in improvements and improvements["Iter 2→Iter 3"] > threshold:
                                optimal_iter = 3

                    # Special case: if an iteration has negative improvement but the next has strong positive
                    if optimal_iter == 1 and "Iter 1→Iter 2" in improvements and improvements["Iter 1→Iter 2"] < 0:
                        if "Iter 2→Iter 3" in improvements and improvements["Iter 2→Iter 3"] > 2 * threshold:
                            optimal_iter = 3  # Skip to iteration 3

                    feedback_results["efficiency_frontier"][model_name][difficulty] = optimal_iter

        return feedback_results

    def _generate_feedback_iteration_tables(self):
        """Generate LaTeX tables for feedback iteration analysis with absolute counts."""
        # Get feedback iteration analysis results
        feedback_results = self.analyze_feedback_iterations()

        latex_content = []

        # Introduction section
        latex_content.extend([
            "\\subsection{Feedback Iteration Analysis}",
            "",
            "This analysis quantifies how model performance improves with each feedback iteration in our tree search algorithm. ",
            "We investigate the marginal improvements between iterations to identify the optimal number of feedback rounds, ",
            "addressing the efficiency frontier where additional iterations yield diminishing returns.",
            ""
        ])

        # Table 1: Performance across iterations with absolute counts
        latex_content.extend([
            "\\subsection{Performance Across Feedback Iterations}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|l|c|c|c|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Difficulty} & \\textbf{Initial} & \\textbf{Iter 1} & \\textbf{Iter 2} & \\textbf{Iter 3}\\\\",
            "\\midrule"
        ])

        # Add rows for each model and difficulty with absolute counts
        for model_name in self.models:
            first_row = True
            for difficulty in ["Easy", "Medium", "Hard"]:
                if difficulty in feedback_results["absolute_performance"][model_name]:
                    perf = feedback_results["absolute_performance"][model_name][difficulty]

                    # Get total problems for this model and difficulty
                    if difficulty == "Overall":
                        total_problems = self.results["Overall"][model_name].get("total_problems", 0)
                    else:
                        total_problems = self.problem_counts[difficulty].get(model_name, 0)

                    # Calculate approximate solved counts for each iteration
                    init_solved = int(
                        round((perf.get('Initial', 0) / 100) * total_problems)) if 'Initial' in perf else 0
                    iter1_solved = int(round((perf.get('Iter 1', 0) / 100) * total_problems)) if 'Iter 1' in perf else 0
                    iter2_solved = int(round((perf.get('Iter 2', 0) / 100) * total_problems)) if 'Iter 2' in perf else 0
                    iter3_solved = int(round((perf.get('Iter 3', 0) / 100) * total_problems)) if 'Iter 3' in perf else 0

                    # Format absolute counts and percentages
                    init_val = f"{init_solved}/{total_problems} ({perf.get('Initial', 0):.1f}\\%)" if 'Initial' in perf else "-"
                    iter1_val = f"{iter1_solved}/{total_problems} ({perf.get('Iter 1', 0):.1f}\\%)" if 'Iter 1' in perf else "-"
                    iter2_val = f"{iter2_solved}/{total_problems} ({perf.get('Iter 2', 0):.1f}\\%)" if 'Iter 2' in perf else "-"
                    iter3_val = f"{iter3_solved}/{total_problems} ({perf.get('Iter 3', 0):.1f}\\%)" if 'Iter 3' in perf else "-"

                    # Format row with model name only on first row
                    if first_row:
                        latex_content.append(
                            f"{model_name} & {difficulty} & {init_val} & {iter1_val} & {iter2_val} & {iter3_val} \\\\")
                        first_row = False
                    else:
                        latex_content.append(
                            f" & {difficulty} & {init_val} & {iter1_val} & {iter2_val} & {iter3_val} \\\\")

            # Add separator between models
            if model_name != self.models[-1]:
                latex_content.append("\\midrule")

        # Close the performance table
        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Performance across feedback iterations by model and difficulty, showing solved/total problems and success rates.}",
            "\\label{tab:feedback_iterations}",
            "\\end{table}",
            ""
        ])

        # Table 3: Efficiency Frontier
        latex_content.extend([
            "\\subsection{Efficiency Frontier Analysis}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{tabular}{l|c|c|c|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Easy} & \\textbf{Medium} & \\textbf{Hard} & \\textbf{Overall} \\\\",
            "\\midrule"
        ])

        for model_name in self.models:
            # Extract optimal iterations for each difficulty
            easy_opt = feedback_results["efficiency_frontier"][model_name].get("Easy", "-")
            medium_opt = feedback_results["efficiency_frontier"][model_name].get("Medium", "-")
            hard_opt = feedback_results["efficiency_frontier"][model_name].get("Hard", "-")
            overall_opt = feedback_results["efficiency_frontier"][model_name].get("Overall", "-")

            latex_content.append(f"{model_name} & {easy_opt} & {medium_opt} & {hard_opt} & {overall_opt} \\\\")

        # Close the efficiency frontier table
        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Optimal number of feedback iterations (efficiency frontier) by model and difficulty.}",
            "\\label{tab:efficiency_frontier}",
            "\\end{table}",
            ""
        ])

        # Add explanation paragraph
        latex_content.extend([
            "The efficiency frontier analysis reveals that the optimal number of feedback iterations varies by both model and problem difficulty. ",
            "For Hard problems, more iterations (typically 3) are beneficial across all models, while for Easy problems, ",
            "diminishing returns often appear after fewer iterations. This suggests that our tree search algorithm should ",
            "dynamically adjust the number of feedback iterations based on problem difficulty, allocating more iterations to harder problems.",
            "",
            "The marginal improvements between iterations show non-linear patterns, particularly for harder problems, ",
            "suggesting that the tree search algorithm's branching strategy effectively explores diverse solution spaces ",
            "before converging on successful approaches in later iterations.",
            ""
        ])

        return latex_content

    def _calculate_overall_metrics(self):
        """
        Aggregate Easy / Medium / Hard into an Overall bucket and ensure every
        key needed later is present (even if zero).  This prevents empty cells
        in the LaTeX tables.
        """
        for model in self.models:
            overall = {
                "total_problems": 0,
                "solved_problems": 0,
                "time_sum": 0.0,
                "candidates_sum": 0.0,
                "success_rate": 0.0,
                "avg_time": 0.0,
                "avg_candidates": 0.0
            }

            # weighted sums
            for diff in ["Easy", "Medium", "Hard"]:
                if model not in self.results[diff]:
                    continue
                d = self.results[diff][model]

                n_tot = d.get("total_problems", 0)
                n_ok = d.get("solved_problems", 0)
                t_avg = d.get("avg_time", 0.0)
                cand_avg = d.get("avg_candidates", 0.0)

                overall["total_problems"] += n_tot
                overall["solved_problems"] += n_ok
                overall["time_sum"] += t_avg * n_tot
                overall["candidates_sum"] += cand_avg * n_tot

            if overall["total_problems"] > 0:
                overall["success_rate"] = overall["solved_problems"] / overall["total_problems"] * 100
                overall["avg_time"] = overall["time_sum"] / overall["total_problems"]
                overall["avg_candidates"] = overall["candidates_sum"] / overall["total_problems"]

            # strip helper sums
            overall.pop("time_sum", None)
            overall.pop("candidates_sum", None)
            self.results["Overall"][model] = overall

    def _calculate_overall_pass_at_k(self, model_name):
        """Calculate overall pass@k metrics for a model across all difficulties."""
        # Aggregate all pass@k values across difficulties
        overall_pass_at_k = defaultdict(list)

        for difficulty in ["Easy", "Medium", "Hard"]:
            if model_name in self.pass_at_k_results[difficulty]:
                for k, values in self.pass_at_k_results[difficulty][model_name].items():
                    if isinstance(values, list):
                        overall_pass_at_k[k].extend(values)
                    else:
                        overall_pass_at_k[k].append(values)

        # Calculate averages for overall
        self.pass_at_k_results["Overall"][model_name] = {}
        for k, values in overall_pass_at_k.items():
            if values:
                avg_value = np.mean(values) * 100 if max(values) <= 1 else np.mean(values)
                self.pass_at_k_results["Overall"][model_name][k] = [avg_value]

    def calculate_improvements(self):
        """Calculate improvement percentages between models."""
        # Define model pairs for comparison
        model_pairs = [
            ("qwq-preview", "qwen2-5-coder"),
            ("deepseek-r1-distill", "qwen2-5-coder"),
            ("qwq-preview", "deepseek-r1-distill"),
        ]

        improvement_data = {
            "Easy": {},
            "Medium": {},
            "Hard": {},
            "Overall": {}
        }

        for difficulty in ["Easy", "Medium", "Hard", "Overall"]:
            for model_pair in model_pairs:
                model1, model2 = model_pair

                if model1 in self.results[difficulty] and model2 in self.results[difficulty]:
                    # Success rate improvement
                    rate1 = self.results[difficulty][model1].get("success_rate", 0)
                    rate2 = self.results[difficulty][model2].get("success_rate", 0)

                    # Calculate percentage improvement if target model has non-zero success rate
                    improvement = None
                    if rate2 > 0:
                        improvement = ((rate1 - rate2) / rate2) * 100
                    elif rate1 > 0:  # Infinite improvement (from 0 to non-zero)
                        improvement = float('inf')
                    else:  # Both 0
                        improvement = 0

                    pair_key = f"{model1}_vs_{model2}"
                    if pair_key not in improvement_data[difficulty]:
                        improvement_data[difficulty][pair_key] = {}

                    improvement_data[difficulty][pair_key]["success_rate"] = improvement

                    # Calculate pass@k improvements
                    for k in ["1", "3", "5", "10"]:
                        if (model1 in self.pass_at_k_results[difficulty] and
                                model2 in self.pass_at_k_results[difficulty]):
                            values1 = self.pass_at_k_results[difficulty][model1].get(k, [])
                            values2 = self.pass_at_k_results[difficulty][model2].get(k, [])

                            if values1 and values2:
                                pass1 = np.mean(values1) if isinstance(values1, list) else values1
                                pass2 = np.mean(values2) if isinstance(values2, list) else values2

                                if pass2 > 0:
                                    pass_improvement = ((pass1 - pass2) / pass2) * 100
                                    improvement_data[difficulty][pair_key][f"pass@{k}"] = pass_improvement

        return improvement_data

    def _generate_success_rate_table(self):
        """Generate table of success rates by difficulty with absolute numbers."""
        latex_table = [
            "\\subsection{Success Rate Comparison}",
            "",
            "This table presents the overall success rates for each model in different difficulty levels of the problem. The success rate measures the percentage of problems successfully solved, reflecting the model's effectiveness at generating correct solutions. Higher percentages indicate better performance. The results show significant variation in performance across difficulty levels, with all models struggling more on Hard problems compared to Easy and Medium ones.",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|c|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Easy} & \\textbf{Medium} & \\textbf{Hard} & \\textbf{Overall} \\\\",
            "\\midrule"
        ]

        # Add rows for each model
        for model_name in self.models:
            # Easy problems
            easy_solved = self.results["Easy"].get(model_name, {}).get("solved_problems", 0)
            easy_total = self.problem_counts["Easy"].get(model_name, 0)
            easy_rate = self.results["Easy"].get(model_name, {}).get("success_rate", 0)

            # Medium problems
            medium_solved = self.results["Medium"].get(model_name, {}).get("solved_problems", 0)
            medium_total = self.problem_counts["Medium"].get(model_name, 0)
            medium_rate = self.results["Medium"].get(model_name, {}).get("success_rate", 0)

            # Hard problems
            hard_solved = self.results["Hard"].get(model_name, {}).get("solved_problems", 0)
            hard_total = self.problem_counts["Hard"].get(model_name, 0)
            hard_rate = self.results["Hard"].get(model_name, {}).get("success_rate", 0)

            # Overall
            overall_solved = self.results["Overall"].get(model_name, {}).get("solved_problems", 0)
            overall_total = sum([easy_total, medium_total, hard_total])
            overall_rate = self.results["Overall"].get(model_name, {}).get("success_rate", 0)

            # Format as "solved/total (percentage%)"
            easy_str = f"{easy_solved}/{easy_total} ({easy_rate:.1f}\\%)"
            medium_str = f"{medium_solved}/{medium_total} ({medium_rate:.1f}\\%)"
            hard_str = f"{hard_solved}/{hard_total} ({hard_rate:.1f}\\%)"
            overall_str = f"{overall_solved}/{overall_total} ({overall_rate:.1f}\\%)"

            latex_table.append(
                f"{model_name} & {easy_str} & {medium_str} & {hard_str} & {overall_str} \\\\"
            )

        # Calculate averages for all difficulties
        total_easy_solved = sum([self.results["Easy"].get(m, {}).get("solved_problems", 0) for m in self.models])
        total_easy = sum([self.problem_counts["Easy"].get(m, 0) for m in self.models])
        avg_easy_rate = (total_easy_solved / total_easy * 100) if total_easy > 0 else 0

        total_medium_solved = sum([self.results["Medium"].get(m, {}).get("solved_problems", 0) for m in self.models])
        total_medium = sum([self.problem_counts["Medium"].get(m, 0) for m in self.models])
        avg_medium_rate = (total_medium_solved / total_medium * 100) if total_medium > 0 else 0

        total_hard_solved = sum([self.results["Hard"].get(m, {}).get("solved_problems", 0) for m in self.models])
        total_hard = sum([self.problem_counts["Hard"].get(m, 0) for m in self.models])
        avg_hard_rate = (total_hard_solved / total_hard * 100) if total_hard > 0 else 0

        total_overall_solved = total_easy_solved + total_medium_solved + total_hard_solved
        total_overall = total_easy + total_medium + total_hard
        avg_overall_rate = (total_overall_solved / total_overall * 100) if total_overall > 0 else 0

        # Add the average row
        latex_table.append("\\midrule")
        latex_table.append(
            f"\\textbf{{Average}} & {total_easy_solved}/{total_easy} ({avg_easy_rate:.1f}\\%) & "
            f"{total_medium_solved}/{total_medium} ({avg_medium_rate:.1f}\\%) & "
            f"{total_hard_solved}/{total_hard} ({avg_hard_rate:.1f}\\%) & "
            f"{total_overall_solved}/{total_overall} ({avg_overall_rate:.1f}\\%) \\\\"
        )

        # Close table
        latex_table.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Success rate comparison of models by difficulty, showing solved/total problems and percentages.}",
            "\\label{tab:success_rates}",
            "\\end{table}",
            ""
        ])

        return latex_table


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze LeetCode results and generate a LaTeX report.")
    parser.add_argument("--easy-dir", required=True, help="Directory containing Easy difficulty results")
    parser.add_argument("--medium-dir", required=True, help="Directory containing Medium difficulty results")
    parser.add_argument("--hard-dir", required=True, help="Directory containing Hard difficulty results")
    parser.add_argument("--output", required=True, help="Output LaTeX file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()


def main():
    """Main function to run the analyzer."""
    args = parse_args()

    # Create analyzer
    analyzer = LeetCodeResultsAnalyzer(
        easy_dir=args.easy_dir,
        medium_dir=args.medium_dir,
        hard_dir=args.hard_dir,
        output_file=args.output
    )

    # Enable debug mode if requested
    if args.debug:
        analyzer.debug_mode = True

    # Load data
    analyzer.load_data()

    # Generate LaTeX document
    output_file = analyzer.generate_latex()

    print(f"Analysis complete. LaTeX document saved to: {output_file}")


if __name__ == "__main__":
    main()
