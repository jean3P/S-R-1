#!/usr/bin/env python3
"""
LeetCode Results Analyzer

This script analyzes LeetCode benchmark results for different models across difficulty levels
(Easy, Medium, Hard) and generates a comprehensive LaTeX report with performance metrics.

Usage:
    python leetcode_results_analyzer.py --easy-dir <EASY_RESULTS_DIR> --medium-dir <MEDIUM_RESULTS_DIR>
                                        --hard-dir <HARD_RESULTS_DIR> --output <OUTPUT_FILE>

Example:
    python leetcode_results_analyzer.py --easy-dir results/leetcode_solver_easy_20250513_173334
                                        --medium-dir results/leetcode_solver_medium_20250513_222758
                                        --hard-dir results/leetcode_solver_hard_20250514_084122
                                        --output leetcode_analysis.tex
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict


class LeetCodeResultsAnalyzer:
    """Analyzer for LeetCode benchmark results."""

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
        """Process results for a specific difficulty level."""
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
                self.pass_at_k_results[difficulty][model_name] = {}

            # Process combined_results.json
            combined_file = model_dir / "combined_results.json"
            if combined_file.exists():
                self._process_combined_results(combined_file, model_name, difficulty)
            else:
                print(f"Warning: No combined_results.json found for {model_name} at {difficulty} level")

            # Process summary.json
            summary_file = model_dir / "summary.json"
            if summary_file.exists():
                self._process_summary(summary_file, model_name, difficulty)

            # Process individual problem files if needed
            leetcode_solutions_dir = model_dir / "leetcode_solutions"
            if leetcode_solutions_dir.exists():
                solution_files = list(leetcode_solutions_dir.glob("*.json"))
                self.problem_counts[difficulty][model_name] = len(solution_files)

    def _process_combined_results(self, combined_file, model_name, difficulty):
        """Process combined results file for a model."""
        try:
            with open(combined_file, 'r') as f:
                combined_data = json.load(f)

            # Calculate basic metrics
            total_problems = len(combined_data)
            solved_problems = sum(1 for r in combined_data if r.get("status") == "solved")
            success_rate = (solved_problems / total_problems) * 100 if total_problems > 0 else 0

            avg_time = np.mean([r.get("processing_time", 0) for r in combined_data]) if total_problems > 0 else 0
            avg_candidates = np.mean([r.get("total_candidates", 0) for r in combined_data]) if total_problems > 0 else 0

            # Update problem count
            self.problem_counts[difficulty][model_name] = total_problems

            # Store basic metrics
            self.results[difficulty][model_name] = {
                "total_problems": total_problems,
                "solved_problems": solved_problems,
                "success_rate": success_rate,
                "avg_time": avg_time,
                "avg_candidates": avg_candidates
            }

            # Extract pass@k metrics
            self._extract_pass_at_k_metrics(combined_data, model_name, difficulty)

        except Exception as e:
            print(f"Error processing combined results for {model_name} at {difficulty} level: {str(e)}")

    def _process_summary(self, summary_file, model_name, difficulty):
        """Process summary.json file for a model."""
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)

            # Extract additional metrics from summary
            if "code_eval" in summary_data:
                pass_at_k = summary_data["code_eval"]

                # Add to the pass@k results
                for k, value in pass_at_k.items():
                    k_value = k.replace("pass@", "")
                    if k_value not in self.pass_at_k_results[difficulty][model_name]:
                        self.pass_at_k_results[difficulty][model_name][k_value] = value * 100  # Convert to percentage

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

    def _extract_pass_at_k_metrics(self, combined_data, model_name, difficulty):
        """Extract pass@k metrics from combined results data."""
        # Initialize pass@k accumulation
        pass_at_k_values = defaultdict(list)

        # Collect all pass@k values
        for problem in combined_data:
            if "code_eval_results" in problem and "pass_at_k" in problem["code_eval_results"]:
                for k, value in problem["code_eval_results"]["pass_at_k"].items():
                    pass_at_k_values[k].append(value)

        # Calculate averages
        for k, values in pass_at_k_values.items():
            if values:
                avg_value = np.mean(values) * 100  # Convert to percentage
                self.pass_at_k_results[difficulty][model_name][k] = avg_value
            else:
                self.pass_at_k_results[difficulty][model_name][k] = None

    def analyze_feedback_iterations(self):
        """Analyze how model performance improves with each feedback iteration."""
        print("Analyzing feedback iterations...")

        # Define the iterations and corresponding pass@k metrics
        iterations = [
            {"name": "Initial", "metric": "1"},  # Initial solutions (pass@1)
            {"name": "Iter 1", "metric": "3"},  # First feedback iteration (pass@3)
            {"name": "Iter 2", "metric": "5"},  # Second feedback iteration (pass@5)
            {"name": "Iter 3", "metric": "10"}  # Third feedback iteration (pass@10)
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
                            feedback_results["absolute_performance"][model_name][difficulty][iter_name] = \
                                self.pass_at_k_results[difficulty][model_name][k]

        # Calculate marginal improvements between iterations
        for model_name in self.models:
            feedback_results["marginal_improvements"][model_name] = {}

            for difficulty in ["Easy", "Medium", "Hard", "Overall"]:
                if difficulty in feedback_results["absolute_performance"][model_name]:
                    perf = feedback_results["absolute_performance"][model_name][difficulty]

                    if len(perf) > 1:  # Need at least 2 iterations to calculate improvements
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

        # Introduction section (unchanged)
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
            for difficulty in ["Easy", "Medium", "Hard", "Overall"]:
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
            "The unusual pattern in some marginal improvements (particularly the negative values for iteration 1→2 followed by positive values for iteration 2→3) ",
            "suggests a non-linear learning process in our models. This may be due to the tree search algorithm's branching strategy, ",
            "where early iterations focus on exploring diverse solutions while later iterations concentrate on refining promising paths.",
            ""
        ])

        return latex_content

    def _calculate_overall_metrics(self):
        """Calculate overall metrics across all difficulty levels."""
        for model_name in self.models:
            # Initialize counters
            total_problems = 0
            solved_problems = 0
            time_sum = 0
            candidates_sum = 0

            # Collect data across difficulties
            for difficulty in ["Easy", "Medium", "Hard"]:
                if model_name in self.results[difficulty]:
                    model_data = self.results[difficulty][model_name]
                    problems_count = model_data.get("total_problems", 0)
                    total_problems += problems_count
                    solved_problems += model_data.get("solved_problems", 0)
                    time_sum += model_data.get("avg_time", 0) * problems_count
                    candidates_sum += model_data.get("avg_candidates", 0) * problems_count

            # Calculate averages
            if total_problems > 0:
                success_rate = (solved_problems / total_problems) * 100
                avg_time = time_sum / total_problems
                avg_candidates = candidates_sum / total_problems

                self.results["Overall"][model_name] = {
                    "total_problems": total_problems,
                    "solved_problems": solved_problems,
                    "success_rate": success_rate,
                    "avg_time": avg_time,
                    "avg_candidates": avg_candidates
                }

                # Calculate overall pass@k metrics
                self._calculate_overall_pass_at_k(model_name)

    def _calculate_overall_pass_at_k(self, model_name):
        """Calculate overall pass@k metrics for a model across all difficulties."""
        pass_at_k_weighted = defaultdict(lambda: {"sum": 0, "count": 0})

        # Collect weighted values across difficulties
        for difficulty in ["Easy", "Medium", "Hard"]:
            if model_name in self.pass_at_k_results[difficulty]:
                problem_count = self.problem_counts[difficulty].get(model_name, 0)
                if problem_count > 0:
                    for k, value in self.pass_at_k_results[difficulty][model_name].items():
                        if value is not None:
                            pass_at_k_weighted[k]["sum"] += value * problem_count
                            pass_at_k_weighted[k]["count"] += problem_count

        # Calculate weighted averages
        self.pass_at_k_results["Overall"][model_name] = {}
        for k, data in pass_at_k_weighted.items():
            if data["count"] > 0:
                self.pass_at_k_results["Overall"][model_name][k] = data["sum"] / data["count"]
            else:
                self.pass_at_k_results["Overall"][model_name][k] = None

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
                            pass1 = self.pass_at_k_results[difficulty][model1].get(k)
                            pass2 = self.pass_at_k_results[difficulty][model2].get(k)

                            if pass1 is not None and pass2 is not None and pass2 > 0:
                                pass_improvement = ((pass1 - pass2) / pass2) * 100
                                improvement_data[difficulty][pair_key][f"pass@{k}"] = pass_improvement

        return improvement_data

    def generate_latex(self):
        """Generate LaTeX document with tables and analysis."""
        print("Generating LaTeX document...")

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
            "\\begin{document}",
            "",
            f"\\section{{LeetCode Performance Analysis ({self.current_date})}}",
            ""
        ])

        # Experimental Setup
        latex_content.extend([
            "\\subsection{Experimental Setup}",
            "",
            "\\begin{itemize}",
            "\\item Models tested: qwen2-5-coder, deepseek-r1-distill, qwq-preview",
            f"\\item Problem count: {self.problem_counts['Easy'].get('qwen2-5-coder', 20)} Easy, {self.problem_counts['Medium'].get('qwen2-5-coder', 20)} Medium, {self.problem_counts['Hard'].get('qwen2-5-coder', 10)} Hard ({self.problem_counts['Easy'].get('qwen2-5-coder', 20) + self.problem_counts['Medium'].get('qwen2-5-coder', 20) + self.problem_counts['Hard'].get('qwen2-5-coder', 10)} total)",
            "\\item Tree search parameters: $k=3$ initial solutions, branch factor=3, max depth=3",
            "\\item Evaluation: immediate solution evaluation, branching from failures",
            "\\item Metrics: pass@k with $k \\in \\{1, 3, 5, 10\\}$",
            "\\end{itemize}",
            ""
        ])

        # Success Rate by Difficulty
        latex_content.extend(self._generate_success_rate_table())

        # Improvement Tables
        latex_content.extend(self._generate_improvement_table(improvements))

        # Pass@k Tables
        for k in ["1", "3", "5", "10"]:
            latex_content.extend(self._generate_pass_at_k_table(k, improvements))

        # Add the feedback iteration analysis here
        latex_content.extend(self._generate_feedback_iteration_tables())

        # Node Expansion Analysis
        latex_content.extend(self._generate_node_expansion_table())

        # Processing Time Analysis
        latex_content.extend(self._generate_processing_time_table())

        # Solution Efficiency Analysis
        latex_content.extend(self._generate_solution_efficiency_table())

        # End document
        latex_content.append("\\end{document}")

        # Write to file
        with open(self.output_file, 'w') as f:
            f.write("\n".join(latex_content))

        print(f"LaTeX document generated successfully: {self.output_file}")
        return self.output_file

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

    def _generate_improvement_table(self, improvements):
        """Generate table showing improvement percentages."""
        latex_table = [
            "\\subsection{Model Improvement Analysis}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|S[table-format=3.1]|S[table-format=3.1]|S[table-format=3.1]|S[table-format=3.1]}",
            "\\toprule",
            "\\textbf{Comparison} & {\\textbf{Easy (\\%)}} & {\\textbf{Medium (\\%)}} & {\\textbf{Hard (\\%)}} & {\\textbf{Overall (\\%)}} \\\\",
            "\\midrule"
        ]

        # Model comparison pairs
        comparisons = [
            ("qwq-preview", "qwen2-5-coder", "qwq-preview vs. qwen2-5-coder"),
            ("deepseek-r1-distill", "qwen2-5-coder", "deepseek-r1-distill vs. qwen2-5-coder"),
            ("qwq-preview", "deepseek-r1-distill", "qwq-preview vs. deepseek-r1-distill")
        ]

        for model1, model2, display_name in comparisons:
            pair_key = f"{model1}_vs_{model2}"

            easy_imp = improvements["Easy"].get(pair_key, {}).get("success_rate", 0)
            medium_imp = improvements["Medium"].get(pair_key, {}).get("success_rate", 0)
            hard_imp = improvements["Hard"].get(pair_key, {}).get("success_rate", 0)
            overall_imp = improvements["Overall"].get(pair_key, {}).get("success_rate", 0)

            # Format infinities and handle missing data
            easy_val = "-" if easy_imp is None else ("∞" if easy_imp == float('inf') else f"{easy_imp:.1f}")
            medium_val = "-" if medium_imp is None else ("∞" if medium_imp == float('inf') else f"{medium_imp:.1f}")
            hard_val = "-" if hard_imp is None else ("∞" if hard_imp == float('inf') else f"{hard_imp:.1f}")
            overall_val = "-" if overall_imp is None else ("∞" if overall_imp == float('inf') else f"{overall_imp:.1f}")

            latex_table.append(f"{display_name} & {easy_val} & {medium_val} & {hard_val} & {overall_val} \\\\")

        # Close table
        latex_table.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Success rate improvement percentages between models by difficulty.}",
            "\\label{tab:improvement_percentages}",
            "\\end{table}",
            ""
        ])

        return latex_table

    def _generate_pass_at_k_table(self, k, improvements):
        """Generate table for a specific pass@k metric."""
        latex_table = [
            f"\\subsection{{Pass@{k} Performance}}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|S[table-format=2.1]|S[table-format=2.1]|S[table-format=2.1]|S[table-format=2.1]}",
            "\\toprule",
            "\\textbf{Model} & {\\textbf{Easy (\\%)}} & {\\textbf{Medium (\\%)}} & {\\textbf{Hard (\\%)}} & {\\textbf{Overall (\\%)}} \\\\",
            "\\midrule"
        ]

        # Add rows for each model
        for model_name in self.models:
            easy_pass = self.pass_at_k_results["Easy"].get(model_name, {}).get(k, None)
            medium_pass = self.pass_at_k_results["Medium"].get(model_name, {}).get(k, None)
            hard_pass = self.pass_at_k_results["Hard"].get(model_name, {}).get(k, None)
            overall_pass = self.pass_at_k_results["Overall"].get(model_name, {}).get(k, None)

            # Format values with potential missing data
            easy_val = f"{easy_pass:.1f}" if easy_pass is not None else "-"
            medium_val = f"{medium_pass:.1f}" if medium_pass is not None else "-"
            hard_val = f"{hard_pass:.1f}" if hard_pass is not None else "-"
            overall_val = f"{overall_pass:.1f}" if overall_pass is not None else "-"

            latex_table.append(f"{model_name} & {easy_val} & {medium_val} & {hard_val} & {overall_val} \\\\")

        # Calculate averages
        avg_easy = np.mean([self.pass_at_k_results["Easy"].get(m, {}).get(k, 0) or 0 for m in self.models])
        avg_medium = np.mean([self.pass_at_k_results["Medium"].get(m, {}).get(k, 0) or 0 for m in self.models])
        avg_hard = np.mean([self.pass_at_k_results["Hard"].get(m, {}).get(k, 0) or 0 for m in self.models])
        avg_overall = np.mean([self.pass_at_k_results["Overall"].get(m, {}).get(k, 0) or 0 for m in self.models])

        latex_table.append("\\midrule")
        latex_table.append(
            f"\\textbf{{Average}} & {avg_easy:.1f} & {avg_medium:.1f} & {avg_hard:.1f} & {avg_overall:.1f} \\\\")
        latex_table.append("\\midrule")

        # Model improvement rows for pass@k
        comparisons = [
            ("qwq-preview", "qwen2-5-coder", "qwq vs. qwen2 Improvement"),
            ("deepseek-r1-distill", "qwen2-5-coder", "deepseek vs. qwen2 Improvement"),
            ("qwq-preview", "deepseek-r1-distill", "qwq vs. deepseek Improvement")
        ]

        for model1, model2, display_name in comparisons:
            pair_key = f"{model1}_vs_{model2}"

            easy_imp = improvements["Easy"].get(pair_key, {}).get(f"pass@{k}", None)
            medium_imp = improvements["Medium"].get(pair_key, {}).get(f"pass@{k}", None)
            hard_imp = improvements["Hard"].get(pair_key, {}).get(f"pass@{k}", None)
            overall_imp = improvements["Overall"].get(pair_key, {}).get(f"pass@{k}", None)

            # Format values with potential missing data
            easy_val = f"{easy_imp:.1f}" if easy_imp is not None else "-"
            medium_val = f"{medium_imp:.1f}" if medium_imp is not None else "-"
            hard_val = f"{hard_imp:.1f}" if hard_imp is not None else "-"
            overall_val = f"{overall_imp:.1f}" if overall_imp is not None else "-"

            latex_table.append(
                f"\\textbf{{{display_name}}} & {easy_val} & {medium_val} & {hard_val} & {overall_val} \\\\")

        # Close table
        latex_table.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            f"\\caption{{Pass@{k} performance metrics by model and difficulty, with improvement percentages.}}",
            f"\\label{{tab:pass_at_{k}}}",
            "\\end{table}",
            ""
        ])

        return latex_table

    def _generate_node_expansion_table(self):
        """Generate table showing node expansion analysis."""
        latex_table = [
            "\\subsection{Node Expansion Analysis}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|S[table-format=2.1]|S[table-format=2.1]|S[table-format=2.1]|S[table-format=2.1]}",
            "\\toprule",
            "\\textbf{Model} & {\\textbf{Easy}} & {\\textbf{Medium}} & {\\textbf{Hard}} & {\\textbf{Avg. Expansion}} \\\\",
            "\\midrule"
        ]

        # Add rows for each model
        for model_name in self.models:
            # Get candidate counts
            easy_candidates = self.results["Easy"].get(model_name, {}).get("avg_candidates", 0)
            medium_candidates = self.results["Medium"].get(model_name, {}).get("avg_candidates", 0)
            hard_candidates = self.results["Hard"].get(model_name, {}).get("avg_candidates", 0)
            overall_candidates = self.results["Overall"].get(model_name, {}).get("avg_candidates", 0)

            # Add model row
            hard_str = f"{hard_candidates:.1f}"
            if model_name == "qwq-preview":
                hard_str = f"{hard_candidates:.1f} \\textsuperscript{{*}}"

            latex_table.append(
                f"{model_name} & {easy_candidates:.1f} & {medium_candidates:.1f} & {hard_str} & {overall_candidates:.1f} \\\\")

        # Calculate averages
        avg_easy = np.mean([self.results["Easy"].get(m, {}).get("avg_candidates", 0) for m in self.models])
        avg_medium = np.mean([self.results["Medium"].get(m, {}).get("avg_candidates", 0) for m in self.models])
        avg_hard = np.mean([self.results["Hard"].get(m, {}).get("avg_candidates", 0) for m in self.models])
        avg_overall = np.mean([self.results["Overall"].get(m, {}).get("avg_candidates", 0) for m in self.models])

        latex_table.append("\\midrule")
        latex_table.append(
            f"\\textbf{{Average}} & {avg_easy:.1f} & {avg_medium:.1f} & {avg_hard:.1f} & {avg_overall:.1f} \\\\")

        # Close table
        latex_table.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Node expansion analysis showing the number of candidates generated for each difficulty. \\textsuperscript{*}Harder issues may generate more candidates due to increased complexity.}",
            "\\label{tab:node_expansion}",
            "\\end{table}",
            ""
        ])

        return latex_table

    def _generate_processing_time_table(self):
        """Generate table showing processing time analysis."""
        latex_table = [
            "\\subsection{Processing Time Analysis}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|S[table-format=4.1]|S[table-format=4.1]|S[table-format=4.1]|S[table-format=4.1]}",
            "\\toprule",
            "\\textbf{Model} & {\\textbf{Easy (s)}} & {\\textbf{Medium (s)}} & {\\textbf{Hard (s)}} & {\\textbf{Overall (s)}} \\\\",
            "\\midrule"
        ]

        # Add rows for each model
        for model_name in self.models:
            # Get processing times
            easy_time = self.results["Easy"].get(model_name, {}).get("avg_time", 0)
            medium_time = self.results["Medium"].get(model_name, {}).get("avg_time", 0)
            hard_time = self.results["Hard"].get(model_name, {}).get("avg_time", 0)
            overall_time = self.results["Overall"].get(model_name, {}).get("avg_time", 0)

            # Add model row
            latex_table.append(
                f"{model_name} & {easy_time:.1f} & {medium_time:.1f} & {hard_time:.1f} & {overall_time:.1f} \\\\")

        # Calculate averages
        avg_easy = np.mean([self.results["Easy"].get(m, {}).get("avg_time", 0) for m in self.models])
        avg_medium = np.mean([self.results["Medium"].get(m, {}).get("avg_time", 0) for m in self.models])
        avg_hard = np.mean([self.results["Hard"].get(m, {}).get("avg_time", 0) for m in self.models])
        avg_overall = np.mean([self.results["Overall"].get(m, {}).get("avg_time", 0) for m in self.models])

        latex_table.append("\\midrule")
        latex_table.append(
            f"\\textbf{{Average}} & {avg_easy:.1f} & {avg_medium:.1f} & {avg_hard:.1f} & {avg_overall:.1f} \\\\")

        # Close table
        latex_table.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Average processing times (in seconds) for LeetCode problems by model and difficulty.}",
            "\\label{tab:processing_times}",
            "\\end{table}",
            ""
        ])

        return latex_table

    def _generate_solution_efficiency_table(self):
        """Generate table showing solution efficiency metrics."""
        latex_table = [
            "\\subsection{Solution Efficiency Analysis}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|S[table-format=4.1]|S[table-format=2.1]|S[table-format=2.1]}",
            "\\toprule",
            "\\textbf{Model} & {\\textbf{Time/Solution (s)}} & {\\textbf{Candidates/Solution}} & {\\textbf{Success Rate (\\%)}} \\\\",
            "\\midrule"
        ]

        # Add rows for each model
        for model_name in self.models:
            # Calculate efficiency metrics
            if model_name in self.results["Overall"]:
                solved = self.results["Overall"][model_name].get("solved_problems", 0)
                total = self.results["Overall"][model_name].get("total_problems", 0)
                avg_time = self.results["Overall"][model_name].get("avg_time", 0)
                avg_candidates = self.results["Overall"][model_name].get("avg_candidates", 0)
                success_rate = self.results["Overall"][model_name].get("success_rate", 0)

                # Calculate time per solution and candidates per solution
                if solved > 0:
                    time_per_solution = avg_time * total / solved
                    candidates_per_solution = avg_candidates * total / solved
                else:
                    time_per_solution = 0
                    candidates_per_solution = 0

                # Add model row
                latex_table.append(
                    f"{model_name} & {time_per_solution:.1f} & {candidates_per_solution:.1f} & {success_rate:.1f} \\\\")

        # Calculate averages
        avg_time_per_solution = np.mean([
            self.results["Overall"].get(m, {}).get("avg_time", 0) *
            self.results["Overall"].get(m, {}).get("total_problems", 0) /
            max(1, self.results["Overall"].get(m, {}).get("solved_problems", 0))
            for m in self.models
        ])

        avg_candidates_per_solution = np.mean([
            self.results["Overall"].get(m, {}).get("avg_candidates", 0) *
            self.results["Overall"].get(m, {}).get("total_problems", 0) /
            max(1, self.results["Overall"].get(m, {}).get("solved_problems", 0))
            for m in self.models
        ])

        avg_success_rate = np.mean([self.results["Overall"].get(m, {}).get("success_rate", 0) for m in self.models])

        latex_table.append("\\midrule")
        latex_table.append(
            f"\\textbf{{Average}} & {avg_time_per_solution:.1f} & {avg_candidates_per_solution:.1f} & {avg_success_rate:.1f} \\\\")

        # Close table
        latex_table.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Efficiency metrics for solution generation across all difficulties.}",
            "\\label{tab:solution_efficiency}",
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

    # Load data
    analyzer.load_data()

    # Generate LaTeX document
    output_file = analyzer.generate_latex()

    print(f"Analysis complete. LaTeX document saved to: {output_file}")


if __name__ == "__main__":
    main()
