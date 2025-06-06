#!/usr/bin/env python3
"""
LeetCode True Baseline Results Analyzer

Analyzes baseline experiment results where models generate only 1 solution per problem.
No multiple attempts, just a single shot - true baseline for comparison.
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


class TrueBaselineResultsAnalyzer:
    """Analyzer for LeetCode true baseline experiment results (1 solution per problem)."""

    def __init__(self, results_dir, output_file):
        """Initialize the baseline analyzer."""
        self.results_dir = Path(results_dir)
        self.output_file = Path(output_file)

        # Models to analyze
        self.models = ["qwen2-5-coder", "deepseek-r1-distill", "qwq-preview"]

        # Difficulties
        self.difficulties = ["Easy", "Medium", "Hard"]

        # Initialize data structures
        self.results = defaultdict(lambda: defaultdict(dict))
        self.problem_details = defaultdict(list)
        self.error_analysis = defaultdict(lambda: defaultdict(int))
        self.execution_times = defaultdict(list)

        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def load_data(self):
        """Load baseline results from the results directory."""
        print("Loading true baseline results (1 solution per problem)...")

        for model in self.models:
            for difficulty in self.difficulties:
                # Look for individual problem files in baseline_results directory
                baseline_dir = self.results_dir / model / difficulty / "baseline_results"

                if not baseline_dir.exists():
                    print(f"Warning: No baseline results directory found for {model}/{difficulty}")
                    continue

                # Process each problem JSON file
                problem_files = list(baseline_dir.glob("*.json"))
                print(f"Found {len(problem_files)} problem files for {model}/{difficulty}")

                total_problems = 0
                solved_problems = 0

                for problem_file in problem_files:
                    with open(problem_file, 'r') as f:
                        problem_result = json.load(f)

                    total_problems += 1

                    # Extract key information
                    problem_id = problem_result.get("problem_id")
                    status = problem_result.get("status")
                    passed = problem_result.get("passed", False)

                    if status == "solved" or passed:
                        solved_problems += 1

                    # Store detailed information
                    self.problem_details[f"{model}_{difficulty}"].append({
                        "problem_id": problem_id,
                        "problem_title": problem_result.get("problem_title"),
                        "status": status,
                        "passed": passed,
                        "error_type": problem_result.get("error_type", "none"),
                        "processing_time": problem_result.get("processing_time", 0),
                        "execution_time": problem_result.get("stats", {}).get("execution_time", 0)
                    })

                    # Track execution times
                    if "processing_time" in problem_result:
                        self.execution_times[f"{model}_{difficulty}"].append(
                            problem_result["processing_time"]
                        )

                    # Track errors
                    if not passed and "error_type" in problem_result:
                        self.error_analysis[model][problem_result["error_type"]] += 1

                # Store aggregate results
                solve_rate = (solved_problems / total_problems * 100) if total_problems > 0 else 0
                self.results[difficulty][model] = {
                    "total_problems": total_problems,
                    "solved": solved_problems,
                    "solve_rate": solve_rate
                }

        # Calculate overall metrics
        self._calculate_overall_metrics()

        print("Data loading complete.")

    def _calculate_overall_metrics(self):
        """Calculate overall metrics across all difficulties."""
        for model in self.models:
            total_problems = 0
            total_solved = 0

            for difficulty in self.difficulties:
                if model in self.results[difficulty]:
                    total_problems += self.results[difficulty][model]["total_problems"]
                    total_solved += self.results[difficulty][model]["solved"]

            if total_problems > 0:
                self.results["Overall"][model] = {
                    "total_problems": total_problems,
                    "solved": total_solved,
                    "solve_rate": (total_solved / total_problems) * 100
                }

    def generate_latex(self):
        """Generate LaTeX document with baseline analysis tables."""
        print("Generating LaTeX document for true baseline results...")

        latex_content = []

        # Document preamble
        latex_content.extend([
            "\\documentclass[10pt]{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{multirow}",
            "\\usepackage{array}",
            "\\usepackage{adjustbox}",
            "\\usepackage{amsmath}",
            "\\usepackage{xcolor}",
            "\\begin{document}",
            "",
            f"\\section{{LeetCode True Baseline Analysis ({self.current_date})}}",
            "",
            "\\subsection{Experimental Setup}",
            "",
            "\\begin{itemize}",
            "\\item \\textbf{Models}: qwen2-5-coder, deepseek-r1-distill, qwq-preview",
            "\\item \\textbf{Approach}: Single solution attempt per problem (true baseline)",
            "\\item \\textbf{Evaluation}: Direct test execution with Pass@1",
            "\\item \\textbf{Dataset}: 18 Easy, 20 Medium, 10 Hard problems (48 total)",
            "\\end{itemize}",
            ""
        ])

        # Add all tables
        latex_content.extend(self._generate_baseline_performance_table())
        latex_content.extend(self._generate_difficulty_breakdown_table())
        latex_content.extend(self._generate_error_analysis_table())
        latex_content.extend(self._generate_execution_time_table())
        latex_content.extend(self._generate_comparison_table())
        latex_content.extend(self._generate_detailed_results_table())

        # End document
        latex_content.append("\\end{document}")

        # Write to file
        with open(self.output_file, 'w') as f:
            f.write("\n".join(latex_content))

        print(f"LaTeX document generated: {self.output_file}")

    def _generate_baseline_performance_table(self):
        """Generate overall baseline performance table."""
        latex = [
            "\\subsection{True Baseline Performance (Single Attempt)}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|c|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Easy} & \\textbf{Medium} & \\textbf{Hard} & \\textbf{Overall} \\\\",
            "\\midrule"
        ]

        for model in self.models:
            row = [model]

            for difficulty in self.difficulties + ["Overall"]:
                if model in self.results[difficulty]:
                    data = self.results[difficulty][model]
                    solved = data["solved"]
                    total = data["total_problems"]
                    rate = data["solve_rate"]
                    cell = f"{solved}/{total} ({rate:.1f}\\%)"
                else:
                    cell = "-"
                row.append(cell)

            latex.append(" & ".join(row) + " \\\\")

        # Add average row
        latex.append("\\midrule")
        avg_row = ["\\textbf{Average}"]
        for difficulty in self.difficulties + ["Overall"]:
            total_solved = sum(self.results[difficulty].get(m, {}).get("solved", 0) for m in self.models)
            total_problems = sum(self.results[difficulty].get(m, {}).get("total_problems", 0) for m in self.models)
            if total_problems > 0:
                avg_rate = (total_solved / total_problems) * 100
                avg_row.append(f"{total_solved}/{total_problems} ({avg_rate:.1f}\\%)")
            else:
                avg_row.append("-")
        latex.append(" & ".join(avg_row) + " \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{True baseline performance with single solution attempt per problem. This represents the models' ability to solve problems on their first try without any feedback or retry mechanism.}",
            "\\label{tab:true_baseline_performance}",
            "\\end{table}",
            ""
        ])

        return latex

    def _generate_difficulty_breakdown_table(self):
        """Generate detailed breakdown by difficulty."""
        latex = [
            "\\subsection{Success Rate by Problem Difficulty}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|ccc|ccc|ccc}",
            "\\toprule",
            "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{3}{c|}{\\textbf{Easy}} & \\multicolumn{3}{c|}{\\textbf{Medium}} & \\multicolumn{3}{c}{\\textbf{Hard}} \\\\",
            "& Solved & Total & Rate & Solved & Total & Rate & Solved & Total & Rate \\\\",
            "\\midrule"
        ]

        for model in self.models:
            row = [model]

            for difficulty in self.difficulties:
                if model in self.results[difficulty]:
                    data = self.results[difficulty][model]
                    solved = data["solved"]
                    total = data["total_problems"]
                    rate = data["solve_rate"]
                    row.extend([str(solved), str(total), f"{rate:.1f}\\%"])
                else:
                    row.extend(["-", "-", "-"])

            latex.append(" & ".join(row) + " \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Detailed breakdown of success rates by difficulty level, showing the challenge progression from Easy to Hard problems.}",
            "\\label{tab:difficulty_breakdown}",
            "\\end{table}",
            ""
        ])

        return latex

    def _generate_error_analysis_table(self):
        """Generate error type analysis table."""
        latex = [
            "\\subsection{Error Type Distribution}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|c|c}",
            "\\toprule",
            "\\textbf{Error Type} & \\textbf{qwen2-5-coder} & \\textbf{deepseek-r1-distill} & \\textbf{qwq-preview} & \\textbf{Total} \\\\",
            "\\midrule"
        ]

        # Get all error types
        all_error_types = set()
        for model in self.models:
            all_error_types.update(self.error_analysis[model].keys())

        # Sort error types by total frequency
        error_totals = {}
        for error_type in all_error_types:
            total = sum(self.error_analysis[model].get(error_type, 0) for model in self.models)
            error_totals[error_type] = total

        sorted_errors = sorted(error_totals.items(), key=lambda x: x[1], reverse=True)

        for error_type, total in sorted_errors:
            if total == 0:
                continue

            row = [error_type.replace("_", " ").title()]

            for model in self.models:
                count = self.error_analysis[model].get(error_type, 0)
                row.append(str(count))

            row.append(f"\\textbf{{{total}}}")
            latex.append(" & ".join(row) + " \\\\")

        # Add totals row
        latex.append("\\midrule")
        row = ["\\textbf{Total Errors}"]
        total_all = 0
        for model in self.models:
            model_total = sum(self.error_analysis[model].values())
            row.append(f"\\textbf{{{model_total}}}")
            total_all += model_total
        row.append(f"\\textbf{{{total_all}}}")
        latex.append(" & ".join(row) + " \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Distribution of error types in failed solution attempts. This shows which types of errors are most common when models fail to solve problems on their first attempt.}",
            "\\label{tab:error_distribution}",
            "\\end{table}",
            ""
        ])

        return latex

    def _generate_execution_time_table(self):
        """Generate execution time analysis table."""
        latex = [
            "\\subsection{Execution Time Analysis}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|c|c}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Avg. Time (s)} & \\textbf{Min Time (s)} & \\textbf{Max Time (s)} & \\textbf{Std Dev (s)} \\\\",
            "\\midrule"
        ]

        for model in self.models:
            # Aggregate execution times across all difficulties
            all_times = []
            for difficulty in self.difficulties:
                key = f"{model}_{difficulty}"
                if key in self.execution_times:
                    all_times.extend(self.execution_times[key])

            if all_times:
                avg_time = np.mean(all_times)
                min_time = np.min(all_times)
                max_time = np.max(all_times)
                std_time = np.std(all_times)

                latex.append(
                    f"{model} & {avg_time:.2f} & {min_time:.2f} & "
                    f"{max_time:.2f} & {std_time:.2f} \\\\"
                )
            else:
                latex.append(f"{model} & - & - & - & - \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Execution time statistics for single solution generation, including model inference and testing time.}",
            "\\label{tab:execution_time}",
            "\\end{table}",
            ""
        ])

        return latex

    def _generate_comparison_table(self):
        """Generate model comparison table."""
        latex = [
            "\\subsection{Model Performance Comparison}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|c|c|c}",
            "\\toprule",
            "\\textbf{Metric} & \\textbf{qwen2-5-coder} & \\textbf{deepseek-r1-distill} & \\textbf{qwq-preview} \\\\",
            "\\midrule"
        ]

        # Overall success rate
        row = ["Overall Success Rate"]
        for model in self.models:
            if model in self.results["Overall"]:
                rate = self.results["Overall"][model]["solve_rate"]
                row.append(f"{rate:.1f}\\%")
            else:
                row.append("-")
        latex.append(" & ".join(row) + " \\\\")

        # Easy problems success
        row = ["Easy Problems"]
        for model in self.models:
            if model in self.results["Easy"]:
                rate = self.results["Easy"][model]["solve_rate"]
                row.append(f"{rate:.1f}\\%")
            else:
                row.append("-")
        latex.append(" & ".join(row) + " \\\\")

        # Medium problems success
        row = ["Medium Problems"]
        for model in self.models:
            if model in self.results["Medium"]:
                rate = self.results["Medium"][model]["solve_rate"]
                row.append(f"{rate:.1f}\\%")
            else:
                row.append("-")
        latex.append(" & ".join(row) + " \\\\")

        # Hard problems success
        row = ["Hard Problems"]
        for model in self.models:
            if model in self.results["Hard"]:
                rate = self.results["Hard"][model]["solve_rate"]
                row.append(f"{rate:.1f}\\%")
            else:
                row.append("-")
        latex.append(" & ".join(row) + " \\\\")

        latex.append("\\midrule")

        # Average execution time
        row = ["Avg. Execution Time"]
        for model in self.models:
            all_times = []
            for difficulty in self.difficulties:
                key = f"{model}_{difficulty}"
                if key in self.execution_times:
                    all_times.extend(self.execution_times[key])

            if all_times:
                avg_time = np.mean(all_times)
                row.append(f"{avg_time:.2f}s")
            else:
                row.append("-")
        latex.append(" & ".join(row) + " \\\\")

        # Most common error
        row = ["Most Common Error"]
        for model in self.models:
            if self.error_analysis[model]:
                most_common = max(self.error_analysis[model].items(), key=lambda x: x[1])
                row.append(most_common[0].replace("_", " ").title())
            else:
                row.append("N/A")
        latex.append(" & ".join(row) + " \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Comprehensive comparison of model performance metrics on single-attempt baseline.}",
            "\\label{tab:model_comparison}",
            "\\end{table}",
            ""
        ])

        return latex

    def _generate_detailed_results_table(self):
        """Generate a sample of detailed results."""
        latex = [
            "\\subsection{Sample Problem Results}",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{adjustbox}{max width=\\textwidth}",
            "\\begin{tabular}{l|l|c|c|c}",
            "\\toprule",
            "\\textbf{Problem ID} & \\textbf{Model} & \\textbf{Difficulty} & \\textbf{Status} & \\textbf{Time (s)} \\\\",
            "\\midrule"
        ]

        # Sample 5 problems from each difficulty
        sample_size = 5

        for difficulty in self.difficulties:
            # Get problems for each model
            for model in self.models:
                key = f"{model}_{difficulty}"
                if key in self.problem_details:
                    problems = self.problem_details[key][:sample_size]

                    for prob in problems:
                        status_symbol = "✓" if prob["passed"] else "✗"
                        latex.append(
                            f"{prob['problem_id'][:20]} & {model} & {difficulty} & "
                            f"{status_symbol} & {prob['processing_time']:.2f} \\\\"
                        )

            if difficulty != self.difficulties[-1]:
                latex.append("\\midrule")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            "\\caption{Sample of individual problem results showing success/failure and execution time.}",
            "\\label{tab:sample_results}",
            "\\end{table}",
            ""
        ])

        return latex


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze LeetCode true baseline results")
    parser.add_argument("--results-dir", required=True, help="Directory containing baseline results")
    parser.add_argument("--output", required=True, help="Output LaTeX file")

    args = parser.parse_args()

    # Create analyzer
    analyzer = TrueBaselineResultsAnalyzer(
        results_dir=args.results_dir,
        output_file=args.output
    )

    # Load and analyze data
    analyzer.load_data()

    # Generate LaTeX report
    analyzer.generate_latex()

    print("Analysis complete!")


if __name__ == "__main__":
    main()
