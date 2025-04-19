
# solve_issue.py
# !/usr/bin/env python3
import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ..config.config import Config
from ..data.data_loader import SWEBenchDataLoader
from ..solution.issue_solver import IssueSolver
from ..utils.logging_utils import setup_logging
from ..utils.file_utils import FileUtils
from ..utils.git_utils import GitUtils
from ..utils.issue_complexity import IssueComplexityAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Solve a specific GitHub issue")

    parser.add_argument("issue_id", type=str, help="ID of the issue to solve")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, choices=["deepseek-32b", "qwen-32b", "qwq-32b"],
                        default="qwen-32b", help="Model to use")
    parser.add_argument("--reasoning", type=str, choices=["chain_of_thought", "tree_of_thought"],
                        default="chain_of_thought", help="Reasoning type")
    parser.add_argument("--apply", action="store_true", help="Apply the solution to the repository")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--log-file", type=str, help="Log file name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--analyze-complexity", action="store_true", help="Analyze issue complexity")

    return parser.parse_args()


def main():
    """Main function for solving a specific issue."""
    args = parse_args()

    # Load configuration
    config_path = args.config
    config = Config(config_path)

    # Set up logging
    if args.debug:
        config["logging"]["log_level"] = "DEBUG"
    setup_logging(config, args.log_file)

    # Set output directory if provided
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config["evaluation"]["results_dir"]) / args.issue_id

    FileUtils.ensure_directory(output_dir)

    # Create issue solver
    solver = IssueSolver(config, args.model, args.reasoning)

    try:
        # Load the issue to analyze complexity if requested
        if args.analyze_complexity:
            data_loader = SWEBenchDataLoader(config)
            issue = data_loader.load_issue(args.issue_id)

            if not issue:
                logging.error(f"Issue {args.issue_id} not found")
                return 1

            # Analyze complexity
            complexity_analyzer = IssueComplexityAnalyzer()
            complexity = complexity_analyzer.calculate_complexity(issue)

            # Print complexity information
            print(f"\nIssue Complexity Analysis:")
            print(f"  Overall Complexity: {complexity['overall']:.2f}/10")
            print(f"  Complexity Level: {complexity['complexity_level']}")
            print(f"  Patch Complexity: {complexity['patch_complexity']['overall']:.2f}/10")
            print(f"  Description Complexity: {complexity['description_complexity']['overall']:.2f}/10")
            print(f"  Files Complexity: {complexity['files_complexity']['overall']:.2f}/10")

            # Save complexity analysis
            complexity_file = output_dir / "complexity_analysis.json"
            FileUtils.write_json(complexity, complexity_file)
            logging.info(f"Complexity analysis saved to {complexity_file}")

        # Solve the issue
        result = solver.solve_issue(args.issue_id)

        # Extract the best solution
        model_name = args.model
        solutions = result["solutions"][model_name]

        # Find the solution with the highest overall score
        best_solution = max(solutions, key=lambda s: s["evaluation"]["overall_score"])

        # Save the solution
        solution_file = output_dir / "solution.json"
        FileUtils.write_json(best_solution, solution_file)

        # Save the patch
        patch_file = output_dir / "solution.patch"
        FileUtils.write_file(best_solution["patch"], patch_file)

        logging.info(f"Solution saved to {solution_file}")
        logging.info(f"Patch saved to {patch_file}")

        # Apply the solution if requested
        if args.apply:
            # Load the issue to get the repository path
            data_loader = SWEBenchDataLoader(config)
            issue = data_loader.load_issue(args.issue_id)

            if not issue:
                logging.error(f"Issue {args.issue_id} not found")
                return 1

            repo = issue.get("repo", "")
            repo_path = Path(config["data"]["swe_bench_path"]) / "repos" / repo

            if not repo_path.exists():
                logging.error(f"Repository not found at {repo_path}")
                return 1

            # Create a branch for the solution
            branch_name = f"solution-{args.issue_id}"
            if GitUtils.create_branch(repo_path, branch_name):
                logging.info(f"Created branch {branch_name}")

                # Apply the patch
                if GitUtils.apply_patch(repo_path, patch_file):
                    logging.info(f"Applied patch successfully")

                    # Commit the changes
                    if GitUtils.commit_changes(repo_path, f"Solution for issue {args.issue_id}"):
                        logging.info(f"Committed changes")
                    else:
                        logging.error(f"Failed to commit changes")
                else:
                    logging.error(f"Failed to apply patch")
            else:
                logging.error(f"Failed to create branch")

        print(f"\nIssue solved successfully!")
        print(f"Evaluation:")
        for metric, value in best_solution["evaluation"].items():
            print(f"  {metric}: {value:.2f}")
        print(f"\nSolution saved to {output_dir}")

        # Print reflection statistics
        if len(solutions) > 1:
            initial_score = solutions[0]["evaluation"]["overall_score"]
            final_score = best_solution["evaluation"]["overall_score"]
            improvement = final_score - initial_score
            improvement_pct = (improvement / initial_score * 100) if initial_score > 0 else 0

            print(f"\nSelf-Reflection Improvement:")
            print(f"  Initial score: {initial_score:.2f}")
            print(f"  Final score: {final_score:.2f}")
            print(f"  Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")

    except Exception as e:
        logging.error(f"Error solving issue: {str(e)}")
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
