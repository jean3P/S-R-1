# scripts/run_experiment.py

import argparse
import logging
from pathlib import Path

from ..config.config import Config
from ..data.data_loader import SWEBenchDataLoader
from ..solution.issue_solver import IssueSolver
from ..evaluation.visualization import Visualizer
from ..utils.logging_utils import setup_logging
from ..utils.file_utils import FileUtils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments on SWE-bench-Verified dataset")

    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, choices=["deepseek-coder", "qwen-coder", "yi-chat", "all"],
                        default="all", help="Model to use")
    parser.add_argument("--reasoning", type=str, choices=["chain_of_thought", "tree_of_thought"],
                        default="chain_of_thought", help="Reasoning type")
    parser.add_argument("--issues", type=str, help="Comma-separated list of issue IDs")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of issues to process")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--log-file", type=str, help="Log file name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def main():
    """Main function for running experiments."""
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
        config["evaluation"]["results_dir"] = args.output

    # Create results directory
    results_dir = Path(config["evaluation"]["results_dir"])
    FileUtils.ensure_directory(results_dir)

    # Load dataset
    data_loader = SWEBenchDataLoader(config)

    # Determine which issues to process
    if args.issues:
        issue_ids = args.issues.split(",")
    else:
        # Load all issues and take the first N
        all_issues = data_loader.load_dataset()
        # Use instance_id instead of id based on SWE-bench dataset structure
        issue_ids = [issue.get("instance_id", issue.get("id", f"issue_{i}")) 
                    for i, issue in enumerate(all_issues[:args.limit])]

    # Determine which model to use
    model_name = None if args.model == "all" else args.model

    # Create issue solver
    solver = IssueSolver(config, model_name, args.reasoning)

    # Solve issues
    results = solver.solve_multiple_issues(issue_ids)

    # Save results
    output_file = results_dir / "results.json"
    FileUtils.write_json(results, output_file)
    logging.info(f"Results saved to {output_file}")

    # Create visualizations
    visualizer = Visualizer(config)

    model_comparison_path = visualizer.visualize_model_comparison(results)
    logging.info(f"Model comparison visualization saved to {model_comparison_path}")

    iteration_improvement_path = visualizer.visualize_iteration_improvement(results)
    logging.info(f"Iteration improvement visualization saved to {iteration_improvement_path}")

    print(f"\nExperiment completed successfully!")
    print(f"Results saved to {output_file}")
    print(f"Visualizations saved to {results_dir}")


if __name__ == "__main__":
    main()
