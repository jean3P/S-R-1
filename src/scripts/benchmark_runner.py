# src/scripts/benchmark_runner.py
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

# Try to import visualization components, but handle if they're not available
try:
    from ..evaluation.enhanced_visualization import EnhancedVisualizer
    from ..statistics.benchmark_report import BenchmarkReport
    visualization_available = True
except ImportError as e:
    logging.warning(f"Visualization components not available: {str(e)}")
    visualization_available = False

# Define visualization_available at module level to avoid reference before assignment
visualization_available = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments on SWE-bench-Verified dataset")

    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str, choices=["deepseek-32b", "qwen-32b", "qwq-32b", "all"],
                        default="all", help="Model to use")
    parser.add_argument("--reasoning", type=str, choices=["chain_of_thought", "tree_of_thought"],
                        default="chain_of_thought", help="Reasoning type")
    parser.add_argument("--issues", type=str, help="Comma-separated list of issue IDs")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of issues to process")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--log-file", type=str, help="Log file name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--skip-solving", action="store_true",
                        help="Skip solving issues, only generate reports from existing results")

    return parser.parse_args()


# Initialize visualization_available at module level
visualization_available = False

def main():
    """Main function for running experiments."""
    global visualization_available
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

    # Skip solving if requested
    if not args.skip_solving:
        # Load dataset
        data_loader = SWEBenchDataLoader(config)

        # Determine which issues to process
        if args.issues:
            issue_ids = args.issues.split(",")
        else:
            # Load all issues and take the first N
            all_issues = data_loader.load_dataset()
            issue_ids = [issue["id"] for issue in all_issues[:args.limit]]

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
    else:
        # Load existing results
        output_file = results_dir / "results.json"
        if output_file.exists():
            results = FileUtils.read_json(output_file)
            logging.info(f"Loaded existing results from {output_file}")
        else:
            logging.error(f"No existing results found at {output_file}")
            return 1

    # Create enhanced visualizations and reports if visualization components are available
    if visualization_available:
        try:
            # Create enhanced visualizations
            visualizer = EnhancedVisualizer(config)
            viz_paths = visualizer.visualize_all(results)
            logging.info(f"Visualizations saved to {results_dir / 'visualizations'}")

            # Generate benchmark reports
            report_generator = BenchmarkReport(config, results)
            full_report = report_generator.generate_full_report()
            logging.info(f"Benchmark reports saved to {results_dir / 'reports'}")

            # Generate model-specific reports
            models = set()
            for result in results:
                if "solutions" in result:
                    models.update(result["solutions"].keys())

            for model in models:
                report_generator.generate_model_report(model)
                
            # Store report for summary
            report_summary = full_report.get("summary", {})
        except Exception as e:
            logging.error(f"Error generating visualizations or reports: {str(e)}")
            visualization_available = False
            report_summary = {}
    else:
        logging.warning("Skipping visualization and report generation due to missing dependencies")
        report_summary = {}

    print(f"\nExperiment completed successfully!")
    print(f"Results saved to {output_file}")
    
    if visualization_available:
        print(f"Visualizations saved to {results_dir / 'visualizations'}")
        print(f"Reports saved to {results_dir / 'reports'}")

        # Print summary from report
        if report_summary:
            print("\nResults Summary:")
            print(f"  Best model: {report_summary.get('best_model', 'N/A')}")
            best_score = report_summary.get('best_model_score')
            if best_score is not None:
                print(f"  Best model score: {best_score:.4f}")
            else:
                print(f"  Best model score: N/A")
            print(f"  Significant differences: {report_summary.get('num_significant_differences', 0)}")
    else:
        print("\nNote: Visualizations and reports were not generated due to missing dependencies.")
        print("To enable visualizations, install required packages with:")
        print("  pip install seaborn matplotlib pandas numpy")


if __name__ == "__main__":
    main()
