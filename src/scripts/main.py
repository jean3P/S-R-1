# src/main.py

import os
import argparse
import logging
import time
import gc
import torch
import json
from pathlib import Path
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(script_dir))

from src.config.config import Config
from src.solution.robust_bug_fixing_pipeline import RobustBugFixingPipeline
from src.utils.file_utils import FileUtils
from src.data.astropy_synthetic_dataloader import AstropySyntheticDataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the robust bug fixing pipeline")

    parser.add_argument("--config", type=str, default="configs/experiments/robust_pipeline.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model", type=str,
                        choices=["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"],
                        default="deepseek-r1-distill", help="Model to use")
    parser.add_argument("--issues", type=str, help="Comma-separated list of issue IDs")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of issues to process")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Maximum number of iterations per issue")
    parser.add_argument("--output", type=str, default="results/robust_pipeline",
                        help="Output directory for results")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    parser.add_argument("--memory-efficient", action="store_true",
                        help="Use memory-efficient processing")
    parser.add_argument("--disable-quantization", action="store_true",
                        help="Disable model quantization")
    parser.add_argument("--disable-flash-attention", action="store_true",
                        help="Disable flash attention")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force using CPU even if CUDA is available")

    return parser.parse_args()


def setup_logging(log_level, output_dir):
    """Set up logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging initialized at {log_file}")
    return log_file


def update_config_from_args(config, args):
    """Update configuration based on command line arguments."""
    # Update device setting
    if args.cpu_only:
        config["models"]["device"] = "cpu"
        logging.info("Forcing CPU mode (CUDA disabled)")

    # Update output directory
    if args.output:
        config["evaluation"]["results_dir"] = args.output

    # Update quantization and flash attention settings
    if args.disable_quantization:
        for model_name in ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]:
            model_config = config.get_model_config(model_name)
            if model_config and "quantization" in model_config:
                del model_config["quantization"]

    if args.disable_flash_attention:
        for model_name in ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]:
            model_config = config.get_model_config(model_name)
            if model_config and model_config.get("use_flash_attention", False):
                model_config["use_flash_attention"] = False

    # Set memory efficiency flag
    if args.memory_efficient:
        config["memory_efficient"] = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        logging.info("Set PYTORCH_CUDA_ALLOC_CONF for better memory management")

    # Update max iterations
    config["max_iterations"] = args.max_iterations
    logging.info(f"Set maximum iterations to {args.max_iterations}")

    return config


def display_gpu_info():
    """Display GPU information."""
    if torch.cuda.is_available():
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            logging.info(f"GPU {i}: {device_properties.name}")
            logging.info(f"  Memory: {device_properties.total_memory / 1024 ** 3:.2f} GB")
            logging.info(f"  CUDA Capability: {device_properties.major}.{device_properties.minor}")
    else:
        logging.info("CUDA not available, using CPU")


def run_pipeline(args):
    """Run the robust bug fixing pipeline."""
    # Load configuration
    config_path = args.config
    config = Config(config_path)

    # Add safety check for config access
    if "evaluation" not in config.defaults:
        config.defaults["evaluation"] = {}

    if "results_dir" not in config.defaults["evaluation"]:
        config.defaults["evaluation"]["results_dir"] = args.output or "results/robust_pipeline"

    # Set up logging
    log_file = setup_logging(args.log_level, args.output)

    # Update config from args
    config = update_config_from_args(config, args)

    # Display GPU information
    display_gpu_info()

    # Create results directory - use get() with default for safety
    results_dir = Path(config.defaults["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the data loader - Use AstropySyntheticDataLoader
    data_loader = AstropySyntheticDataLoader(config)

    # Load issue IDs
    if args.issues:
        issue_ids = args.issues.split(",")
    else:
        # Load all issues from dataset
        all_issues = data_loader.load_dataset()
        logging.info(f"Loaded {len(all_issues)} issues from dataset")
        # Take first N issues
        issue_limit = min(args.limit, len(all_issues))
        issue_ids = []
        for i, issue in enumerate(all_issues[:issue_limit]):
            issue_id = issue.get("instance_id") or issue.get("branch_name")
            if issue_id:
                issue_ids.append(issue_id)

    logging.info(f"Processing {len(issue_ids)} issues: {issue_ids}")

    # Verify that issues exist with valid data
    valid_issue_ids = []
    for issue_id in issue_ids:
        issue = data_loader.load_issue(issue_id)
        if not issue:
            logging.warning(f"Skipping issue {issue_id}: could not load issue data.")
            continue

        # Check for essential fields
        if "FAIL_TO_PASS" not in issue or not issue["FAIL_TO_PASS"]:
            logging.warning(f"Skipping issue {issue_id}: missing FAIL_TO_PASS field.")
            continue

        # Verify branch_name exists - needed for checkout
        if "branch_name" not in issue or not issue["branch_name"]:
            logging.warning(f"Skipping issue {issue_id}: missing branch_name field.")
            continue

        # Verify repo path exists
        repo_path = issue.get("Path_repo", "")
        if not repo_path or not os.path.exists(repo_path):
            logging.warning(f"Skipping issue {issue_id}: repository path {repo_path} does not exist.")
            continue

        valid_issue_ids.append(issue_id)

    if not valid_issue_ids:
        logging.error("No valid issues to process after validation.")
        return

    # Initialize the pipeline
    pipeline = RobustBugFixingPipeline(config, args.model)

    # Process each issue
    results = []

    for issue_id in valid_issue_ids:
        logging.info(f"Processing issue {issue_id} with model {args.model}")

        try:
            # Clear memory before starting new issue
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Track start time
            start_time = time.time()

            # Run the pipeline
            issue_result = pipeline.solve_bug(issue_id)

            # Add timing information
            elapsed_time = time.time() - start_time
            issue_result["processing_time"] = elapsed_time
            logging.info(f"Processed issue {issue_id} in {elapsed_time:.2f} seconds")

            # Add to results
            results.append(issue_result)

        except Exception as e:
            logging.error(f"Error processing issue {issue_id}: {str(e)}", exc_info=True)
            results.append({
                "bug_id": issue_id,
                "status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time
            })

    # Save combined results
    combined_results_file = results_dir / "combined_results.json"
    FileUtils.write_json(results, combined_results_file)
    logging.info(f"Saved combined results to {combined_results_file}")

    # Generate summary
    generate_summary(results, results_dir)

    return results


def generate_summary(results, results_dir):
    """Generate a summary of the results."""
    summary = {
        "total_issues": len(results),
        "successful_issues": 0,
        "error_issues": 0,
        "no_solution_issues": 0,
        "average_iterations": 0,
        "average_time": 0,
        "iteration_statistics": {
            "CoT": {
                "count": 0,
                "success_count": 0
            },
            "Self-Reflection": {
                "count": 0,
                "success_count": 0
            }
        }
    }

    # Calculate statistics
    total_iterations = 0
    total_time = 0

    for result in results:
        status = result.get("status", "unknown")

        if status == "passed":
            summary["successful_issues"] += 1
        elif status == "error":
            summary["error_issues"] += 1
        elif status == "no_solution":
            summary["no_solution_issues"] += 1

        # Add iterations
        iterations = result.get("iterations", 0)
        total_iterations += iterations

        # Add processing time
        proc_time = result.get("processing_time", 0)
        if proc_time > 0:
            total_time += proc_time

        # Analyze iteration data
        history = result.get("history", [])
        for entry in history:
            phase = entry.get("phase", "unknown")
            if phase in summary["iteration_statistics"]:
                summary["iteration_statistics"][phase]["count"] += 1

                # Check if this was the successful iteration
                if entry.get("test_result", {}).get("status") == "pass":
                    summary["iteration_statistics"][phase]["success_count"] += 1

    # Calculate averages
    if summary["total_issues"] > 0:
        summary["average_iterations"] = total_iterations / summary["total_issues"]
        summary["average_time"] = total_time / summary["total_issues"]
        summary["success_rate"] = summary["successful_issues"] / summary["total_issues"]

    # Save summary
    summary_file = results_dir / "summary.json"
    FileUtils.write_json(summary, summary_file)
    logging.info(f"Saved summary to {summary_file}")

    # Print summary to console
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total issues processed: {summary['total_issues']}")
    print(f"Successfully fixed: {summary['successful_issues']} " +
          f"({summary['successful_issues'] / summary['total_issues'] * 100:.1f}%)")
    print(f"No solution found: {summary['no_solution_issues']}")
    print(f"Errors: {summary['error_issues']}")
    print(f"Average iterations: {summary['average_iterations']:.2f}")
    print(f"Average processing time: {summary['average_time']:.1f} seconds")

    print("\nIteration Statistics:")
    for phase_name, stats in summary["iteration_statistics"].items():
        if stats["count"] > 0:
            success_rate = stats["success_count"] / stats["count"] * 100
            print(f"  {phase_name}:")
            print(f"    Total iterations: {stats['count']}")
            print(f"    Success rate: {stats['success_count']} ({success_rate:.1f}%)")

    # Generate visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create visualization directory
        viz_dir = results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Create success rate chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Data
        categories = ['Overall', 'CoT', 'Self-Reflection']
        success_rates = [
            summary["successful_issues"] / summary["total_issues"] * 100,
            summary["iteration_statistics"]["CoT"]["success_count"] / max(1, summary["iteration_statistics"]["CoT"][
                "count"]) * 100,
            summary["iteration_statistics"]["Self-Reflection"]["success_count"] / max(1,
                                                                                      summary["iteration_statistics"][
                                                                                          "Self-Reflection"][
                                                                                          "count"]) * 100
        ]

        # Create bar chart
        bars = ax.bar(categories, success_rates, color=['blue', 'green', 'orange'])

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')

        # Customize chart
        ax.set_ylim(0, 100)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rates by Phase')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save figure
        plt.tight_layout()
        plt.savefig(viz_dir / "success_rates.png")
        plt.close()

        # Create iterations distribution chart
        iterations_data = [result.get("iterations", 0) for result in results]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram
        ax.hist(iterations_data, bins=range(1, max(iterations_data) + 2), alpha=0.7, color='blue', edgecolor='black')

        # Customize chart
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Number of Issues')
        ax.set_title('Distribution of Iterations Required')
        ax.set_xticks(range(1, max(iterations_data) + 1))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save figure
        plt.tight_layout()
        plt.savefig(viz_dir / "iterations_distribution.png")
        plt.close()

        logging.info(f"Generated visualizations in {viz_dir}")

    except ImportError:
        logging.warning("Could not generate visualizations: matplotlib not available")
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")

    return summary


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Run the pipeline
    results = run_pipeline(args)

    # Exit with success
    sys.exit(0)
