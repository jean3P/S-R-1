import os
import argparse
import logging
import time
import gc
import torch
from pathlib import Path
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(script_dir)))

from src.config.config import Config
from src.solution.integrated_pipeline import IntegratedBugFixingPipeline
from src.utils.file_utils import FileUtils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Astropy bug fixing pipeline with integrated validation")

    parser.add_argument("--config", type=str, default="configs/experiments/integrated_pipeline.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model", type=str,
                        choices=["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"],
                        default="deepseek-r1-distill", help="Model to use")
    parser.add_argument("--issues", type=str, help="Comma-separated list of issue IDs")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of issues to process")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum number of complete pipeline iterations per issue")
    parser.add_argument("--output", type=str, default="results/integrated_pipeline",
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
    """Run the integrated bug fixing pipeline for Astropy."""
    # Load configuration
    config_path = args.config
    config = Config(config_path)

    # Set up logging
    log_file = setup_logging(args.log_level, args.output)

    # Update config from args
    config = update_config_from_args(config, args)

    # Add pipeline-specific parameters to config
    if "reasoning" not in config.defaults:
        config["reasoning"] = {}
    config["reasoning"]["tot_breadth"] = 3  # Number of branches in ToT phase
    config["reasoning"]["tot_depth"] = 3  # Depth of analysis in ToT phase
    config["reasoning"]["reflection_iterations"] = 2  # Number of iterations for reflection

    # Use max_iterations from command line arguments
    config["reasoning"]["max_iterations"] = args.max_iterations
    logging.info(f"Running with maximum {args.max_iterations} iterations per issue")

    # Add bug detector configuration if not present
    if "bug_detector" not in config.defaults:
        config["bug_detector"] = {
            "output_dir": Path(config["evaluation"]["results_dir"]) / "enhanced_bug_detector",
            "bug_locations_file": "bug_locations.json",
            "max_test_runs": 3,
            "test_timeout": 300
        }

    # Display GPU information
    display_gpu_info()

    # Create results directory
    results_dir = Path(config["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data loader
    from src.data.astropy_synthetic_dataloader import AstropySyntheticDataLoader
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

    # Verify that issues exist
    valid_issue_ids = []
    for issue_id in issue_ids:
        issue = data_loader.load_issue(issue_id)
        if not issue:
            logging.warning(f"Skipping issue {issue_id}: could not load issue data.")
            continue

        valid_issue_ids.append(issue_id)

    if not valid_issue_ids:
        logging.error("No valid issues to process.")
        return

    # Initialize the pipeline
    pipeline = IntegratedBugFixingPipeline(config, args.model)

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

            # Setup for iterative approach
            max_iterations = args.max_iterations
            best_solution = None
            current_iteration = 0
            solution_found = False
            all_iterations_results = []

            # Run the pipeline iteratively
            while current_iteration < max_iterations and not solution_found:
                logging.info(f"Starting iteration {current_iteration + 1}/{max_iterations} for issue {issue_id}")

                # Run the pipeline with previous best solution as input
                current_result = pipeline.solve_issue(issue_id, best_solution)

                # Store iteration number
                current_result["iteration"] = current_iteration + 1
                all_iterations_results.append(current_result)

                # Check if solution was successful
                if current_result.get("success", False):
                    logging.info(f"Found successful solution in iteration {current_iteration + 1}")
                    solution_found = True

                # Update best solution based on depth score
                current_depth = current_result.get("depth_scores", {}).get("combined", 0.0)
                if best_solution is None or current_depth > best_solution.get("depth", 0.0):
                    best_solution = {
                        "solution": current_result.get("solution"),
                        "depth": current_depth,
                        "feedback": current_result.get("feedback", []),
                        "iteration": current_iteration + 1
                    }
                    logging.info(f"Updated best solution with depth {current_depth}")

                # Save interim result for this iteration
                interim_result_file = results_dir / f"interim_{issue_id}_iter{current_iteration + 1}.json"
                FileUtils.write_json(current_result, interim_result_file)
                logging.info(f"Saved interim result for iteration {current_iteration + 1} to {interim_result_file}")

                # Increment counter
                current_iteration += 1

            # Prepare final result
            final_result = current_result.copy() if current_result else {"issue_id": issue_id}
            final_result["total_iterations"] = current_iteration
            final_result["early_stopped"] = solution_found and current_iteration < max_iterations
            final_result["best_solution"] = best_solution
            final_result["all_iterations"] = all_iterations_results

            # Add timing information
            elapsed_time = time.time() - start_time
            final_result["processing_time"] = elapsed_time
            logging.info(
                f"Processed issue {issue_id} in {elapsed_time:.2f} seconds after {current_iteration} iterations")

            # Add result
            results.append(final_result)

            # Save final result for this issue
            final_result_file = results_dir / f"result_{issue_id}.json"
            FileUtils.write_json(final_result, final_result_file)
            logging.info(f"Saved final result to {final_result_file}")

        except Exception as e:
            logging.error(f"Error processing issue {issue_id}: {str(e)}", exc_info=True)
            results.append({
                "issue_id": issue_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            })

    # Save combined results
    combined_file = results_dir / "results.json"
    FileUtils.write_json(results, combined_file)
    logging.info(f"Saved combined results to {combined_file}")

    # Generate summary
    generate_summary(results, results_dir)

    return results


def generate_summary(results, results_dir):
    """Generate a summary of the results."""
    summary = {
        "total_issues": len(results),
        "successful_issues": 0,
        "errors": 0,
        "average_depth": 0.0,
        "average_time": 0.0,
        "test_successes": 0,
        "phase_statistics": {
            "bug_detection": {
                "count": 0,
                "average_depth": 0.0,
                "valid_solutions": 0
            },
            "cot_solution": {
                "count": 0,
                "average_depth": 0.0,
                "valid_solutions": 0
            }
        }
    }

    # Calculate statistics
    total_depth = 0.0
    total_time = 0.0
    success_count = 0
    error_count = 0
    test_success_count = 0

    for result in results:
        # Check for errors
        if "error" in result:
            error_count += 1
            continue

        # Check for success based on test validation
        final_solution = result.get("solution", {})
        validation = final_solution.get("validation", {})

        if validation.get("success", False):
            success_count += 1

            # Check if test validation was successful
            if validation.get("test_run", False) and validation.get("success", False):
                test_success_count += 1

        # Add depth score
        depth = result.get("depth_scores", {}).get("combined", 0.0)
        if depth > 0:
            total_depth += depth

        # Add processing time
        proc_time = result.get("processing_time", 0.0)
        if proc_time > 0:
            total_time += proc_time

        # Process phase statistics
        phases = result.get("phases", [])
        for phase in phases:
            phase_name = phase.get("name")
            if phase_name in summary["phase_statistics"]:
                stats = summary["phase_statistics"][phase_name]
                stats["count"] += 1
                stats["average_depth"] += phase.get("depth", 0.0)

                if phase.get("early_stopped", False):
                    stats["valid_solutions"] += 1

    # Update summary with calculated values
    summary["successful_issues"] = success_count
    summary["errors"] = error_count
    summary["test_successes"] = test_success_count

    # Calculate averages
    if summary["total_issues"] - error_count > 0:
        summary["average_depth"] = total_depth / (summary["total_issues"] - error_count)
        summary["average_time"] = total_time / summary["total_issues"]
        summary["success_rate"] = success_count / (summary["total_issues"] - error_count)
        summary["test_success_rate"] = test_success_count / (summary["total_issues"] - error_count)

    # Calculate phase averages
    for phase_name, stats in summary["phase_statistics"].items():
        if stats["count"] > 0:
            stats["average_depth"] = stats["average_depth"] / stats["count"]

    # Save summary
    summary_file = results_dir / "summary.json"
    FileUtils.write_json(summary, summary_file)
    logging.info(f"Saved summary to {summary_file}")

    # Print summary to console
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total issues processed: {summary['total_issues']}")
    print(
        f"Successfully fixed (validated): {summary['successful_issues']} ({summary['successful_issues'] / (summary['total_issues'] - error_count) * 100:.1f}%)")
    print(
        f"Test validation successes: {summary['test_successes']} ({summary['test_successes'] / (summary['total_issues'] - error_count) * 100:.1f}%)")
    print(f"Errors: {summary['errors']}")
    print(f"Average combined depth: {summary['average_depth']:.3f}")
    print(f"Average processing time: {summary['average_time']:.1f} seconds")

    print("\nPhase Statistics:")
    for phase_name, stats in summary["phase_statistics"].items():
        if stats["count"] > 0:
            print(f"  {phase_name}:")
            print(f"    Completed in {stats['count']} issues")
            print(f"    Average depth: {stats['average_depth']:.3f}")
            print(f"    Valid solutions: {stats['valid_solutions']}")

    # Generate visualizations if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Ensure directory exists
        viz_dir = results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Create depth score chart
        plt.figure(figsize=(10, 6))

        # Extract depth scores for each issue
        issue_ids = [result.get("issue_id", f"Issue {i}") for i, result in enumerate(results) if "error" not in result]
        depths = []

        for result in results:
            if "error" in result:
                continue

            depth_scores = result.get("depth_scores", {})
            depths.append([
                depth_scores.get("location_specificity", 0),
                depth_scores.get("solution_completeness", 0),
                depth_scores.get("combined", 0)
            ])

        # Transpose for plotting
        if depths:
            depths = list(zip(*depths))

            # Plot stacked bars
            x = range(len(issue_ids))
            width = 0.6

            if len(x) > 0:
                plt.bar(x, depths[0], width, label='Location Specificity')
                plt.bar(x, depths[1], width, bottom=depths[0], label='Solution Completeness')

                # Plot combined score as a line
                plt.plot(x, depths[2], 'ro-', linewidth=2, label='Combined Score')

                plt.xlabel('Issues')
                plt.ylabel('Depth Score')
                plt.title('Depth Scores by Issue')
                plt.xticks(x, issue_ids, rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()

                # Save the figure
                plt.savefig(viz_dir / "depth_scores.png")
                plt.close()

        # Create success rate by phase chart
        plt.figure(figsize=(8, 6))

        phases = ["bug_detection", "cot_solution"]
        phase_counts = [summary["phase_statistics"][p]["count"] for p in phases]
        valid_solutions = [summary["phase_statistics"][p]["valid_solutions"] for p in phases]

        x = range(len(phases))
        width = 0.35

        plt.bar(x, phase_counts, width, label='Phase Reached')
        plt.bar([i + width for i in x], valid_solutions, width, label='Valid Solutions')

        plt.xlabel('Pipeline Phase')
        plt.ylabel('Count')
        plt.title('Success Rate by Pipeline Phase')
        plt.xticks([i + width / 2 for i in x], [p.replace('_', ' ').title() for p in phases])
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(viz_dir / "phase_success.png")
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
