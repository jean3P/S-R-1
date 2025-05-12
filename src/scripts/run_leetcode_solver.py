"""
Run the LeetCode Solution Generator with self-reflection and testing.
"""

import argparse
import logging
import os
import sys
import time
import json
import gc
from pathlib import Path

# Add parent directory to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from src.config.config import Config
from src.data.leetcode_dataloader import LeetCodeDataLoader
from src.solution.leetcode_solution_pipeline import LeetCodeSolutionPipeline
from src.utils.file_utils import FileUtils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the LeetCode Solution Generator with self-reflection and testing"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/leetcode_solver.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"],
        default="deepseek-r1-distill",
        help="Model to use"
    )
    parser.add_argument(
        "--problems",
        type=str,
        help="Comma-separated list of problem IDs (e.g., 'two-sum,add-two-numbers')"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["Easy", "Medium", "Hard"],
        help="Filter problems by difficulty"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of problems to process"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=3,
        help="Number of solution candidates to generate per round"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of self-reflection rounds"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/leetcode_solver",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Use memory-efficient processing"
    )
    parser.add_argument(
        "--disable-quantization",
        action="store_true",
        help="Disable model quantization"
    )
    parser.add_argument(
        "--disable-flash-attention",
        action="store_true",
        help="Disable flash attention"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force using CPU even if CUDA is available"
    )
    parser.add_argument(
        "--indices",
        type=str,
        help="Comma-separated list of problem indices to process (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--use-code-eval",
        action="store_true",
        help="Enable code_eval metrics from HuggingFace"
    )

    parser.add_argument(
        "--initial-k",
        type=int,
        default=3,
        help="Number of initial solution candidates to generate"
    )
    parser.add_argument(
        "--branch-factor",
        type=int,
        default=3,
        help="Number of solutions to generate per failed solution (k parameter)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth of the solution tree"
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=False,
        help="Stop branching once a correct solution is found"
    )

    return parser.parse_args()


def setup_logging(log_level, output_dir):
    """Set up logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"leetcode_solver_{time.strftime('%Y%m%d_%H%M%S')}.log"

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

    # Update LeetCode-specific settings
    if "leetcode" not in config.defaults:
        config["leetcode"] = {}

    config["leetcode"]["num_candidates"] = args.candidates
    config["leetcode"]["reflection_rounds"] = args.rounds
    config["leetcode"]["early_stopping"] = args.early_stopping
    config["leetcode"]["initial_solutions"] = args.initial_k
    config["leetcode"]["branch_factor"] = args.branch_factor
    config["leetcode"]["max_depth"] = args.max_depth
    config["leetcode"]["early_stopping"] = args.early_stopping

    # Update code_eval settings
    if args.use_code_eval:
        if "evaluation" not in config.defaults:
            config["evaluation"] = {}
        config["evaluation"]["use_code_eval"] = True
        logging.info("Enabled code_eval metrics from HuggingFace")

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


def run_leetcode_solver(args):
    """Run the LeetCode solution generator."""
    # Load configuration
    config_path = args.config
    config = Config(config_path)

    # Set up logging
    log_file = setup_logging(args.log_level, args.output)

    # Update config from args
    config = update_config_from_args(config, args)

    # Display GPU information
    display_gpu_info()

    # Create results directory
    results_dir = Path(config["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Install datasets package if not already installed
    try:
        import datasets
        logging.info(f"Using datasets library version {datasets.__version__}")
    except ImportError:
        logging.warning("Huggingface datasets library not installed. Attempting to install...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            logging.info("Successfully installed datasets library")
        except Exception as e:
            logging.error(f"Failed to install datasets library: {str(e)}")
            return

    # Install evaluate package if code_eval is enabled
    if config["evaluation"].get("use_code_eval", False):
        try:
            import evaluate
            logging.info(f"Using evaluate library version {evaluate.__version__}")
        except ImportError:
            logging.warning("Huggingface evaluate library not installed. Attempting to install...")
            import subprocess
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate"])
                logging.info("Successfully installed evaluate library")
            except Exception as e:
                logging.error(f"Failed to install evaluate library: {str(e)}")
                return

    # Initialize data loader
    try:
        data_loader = LeetCodeDataLoader(config)
        if data_loader.dataset is None:
            logging.error("Failed to load dataset. Exiting.")
            return
    except Exception as e:
        logging.error(f"Error initializing data loader: {str(e)}", exc_info=True)
        return

    # Determine problems to process
    problem_indices = []
    if args.indices:
        try:
            problem_indices = [int(idx) for idx in args.indices.split(',')]
            logging.info(f"Using specified problem indices: {problem_indices}")
        except ValueError:
            logging.error("Invalid problem indices format. Use comma-separated integers.")
            return

    # Load specific problem IDs if provided
    problem_ids = []
    if args.problems:
        problem_ids = args.problems.split(",")
        logging.info(f"Using specified problem IDs: {problem_ids}")
    elif problem_indices:
        # Process problems by indices
        problems = []
        for idx in problem_indices:
            problem = data_loader.load_problem(idx)
            if problem:
                problems.append(problem)
    else:
        # List problems based on filters
        problems = data_loader.list_problems(difficulty=args.difficulty, limit=args.limit)
        problem_ids = [p["problem_id"] for p in problems]

    if not problem_ids and not problem_indices:
        logging.error("No problems to process")
        return

    logging.info(f"Processing {len(problem_ids)} problems by ID: {problem_ids}")
    logging.info(f"Processing {len(problem_indices)} problems by index: {problem_indices}")

    # Initialize the pipeline
    pipeline = LeetCodeSolutionPipeline(config, args.model)

    # Process each problem
    results = []

    # Process by problem ID
    for problem_id in problem_ids:
        logging.info(f"Processing problem ID {problem_id} with model {args.model}")

        try:
            # Clear memory before starting new problem
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Track start time
            start_time = time.time()

            # Load problem data
            problem_data = data_loader.load_problem_by_id(problem_id)

            if not problem_data:
                logging.error(f"Problem {problem_id} not found")
                continue

            # Run the solution pipeline
            result = pipeline.solve_problem(problem_data)

            # Add timing information
            elapsed_time = time.time() - start_time
            result["processing_time"] = elapsed_time

            logging.info(f"Processed problem {problem_id} in {elapsed_time:.2f} seconds")

            # Add result
            results.append(result)

        except Exception as e:
            logging.error(f"Error processing problem {problem_id}: {str(e)}", exc_info=True)
            results.append({
                "problem_id": problem_id,
                "status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time
            })

    # Process by problem index
    for idx in problem_indices:
        logging.info(f"Processing problem index {idx} with model {args.model}")

        try:
            # Clear memory before starting new problem
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Track start time
            start_time = time.time()

            # Load problem data
            problem_data = data_loader.load_problem(idx)

            if not problem_data:
                logging.error(f"Problem index {idx} not found")
                continue

            # Run the solution pipeline
            result = pipeline.solve_problem(problem_data)

            # Add timing information
            elapsed_time = time.time() - start_time
            result["processing_time"] = elapsed_time

            logging.info(f"Processed problem index {idx} in {elapsed_time:.2f} seconds")

            # Add result
            results.append(result)

        except Exception as e:
            logging.error(f"Error processing problem index {idx}: {str(e)}", exc_info=True)
            problem_id = f"index_{idx}"
            results.append({
                "problem_id": problem_id,
                "status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time
            })

    # Save combined results
    combined_file = results_dir / "combined_results.json"
    FileUtils.write_json(results, combined_file)
    logging.info(f"Saved combined results to {combined_file}")

    # Generate summary
    generate_summary(results, results_dir)

    return results


def generate_summary(results, results_dir):
    """Generate a summary of the results."""
    summary = {
        "total_problems": len(results),
        "solved_problems": sum(1 for r in results if r.get("status") == "solved"),
        "average_time": sum(r.get("processing_time", 0) for r in results) / len(results) if results else 0,
        "total_candidates": sum(r.get("total_candidates", 0) for r in results),
        "by_difficulty": {
            "Easy": {
                "total": 0,
                "solved": 0
            },
            "Medium": {
                "total": 0,
                "solved": 0
            },
            "Hard": {
                "total": 0,
                "solved": 0
            }
        }
    }

    # Process results by difficulty
    for result in results:
        difficulty = result.get("difficulty", "Unknown")

        if difficulty in summary["by_difficulty"]:
            summary["by_difficulty"][difficulty]["total"] += 1
            if result.get("status") == "solved":
                summary["by_difficulty"][difficulty]["solved"] += 1

    # Calculate solve rates
    for diff in summary["by_difficulty"]:
        total = summary["by_difficulty"][diff]["total"]
        solved = summary["by_difficulty"][diff]["solved"]

        if total > 0:
            summary["by_difficulty"][diff]["solve_rate"] = solved / total
        else:
            summary["by_difficulty"][diff]["solve_rate"] = 0

    # Overall solve rate
    if summary["total_problems"] > 0:
        summary["solve_rate"] = summary["solved_problems"] / summary["total_problems"]
    else:
        summary["solve_rate"] = 0

    # Extract code_eval metrics if available
    code_eval_metrics = {}
    for result in results:
        if "code_eval_results" in result and "pass_at_k" in result["code_eval_results"]:
            for k, value in result["code_eval_results"]["pass_at_k"].items():
                if k not in code_eval_metrics:
                    code_eval_metrics[k] = []
                code_eval_metrics[k].append(value)

    # Calculate average pass@k metrics
    if code_eval_metrics:
        summary["code_eval"] = {
            k: sum(values) / len(values) if values else 0
            for k, values in code_eval_metrics.items()
        }

    # Save summary
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Saved summary to {summary_file}")

    # Print summary to console
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total problems processed: {summary['total_problems']}")
    print(f"Successfully solved: {summary['solved_problems']} ({summary['solve_rate'] * 100:.1f}%)")
    print(f"Average processing time: {summary['average_time']:.2f} seconds")
    print(f"Total candidate solutions: {summary['total_candidates']}")

    print("\nBy Difficulty:")
    for diff, stats in summary["by_difficulty"].items():
        if stats["total"] > 0:
            print(f"  {diff}: {stats['solved']}/{stats['total']} ({stats['solve_rate'] * 100:.1f}%)")

    # Print code_eval metrics if available
    if "code_eval" in summary:
        print("\nCode Evaluation Metrics:")
        for k, value in summary["code_eval"].items():
            print(f"  {k}: {value * 100:.1f}%")

    return summary


if __name__ == "__main__":
    args = parse_args()
    run_leetcode_solver(args)
