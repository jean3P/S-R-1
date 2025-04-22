# scripts/run_experiment.py

import argparse
import logging
import os
import torch
import gc
import traceback
from pathlib import Path
from ..config.config import Config
from ..data.data_loader import SWEBenchDataLoader
from ..solution.issue_solver import IssueSolver, run_with_memory_efficient_llm_guidance
from ..evaluation.visualization import Visualizer
from ..utils.logging_utils import setup_logging
from ..utils.repository_explorer import RepositoryExplorer
from ..utils.file_utils import FileUtils


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
        gc.collect()

        # Log memory usage
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert to GB
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # Convert to GB
            logging.info(f"GPU {i} memory after cleanup: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
    else:
        gc.collect()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments on SWE-bench-Verified dataset")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model", type=str,
                        choices=[
                            "deepseek-r1-distill",
                            "qwen2-5-coder",
                            "qwq-preview",
                            "all"
                        ],
                        default="all", help="Model to use")
    parser.add_argument("--reasoning", type=str, choices=["chain_of_thought", "tree_of_thought"],
                        default="chain_of_thought", help="Reasoning type")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations for self-reflection (default: 3)")
    parser.add_argument("--issues", type=str, help="Comma-separated list of issue IDs")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of issues to process")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--log-file", type=str, help="Log file name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--disable-quantization", action="store_true",
                        help="Disable model quantization (helpful if bitsandbytes is not installed)")
    parser.add_argument("--disable-flash-attention", action="store_true",
                        help="Disable flash attention (helpful if flash-attn is not installed)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force using CPU even if CUDA is available")
    # Add new argument for progressive hybrid approach
    parser.add_argument("--use-progressive-patch", action="store_true",
                        help="Use the progressive hybrid approach for patch creation")
    # Add argument for patch reflection
    parser.add_argument("--patch-reflection", action="store_true",
                        help="Enable self-reflection on patches for improved quality")
    # Add argument for LLM guidance mode
    parser.add_argument("--use-llm-guidance", action="store_true",
                        help="Use the LLM Code Location Guidance Framework")
    # Add arguments for LLM guidance options
    parser.add_argument("--suspected-files", type=str,
                        help="Comma-separated list of suspected files for LLM guidance")
    parser.add_argument("--guidance-iterations", type=int, default=3,
                        help="Number of iterations for LLM guidance refinement")
    # Add memory optimization arguments
    parser.add_argument("--memory-efficient", action="store_true",
                        help="Enable memory-efficient RAG-like approach to reduce CUDA memory usage")
    parser.add_argument("--offload-layers", action="store_true",
                        help="Offload some model layers to CPU to save GPU memory")
    parser.add_argument("--max-context-length", type=int, default=8192,
                        help="Maximum context length to use (lower values use less memory)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate (lower values use less memory)")
    return parser.parse_args()


def run_with_llm_guidance(config, data_loader, issue_ids, model_name,
                          suspected_files, max_iterations, reasoning_type,
                          reflection_iterations):
    """
    Run issues with LLM Code Location Guidance Framework.

    Args:
        config: Configuration object.
        data_loader: SWEBenchDataLoader instance.
        issue_ids: List of issue IDs to process.
        model_name: Name of the model to use.
        suspected_files: List of suspected files.
        error_output: Optional error output or test info.
        max_iterations: Maximum iterations for progressive refinement.

    Returns:
        List of results.
    """
    # This is a wrapper around the memory-efficient version
    # Keep for backwards compatibility
    return run_with_memory_efficient_llm_guidance(
        config, data_loader, issue_ids, model_name,
        suspected_files, max_iterations, reasoning_type,
        reflection_iterations
    )


def print_llm_guidance_summary(results):
    """
    Print summary of LLM guidance results.

    Args:
        results: List of result dictionaries.
    """
    print("\nLLM Guidance Framework Summary:")
    print("-------------------------------")

    successful_fixes = 0
    total_iterations = 0
    issues_count = len(results)

    for result in results:
        issue_id = result.get("issue_id", "unknown")
        print(f"\nIssue: {issue_id}")

        if "solutions" in result:
            for model_name, solutions in result["solutions"].items():
                print(f"  Model: {model_name}")

                if solutions:
                    # Get the final solution
                    final_solution = solutions[-1]

                    # Check if it was successful
                    success = final_solution.get("patch_validation", {}).get("success", False)
                    if success:
                        successful_fixes += 1

                    # Track iterations
                    iterations = final_solution.get("iteration", 1)
                    total_iterations += iterations

                    # Print summary
                    print(f"    Success: {success}")
                    print(f"    Guidance Iterations: {iterations}")

                    if "reflections" in final_solution:
                        print(f"    Reflection Iterations: {len(final_solution['reflections'])}")

                    # Print patch info if available
                    if "patch" in final_solution:
                        patch_lines = final_solution["patch"].count("\n") + 1
                        print(f"    Patch Size: {patch_lines} lines")
                else:
                    print("    No solutions generated")

    # Print overall summary
    if issues_count > 0:
        print("\nOverall Summary:")
        print(f"  Total Issues: {issues_count}")
        print(f"  Successful Fixes: {successful_fixes} ({successful_fixes / issues_count * 100:.1f}%)")
        print(f"  Average Guidance Iterations: {total_iterations / issues_count:.2f}")


def print_progressive_patch_summary(results):
    """
    Print summary of progressive patch approach results.

    Args:
        results: List of result dictionaries.
    """
    print("\nProgressive Hybrid Approach Summary:")
    print("-----------------------------------")
    issues_with_hints = 0
    for result in results:
        if "hints_available" in result and result["hints_available"]:
            issues_with_hints += 1
        if "solutions" in result:
            for model_name, solutions in result["solutions"].items():
                for solution in solutions:
                    if "evaluation" in solution:
                        print(f"  Issue: {result['issue_id']}, Model: {model_name}")
                        print(f"    Patch Quality Score: {solution['evaluation']['patch_quality']:.4f}")
                        print(f"    Overall Score: {solution['evaluation']['overall_score']:.4f}")
    print(f"\n  {issues_with_hints}/{len(results)} issues had hints available")


def print_patch_reflection_summary(results):
    """
    Print summary of patch reflection results.

    Args:
        results: List of result dictionaries.
    """
    print("\nPatch Reflection Summary:")
    print("------------------------")
    issues_with_hints = 0
    for result in results:
        if "hints_available" in result and result["hints_available"]:
            issues_with_hints += 1
        if "solutions" in result:
            for model_name, solutions in result["solutions"].items():
                for solution in solutions:
                    if "initial_patch_quality" in solution and "evaluation" in solution:
                        initial = solution["initial_patch_quality"]
                        final = solution["evaluation"]["patch_quality"]
                        improvement = final - initial
                        print(f"  Issue: {result['issue_id']}, Model: {model_name}")
                        print(f"    Initial Patch Quality: {initial:.4f}")
                        print(f"    Final Patch Quality: {final:.4f}")
                        print(f"    Improvement: {improvement:.4f} ({improvement / initial * 100:.1f}% better)")
    print(f"\n  {issues_with_hints}/{len(results)} issues had hints available")


def print_memory_efficient_summary(results):
    """
    Print summary of memory-efficient approach results.

    Args:
        results: List of result dictionaries.
    """
    print("\nMemory-Efficient RAG Approach Summary:")
    print("-------------------------------------")
    print(f"Successfully processed {len(results)} issues")

    # Count successful fixes
    successful_fixes = 0
    total_issues = len(results)

    for result in results:
        if "solutions" in result:
            for model_name, solutions in result["solutions"].items():
                for solution in solutions:
                    if solution.get("patch_validation", {}).get("success", False):
                        successful_fixes += 1
                        break  # Count one successful fix per model per issue

    if total_issues > 0:
        success_rate = successful_fixes / total_issues * 100
        print(f"Success rate: {successful_fixes}/{total_issues} ({success_rate:.1f}%)")

    # Print memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
            max_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)
            print(f"Maximum GPU {i} memory used: {max_allocated:.2f}GB (reserved: {max_reserved:.2f}GB)")

    print("\nPerformance benefits:")
    print("  - Reduced memory usage through targeted code chunk retrieval")
    print("  - Smaller context windows for faster generation")
    print("  - Better focus on relevant code sections")


def main():
    """Main function for running experiments with improved error handling."""
    args = parse_args()

    # Load configuration
    config_path = args.config
    config = Config(config_path)

    # Set up logging
    if args.debug:
        config["logging"]["log_level"] = "DEBUG"
    setup_logging(config, args.log_file)

    # Log the command line args for reproducibility
    logging.info(f"Command line arguments: {args}")

    # Set output directory if provided
    if args.output:
        config["evaluation"]["results_dir"] = args.output

    # Create results directory
    results_dir = Path(config["evaluation"]["results_dir"])
    FileUtils.ensure_directory(results_dir)

    # Apply command line overrides to config
    if args.cpu_only:
        config["models"]["device"] = "cpu"
        logging.info("Forcing CPU mode (CUDA disabled)")

    # Check for HF_TOKEN
    if "HF_TOKEN" not in os.environ and "HUGGINGFACE_TOKEN" not in os.environ:
        logging.warning("HF_TOKEN or HUGGINGFACE_TOKEN environment variable not set.")
        os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Ensure we can still download models
        logging.info("Set TRANSFORMERS_OFFLINE=0 to allow downloads without token")

    # Apply quantization and flash attention flags
    if args.disable_quantization:
        logging.info("Disabling model quantization")
        for model_name in ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]:
            model_config = config.get_model_config(model_name)
            if model_config and "quantization" in model_config:
                del model_config["quantization"]

    if args.disable_flash_attention:
        logging.info("Disabling flash attention")
        for model_name in ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]:
            model_config = config.get_model_config(model_name)
            if model_config and model_config.get("use_flash_attention", False):
                model_config["use_flash_attention"] = False

    # Apply memory optimization settings
    if args.memory_efficient:
        logging.info("Enabling memory-efficient RAG-like approach")
        config["memory_efficient"] = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        logging.info("Set PYTORCH_CUDA_ALLOC_CONF for better memory management")

    if args.offload_layers:
        logging.info("Enabling layer offloading to CPU")
        config["offload_layers"] = True

    if args.max_context_length:
        logging.info(f"Setting maximum context length to {args.max_context_length}")
        config["data"]["max_context_length"] = args.max_context_length

    if args.max_new_tokens:
        logging.info(f"Setting maximum new tokens to {args.max_new_tokens}")
        config["models"]["max_new_tokens"] = args.max_new_tokens

    # Clear CUDA cache before starting
    clear_cuda_cache()

    # Log CUDA device info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            free = total_mem - allocated
            logging.info(
                f"GPU {i}: {props.name}, Total memory: {total_mem:.2f}GB, Allocated: {allocated:.2f}GB, Free: {free:.2f}GB")
    else:
        logging.info("No CUDA devices available, using CPU")

    try:
        # Load dataset
        data_loader = SWEBenchDataLoader(config)
        data_loader.cleanup_environments(max_age_days=7)
        all_issues = data_loader.load_dataset()
        logging.info(f"Loaded {len(all_issues)} issues from dataset")

        # Determine which issues to process
        if args.issues:
            issue_ids = args.issues.split(",")
            logging.info(f"Processing specified issues: {issue_ids}")
        else:
            # Take first N issues
            issue_limit = min(args.limit, len(all_issues))
            issue_ids = [
                issue.get("instance_id", issue.get("id", f"issue_{i}"))
                for i, issue in enumerate(all_issues[:issue_limit])
            ]
            logging.info(f"Processing first {issue_limit} issues: {issue_ids}")

        # Ensure repositories exist for the selected issues
        repo_explorer = RepositoryExplorer(config)

        valid_issue_ids = []
        for issue_id in issue_ids:
            try:
                issue = data_loader.load_issue(issue_id)
                logging.info(f"Issue: {issue}")
                if not issue:
                    logging.warning(f"Skipping issue {issue_id}: could not load issue data.")
                    continue

                # Check issue description
                description = data_loader.get_issue_description(issue)
                if not description or len(description.strip()) < 10:
                    logging.warning(f"Issue {issue_id} has very short or empty description: '{description}'")
                    # Use fallback description
                    issue[
                        "problem_statement"] = f"Fix issue {issue_id}. Check the repository structure and make minimal changes to fix any bugs."

                # Ensure repo exists
                if not repo_explorer.ensure_repository_exists(issue):
                    logging.warning(f"Skipping issue {issue_id}: repository could not be downloaded.")
                    continue
                valid_issue_ids.append(issue_id)
            except Exception as e:
                logging.error(f"Error processing issue {issue_id}: {e}")
                logging.debug(traceback.format_exc())

        if not valid_issue_ids:
            logging.error("No valid issues to process after checking repositories.")
            return 1  # Error exit code

        issue_ids = valid_issue_ids

        # Determine which model to use
        model_name = None if args.model == "all" else args.model
        if model_name is None:
            logging.info("Running experiment with all models: deepseek-r1-distill, qwen2-5-coder, qwq-preview")
            model_name = ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]

        # Process suspected files if provided
        suspected_files = None
        if args.suspected_files:
            suspected_files = [f.strip() for f in args.suspected_files.split(",")]

        # Run experiment
        if args.use_llm_guidance:
            logging.info("Using LLM Code Location Guidance Framework")
            if args.memory_efficient:
                results = run_with_memory_efficient_llm_guidance(
                    config, data_loader, issue_ids, model_name,
                    suspected_files, args.guidance_iterations, args.reasoning, args.iterations
                )
            else:
                results = run_with_llm_guidance(
                    config, data_loader, issue_ids, model_name,
                    suspected_files, args.guidance_iterations, args.reasoning, args.iterations
                )
        else:
            solver = IssueSolver(config, model_name, args.reasoning, num_iterations=args.iterations)
            if args.use_progressive_patch:
                from ..solution.improved_patch_creator import ImprovedPatchCreator
                solver.patch_creator = ImprovedPatchCreator(config)
                logging.info("Using progressive hybrid approach for patch creation")

            if args.patch_reflection:
                logging.info("Using self-reflection on patches")
                results = solver.solve_issues_with_patch_reflection(issue_ids)
            elif args.memory_efficient:
                logging.info("Using memory-efficient RAG-like approach")
                results = solver.solve_multiple_issues(issue_ids)
            else:
                results = solver.solve_multiple_issues(issue_ids)

        # Save results
        output_file = results_dir / "results.json"
        FileUtils.write_json(results, output_file)
        logging.info(f"Results saved to {output_file}")

        # Log a summary of the results
        logging.info(f"Generated results for {len(results)} issues")
        for result in results:
            issue_id = result.get("issue_id", "unknown")
            solutions = result.get("solutions", {})
            solution_count = sum(len(model_solutions) for model_solutions in solutions.values())
            logging.info(f"Issue {issue_id}: {solution_count} total solutions generated")

        # Visualizations
        try:
            visualizer = Visualizer(config)
            path1 = visualizer.visualize_model_comparison(results)
            logging.info(f"Model comparison saved to {path1}")
            path2 = visualizer.visualize_iteration_improvement(results)
            logging.info(f"Iteration improvement saved to {path2}")
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
            logging.debug(traceback.format_exc())

        print(f"\nExperiment completed successfully!")
        print(f"Results saved to {output_file}")
        print(f"Visualizations saved to {results_dir}")

        if args.use_llm_guidance:
            print_llm_guidance_summary(results)
        elif args.use_progressive_patch:
            print_progressive_patch_summary(results)
        elif args.patch_reflection:
            print_patch_reflection_summary(results)
        elif args.memory_efficient:
            print_memory_efficient_summary(results)

        return 0  # Success exit code

    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")
        logging.debug(traceback.format_exc())

        # Try to save partial results if available
        try:
            if 'results' in locals():
                partial_file = results_dir / "partial_results.json"
                FileUtils.write_json(results, partial_file)
                logging.info(f"Partial results saved to {partial_file}")
                print(f"Partial results saved to {partial_file}")
        except Exception as save_err:
            logging.error(f"Error saving partial results: {save_err}")

        return 1  # Error exit code


if __name__ == "__main__":
    main()
