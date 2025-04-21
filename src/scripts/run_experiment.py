# scripts/run_experiment.py
import argparse
import logging
import re
import os
import torch
import gc
from pathlib import Path

from ..utils.llm_guidance import LLMCodeLocationGuidance
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

    # Apply command line overrides to config
    if args.cpu_only:
        config["models"]["device"] = "cpu"
        logging.info("Forcing CPU mode (CUDA disabled)")

    # Check for HF_TOKEN
    if "HF_TOKEN" not in os.environ and "HUGGINGFACE_TOKEN" not in os.environ:
        logging.warning("HF_TOKEN or HUGGINGFACE_TOKEN environment variable not set.")
        logging.warning("You may need it to access some models. Set it with:")
        logging.warning("export HF_TOKEN=your_huggingface_token")

    # Set output directory if provided
    if args.output:
        config["evaluation"]["results_dir"] = args.output

    # Create results directory
    results_dir = Path(config["evaluation"]["results_dir"])
    FileUtils.ensure_directory(results_dir)

    # Apply quantization and flash attention flags
    if args.disable_quantization:
        logging.info("Disabling model quantization")
        # Disable quantization for all models
        for model_name in ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]:
            model_config = config.get_model_config(model_name)
            if model_config and "quantization" in model_config:
                del model_config["quantization"]

    if args.disable_flash_attention:
        logging.info("Disabling flash attention")
        # Disable flash attention for all models
        for model_name in ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]:
            model_config = config.get_model_config(model_name)
            if model_config and model_config.get("use_flash_attention", False):
                model_config["use_flash_attention"] = False

    # Apply memory optimization settings
    if args.memory_efficient:
        logging.info("Enabling memory-efficient RAG-like approach")
        config["memory_efficient"] = True

        # Enable expandable segments for PyTorch memory allocator
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        logging.info(
            "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 for better memory management")

    if args.offload_layers:
        logging.info("Enabling layer offloading to CPU")
        config["offload_layers"] = True

    if args.max_context_length:
        logging.info(f"Setting maximum context length to {args.max_context_length}")
        config["data"]["max_context_length"] = args.max_context_length

    if args.max_new_tokens:
        logging.info(f"Setting maximum new tokens to {args.max_new_tokens}")
        config["models"]["max_new_tokens"] = args.max_new_tokens

    # Clear CUDA cache and collect garbage before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

        # Log GPU memory information
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            free = total_mem - allocated
            logging.info(f"GPU {i}: Total memory: {total_mem:.2f}GB, "
                         f"Allocated: {allocated:.2f}GB, "
                         f"Free: {free:.2f}GB")

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

    # If all models selected, specify the model names
    if model_name is None:
        logging.info("Running experiment with all models: deepseek-r1-distill, qwen2-5-coder, qwq-preview")
        model_name = ["deepseek-r1-distill", "qwen2-5-coder", "qwq-preview"]

    # Process suspected files if provided
    suspected_files = None
    if args.suspected_files:
        suspected_files = [f.strip() for f in args.suspected_files.split(",")]

    # Run with LLM guidance if requested
    if args.use_llm_guidance:
        logging.info("Using LLM Code Location Guidance Framework")
        if args.memory_efficient:
            # Use memory-efficient version of LLM guidance
            results = run_with_memory_efficient_llm_guidance(
                config,
                data_loader,
                issue_ids,
                model_name,
                suspected_files,
                args.guidance_iterations,
                args.reasoning,
                args.iterations
            )
        else:
            # Use standard LLM guidance
            results = run_with_llm_guidance(
                config,
                data_loader,
                issue_ids,
                model_name,
                suspected_files,
                args.guidance_iterations,
                args.reasoning,
                args.iterations
            )
    else:
        # Create issue solver
        solver = IssueSolver(config, model_name, args.reasoning, num_iterations=args.iterations)

        # Use the progressive hybrid approach for patch creation if requested
        if args.use_progressive_patch:
            from ..solution.improved_patch_creator import ImprovedPatchCreator
            solver.patch_creator = ImprovedPatchCreator(config)
            logging.info("Using progressive hybrid approach for patch creation")

        # Solve issues with or without patch reflection
        if args.patch_reflection:
            logging.info("Using self-reflection on patches")
            results = solver.solve_issues_with_patch_reflection(issue_ids)
        elif args.memory_efficient:
            logging.info("Using memory-efficient RAG-like approach")
            # No need for separate method call - IssueSolver should handle memory optimization internally
            # when the config flag is set
            results = solver.solve_multiple_issues(issue_ids)
        else:
            results = solver.solve_multiple_issues(issue_ids)

    # Save results
    output_file = results_dir / "results.json"
    FileUtils.write_json(results, output_file)
    logging.info(f"Results saved to {output_file}")

    # Create visualizations
    try:
        visualizer = Visualizer(config)
        model_comparison_path = visualizer.visualize_model_comparison(results)
        logging.info(f"Model comparison visualization saved to {model_comparison_path}")
        iteration_improvement_path = visualizer.visualize_iteration_improvement(results)
        logging.info(f"Iteration improvement visualization saved to {iteration_improvement_path}")
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")

    print(f"\nExperiment completed successfully!")
    print(f"Results saved to {output_file}")
    print(f"Visualizations saved to {results_dir}")

    # Print summary of results based on the approach used
    if args.use_llm_guidance:
        print_llm_guidance_summary(results)
    elif args.use_progressive_patch:
        print_progressive_patch_summary(results)
    elif args.patch_reflection:
        print_patch_reflection_summary(results)
    elif args.memory_efficient:
        print_memory_efficient_summary(results)


def run_with_memory_efficient_llm_guidance(config, data_loader, issue_ids, model_name,
                                           suspected_files, max_iterations, reasoning_type,
                                           reflection_iterations):
    """
    Run issues with memory-efficient LLM Code Location Guidance Framework.
    This version incorporates memory optimizations to reduce CUDA memory usage.

    Args:
        config: Configuration object.
        data_loader: SWEBenchDataLoader instance.
        issue_ids: List of issue IDs to process.
        model_name: Name of the model to use.
        suspected_files: List of suspected files.
        max_iterations: Maximum number of guidance iterations.
        reasoning_type: Type of reasoning to use.
        reflection_iterations: Number of self-reflection iterations.

    Returns:
        List of results.
    """
    from ..models import create_model
    from ..solution.issue_solver import IssueSolver
    from ..utils.patch_validator import PatchValidator

    # Initialize the LLM guidance framework
    guidance = LLMCodeLocationGuidance(config)

    # Initialize patch validator
    patch_validator = PatchValidator(config)

    # Initialize results list
    results = []

    # Process each issue
    for issue_id in issue_ids:
        logging.info(f"Processing issue {issue_id} with memory-efficient LLM guidance")

        # Clear memory before processing new issue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Load the issue
        issue = data_loader.load_issue(issue_id)
        if not issue:
            logging.error(f"Issue {issue_id} not found")
            continue

        # Create the initial guidance prompt - use simpler version if memory efficient
        initial_prompt = guidance.create_guidance_prompt(
            issue,
            suspected_files=suspected_files
        )

        # Get model(s) to use
        if isinstance(model_name, list):
            models_to_use = model_name
        else:
            models_to_use = [model_name]

        # Process with each model
        issue_results = {
            "issue_id": issue_id,
            "issue_description": issue.get("issue_description", ""),
            "solutions": {}
        }

        for model_id in models_to_use:
            logging.info(f"Using model {model_id} for issue {issue_id}")

            # Clear memory before loading model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Initialize model
            model = create_model(model_id, config)

            # Track iterations
            guided_iterations = []
            current_prompt = initial_prompt

            # Run guidance iterations
            for iteration in range(max_iterations):
                logging.info(f"Guidance iteration {iteration + 1}/{max_iterations}")

                # Clear memory before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                # Generate analysis with model
                try:
                    analysis_response = model.generate(current_prompt)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logging.warning("CUDA OOM during guidance. Trying with reduced context...")
                        # Truncate prompt and try again
                        truncated_prompt = _truncate_prompt(current_prompt)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        analysis_response = model.generate(truncated_prompt)
                    else:
                        raise

                # Extract patch from response
                patch = extract_patch_from_response(analysis_response)

                # Clear memory before validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                # Validate patch
                validation_result = patch_validator.validate_patch(patch, issue_id)

                # Record iteration
                guided_iterations.append({
                    "iteration": iteration + 1,
                    "response": analysis_response,
                    "patch": patch,
                    "validation": validation_result
                })

                # If patch is valid or we've reached max iterations, proceed to solver
                if validation_result.get("success", False) or iteration + 1 >= max_iterations:
                    break

                # Otherwise, refine prompt with feedback
                feedback = f"The patch has issues: {validation_result.get('feedback', 'Unknown validation error')}"
                current_prompt = guidance.apply_feedback(current_prompt, analysis_response, feedback)

            # Apply self-reflection if any valid patch was found
            final_solutions = []
            for iteration in guided_iterations:
                if iteration["patch"]:
                    # Use the patch from this iteration
                    patch = iteration["patch"]

                    # Create a solution entry
                    solution = {
                        "iteration": iteration["iteration"],
                        "guidance_response": iteration["response"],
                        "patch": patch,
                        "patch_validation": iteration["validation"],
                        "guided_fix": True
                    }

                    # If we should apply self-reflection, do so
                    if reflection_iterations > 0:
                        from ..reasoning.self_reflection import SelfReflection

                        # Clear memory before reflection
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()

                        reflector = SelfReflection(config, model)

                        # Create context for reflection
                        reflection_context = (
                            f"Initial patch based on guided analysis:\n{patch}\n\n"
                            f"Validation feedback: {iteration['validation'].get('feedback', '')}"
                        )

                        # Apply self-reflection to refine the solution
                        refined_data = reflector.refine_solution(
                            patch,
                            issue.get("issue_description", ""),
                            reflection_context
                        )

                        # Update the solution with reflection results
                        solution["reflections"] = refined_data.get("reflections", [])
                        solution["final_solution"] = refined_data.get("final_solution", patch)

                    final_solutions.append(solution)

            # Add solutions to results
            issue_results["solutions"][model_id] = final_solutions

            # Clean up model to free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Add issue results
        results.append(issue_results)

    return results


def _truncate_prompt(prompt: str, max_length: int = 8000) -> str:
    """
    Truncate a prompt to reduce memory usage.

    Args:
        prompt: The original prompt
        max_length: Maximum length to keep

    Returns:
        Truncated prompt
    """
    if len(prompt) <= max_length:
        return prompt

    # Add a note about truncation
    truncation_note = "\n[Note: The context was too long and has been truncated.]\n\n"

    # Keep the beginning and end of the prompt
    beginning_length = max_length // 2
    ending_length = max_length - beginning_length - len(truncation_note)

    return prompt[:beginning_length] + truncation_note + prompt[-ending_length:]


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
        max_iterations: Maximum number of guidance iterations.
        reasoning_type: Type of reasoning to use.
        reflection_iterations: Number of self-reflection iterations.

    Returns:
        List of results.
    """
    from ..models import create_model
    from ..solution.issue_solver import IssueSolver
    from ..utils.patch_validator import PatchValidator

    # Initialize the LLM guidance framework
    guidance = LLMCodeLocationGuidance(config)

    # Initialize patch validator
    patch_validator = PatchValidator(config)

    # Initialize results list
    results = []

    # Process each issue
    for issue_id in issue_ids:
        logging.info(f"Processing issue {issue_id} with LLM guidance")

        # Load the issue
        issue = data_loader.load_issue(issue_id)
        if not issue:
            logging.error(f"Issue {issue_id} not found")
            continue

        # Create the initial guidance prompt
        initial_prompt = guidance.create_guidance_prompt(
            issue,
            suspected_files=suspected_files
        )

        # Get model(s) to use
        if isinstance(model_name, list):
            models_to_use = model_name
        else:
            models_to_use = [model_name]

        # Process with each model
        issue_results = {
            "issue_id": issue_id,
            "issue_description": issue.get("issue_description", ""),
            "solutions": {}
        }

        for model_id in models_to_use:
            logging.info(f"Using model {model_id} for issue {issue_id}")

            # Initialize model
            model = create_model(model_id, config)

            # Track iterations
            guided_iterations = []
            current_prompt = initial_prompt

            # Run guidance iterations
            for iteration in range(max_iterations):
                logging.info(f"Guidance iteration {iteration + 1}/{max_iterations}")

                # Generate analysis with model
                analysis_response = model.generate(current_prompt)

                # Extract patch from response
                patch = extract_patch_from_response(analysis_response)

                # Validate patch
                validation_result = patch_validator.validate_patch(patch, issue_id)

                # Record iteration
                guided_iterations.append({
                    "iteration": iteration + 1,
                    "prompt": current_prompt,
                    "response": analysis_response,
                    "patch": patch,
                    "validation": validation_result
                })

                # If patch is valid or we've reached max iterations, proceed to solver
                if validation_result.get("success", False) or iteration + 1 >= max_iterations:
                    break

                # Otherwise, refine prompt with feedback
                feedback = f"The patch has issues: {validation_result.get('feedback', 'Unknown validation error')}"
                current_prompt = guidance.apply_feedback(current_prompt, analysis_response, feedback)

            # Create solver with this model
            solver = IssueSolver(config, model_id, reasoning_type, num_iterations=reflection_iterations)

            # Apply self-reflection if any valid patch was found
            final_solutions = []
            for iteration in guided_iterations:
                if iteration["patch"]:
                    # Use the patch from this iteration
                    patch = iteration["patch"]

                    # Create a solution entry
                    solution = {
                        "iteration": iteration["iteration"],
                        "guidance_prompt": iteration["prompt"],
                        "guidance_response": iteration["response"],
                        "patch": patch,
                        "patch_validation": iteration["validation"],
                        "guided_fix": True
                    }

                    # If we should apply self-reflection, do so
                    if reflection_iterations > 0:
                        from ..reasoning.self_reflection import SelfReflection
                        reflector = SelfReflection(config, model)

                        # Create context for reflection
                        reflection_context = (
                            f"Initial patch based on guided analysis:\n{patch}\n\n"
                            f"Validation feedback: {iteration['validation'].get('feedback', '')}"
                        )

                        # Apply self-reflection to refine the solution
                        refined_data = reflector.refine_solution(
                            patch,
                            issue.get("issue_description", ""),
                            reflection_context
                        )

                        # Update the solution with reflection results
                        solution["reflections"] = refined_data.get("reflections", [])
                        solution["final_solution"] = refined_data.get("final_solution", patch)

                    final_solutions.append(solution)

            # Add solutions to results
            issue_results["solutions"][model_id] = final_solutions

        # Add issue results
        results.append(issue_results)

    return results


def extract_patch_from_response(response: str) -> str:
    """
    Extract a Git patch from model response.

    Args:
        response: Model response text.

    Returns:
        Extracted patch string.
    """
    # Look for diff sections with Git format
    diff_pattern = r'(diff --git.*?)(?=^```|\Z)'
    diff_match = re.search(diff_pattern, response, re.MULTILINE | re.DOTALL)

    if diff_match:
        return diff_match.group(1).strip()

    # Look for code blocks that might contain patches
    code_block_pattern = r'```(?:diff|patch)?\n(.*?)```'
    code_match = re.search(code_block_pattern, response, re.MULTILINE | re.DOTALL)

    if code_match:
        patch_content = code_match.group(1).strip()
        # If it looks like a Git patch, return it
        if patch_content.startswith("diff --git"):
            return patch_content

    # No valid patch found
    return ""


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


if __name__ == "__main__":
    main()
