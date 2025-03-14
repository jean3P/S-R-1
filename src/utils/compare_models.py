# src/utils/compare_models.py

"""
Compare results from different model experiments.

This script analyzes result files from different model runs and produces
a comparison table showing key metrics.
"""

import json
import os
import glob
import re
from typing import Dict, List, Any, Set


def load_result_file(file_path: str) -> Dict[str, Any]:
    """Load a result JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_model_name_from_path(file_path: str) -> str:
    """Extract model name from directory path."""
    # Try to extract from pattern like "qwq_preview_20250309_192609"
    dir_name = os.path.basename(os.path.dirname(file_path))

    # Try to match model name patterns in the directory path
    model_patterns = [
        r'(\w+)_experiment_\d+_\d+',  # Pattern: name_experiment_date_time
        r'(\w+)_preview_\d+_\d+',  # Pattern: name_preview_date_time
        r'(\w+)_coder_\d+_\d+',  # Pattern: name_coder_date_time
        r'(\w+)_qwen_\d+_\d+',  # Pattern: name_qwen_date_time
    ]

    for pattern in model_patterns:
        match = re.match(pattern, dir_name)
        if match:
            return match.group(1)

    # Alternative approach: first part before underscore
    parts = dir_name.split('_')
    if parts:
        return parts[0]

    return "unknown"


def extract_metrics(result: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    """Extract key metrics from experiment results."""
    # Try to get model ID from different locations
    model_id = "unknown"

    # Try to get from task
    if "task" in result and isinstance(result["task"], dict) and "name" in result["task"]:
        task_name = result["task"]["name"]
        if "qwq" in task_name.lower():
            model_id = "qwq_preview"
        elif "qwen" in task_name.lower():
            model_id = "qwen_coder"
        elif "deepseek" in task_name.lower():
            model_id = "deepseek_qwen"

    # If not found, try to extract from file path
    if model_id == "unknown":
        model_id = extract_model_name_from_path(file_path)

        # Fix common model name issues
        if "qwq" in model_id.lower():
            model_id = "qwq_preview"
        elif "qwen" in model_id.lower() and "deepseek" not in file_path.lower():
            model_id = "qwen_coder"
        elif "deepseek" in model_id.lower() or "deepseek" in file_path.lower():
            model_id = "deepseek_qwen"

    metrics = {
        "model": model_id,
        "success": result.get("success", False),
        "iterations": len(result.get("iterations", [])),
        "total_tokens": result.get("metrics", {}).get("total_tokens_used", 0),
        "generation_time": result.get("metrics", {}).get("average_generation_time", 0),
    }

    # Add code metrics from best solution if available
    best_iteration = result.get("best_iteration")
    if best_iteration is not None and best_iteration > 0 and best_iteration <= len(result.get("iterations", [])):
        try:
            iteration = result["iterations"][best_iteration - 1]
            code_metrics = iteration.get("code_metrics", {})
            metrics.update({
                "complexity": code_metrics.get("complexity", 0),
                "docstring_coverage": code_metrics.get("docstring_coverage", 0),
                "function_count": code_metrics.get("function_count", 0),
            })
        except IndexError:
            print(f"Warning: Best iteration index {best_iteration} out of range for {file_path}")

    # Add file path for debugging
    metrics["result_file"] = file_path

    return metrics


def compare_models(results_dir: str) -> List[Dict[str, Any]]:
    """Compare results from different models."""
    # Look for any JSON files in the results directory
    result_files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
    print(f"Found {len(result_files)} JSON files in {results_dir}")

    # Keep track of processed models to avoid duplicates
    processed_models = set()

    # Filter out non-result files
    valid_result_files = []
    for file_path in result_files:
        # Skip files in specific directories or with specific names
        if "config" in file_path or "cache" in file_path or "metadata" in file_path:
            continue

        # Check if file has proper experiment result structure
        try:
            with open(file_path, 'r') as f:
                # Read first few bytes to check structure
                start = f.read(100)
                if '"iterations"' in start or '"task"' in start or '"metrics"' in start:
                    valid_result_files.append(file_path)
                    print(f"Valid result file found: {file_path}")
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    # Group files by model to select only one per model
    model_files = {}
    for file_path in valid_result_files:
        try:
            result = load_result_file(file_path)
            # Get preliminary model info to group files
            model_id = "unknown"

            # Try to identify model from task name
            if "task" in result and isinstance(result["task"], dict) and "name" in result["task"]:
                task_name = result["task"]["name"].lower()
                if "qwq" in task_name:
                    model_id = "qwq_preview"
                elif "qwen" in task_name:
                    model_id = "qwen_coder"
                elif "deepseek" in task_name:
                    model_id = "deepseek_qwen"

            # If still unknown, try path-based identification
            if model_id == "unknown":
                path_lower = file_path.lower()
                if "qwq" in path_lower:
                    model_id = "qwq_preview"
                elif "qwen" in path_lower and "deepseek" not in path_lower:
                    model_id = "qwen_coder"
                elif "deepseek" in path_lower:
                    model_id = "deepseek_qwen"
                else:
                    model_id = extract_model_name_from_path(file_path)

            # Store the most recent file for each model (assuming newer = better)
            if model_id not in model_files or os.path.getmtime(file_path) > os.path.getmtime(model_files[model_id]):
                model_files[model_id] = file_path

        except Exception as e:
            print(f"Error examining file {file_path}: {e}")

    # Process the selected files (one per model)
    comparison = []
    for model_id, file_path in model_files.items():
        try:
            print(f"Processing metrics for model {model_id} from {file_path}")
            result = load_result_file(file_path)
            metrics = extract_metrics(result, file_path)
            comparison.append(metrics)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return comparison


def print_comparison_table(comparison: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    if not comparison:
        print("No valid result files found.")
        return

    print("\nModel Comparison:")
    print("=" * 100)
    print(f"{'Model':<20} {'Success':<10} {'Iterations':<10} {'Tokens':<10} {'Time (s)':<10} {'Complexity':<10}")
    print("-" * 100)

    for metrics in comparison:
        try:
            print(
                f"{metrics['model']:<20} {str(metrics['success']):<10} {metrics['iterations']:<10} "
                f"{metrics['total_tokens']:<10} {metrics['generation_time']:.2f}{'s':<6} "
                f"{metrics.get('complexity', 'N/A'):<10}")
        except KeyError as e:
            print(f"Error displaying metrics: {e} - {metrics}")


def save_comparison_to_file(comparison: List[Dict[str, Any]], output_file: str) -> None:
    """Save comparison results to a file."""
    with open(output_file, 'w') as f:
        f.write("\nModel Comparison:\n")
        f.write("=" * 100 + "\n")
        f.write(
            f"{'Model':<20} {'Success':<10} {'Iterations':<10} {'Tokens':<10} {'Time (s)':<10} {'Complexity':<10}\n")
        f.write("-" * 100 + "\n")

        for metrics in comparison:
            f.write(
                f"{metrics['model']:<20} {str(metrics['success']):<10} {metrics['iterations']:<10} "
                f"{metrics['total_tokens']:<10} {metrics['generation_time']:.2f}{'s':<6} "
                f"{metrics.get('complexity', 'N/A'):<10}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output", help="Output file to save comparison results (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        print(f"Analyzing results in directory: {args.results_dir}")

    comparison = compare_models(args.results_dir)

    # Print comparison table
    print_comparison_table(comparison)

    # Save to file if requested
    if args.output:
        save_comparison_to_file(comparison, args.output)
        print(f"\nComparison saved to {args.output}")
