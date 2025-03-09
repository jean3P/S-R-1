# src/utils/compare_models.py

import json
import os
import glob
from typing import Dict, List, Any


def load_result_file(file_path: str) -> Dict[str, Any]:
    """Load a result JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from experiment results."""
    metrics = {
        "model": result.get("task", {}).get("model_id", "unknown"),
        "success": result.get("success", False),
        "iterations": len(result.get("iterations", [])),
        "total_tokens": result.get("metrics", {}).get("total_tokens_used", 0),
        "generation_time": result.get("metrics", {}).get("average_generation_time", 0),
    }

    # Add code metrics from best solution if available
    best_iteration = result.get("best_iteration")
    if best_iteration is not None and best_iteration > 0:
        iteration = result["iterations"][best_iteration - 1]
        code_metrics = iteration.get("code_metrics", {})
        metrics.update({
            "complexity": code_metrics.get("complexity", 0),
            "docstring_coverage": code_metrics.get("docstring_coverage", 0),
            "function_count": code_metrics.get("function_count", 0),
        })

    return metrics


def compare_models(results_dir: str) -> List[Dict[str, Any]]:
    """Compare results from different models."""
    result_files = glob.glob(os.path.join(results_dir, "**/experiment_results.json"), recursive=True)

    comparison = []
    for file_path in result_files:
        result = load_result_file(file_path)
        metrics = extract_metrics(result)
        metrics["result_file"] = file_path
        comparison.append(metrics)

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()

    comparison = compare_models(args.results_dir)

    # Print comparison table
    print("\nModel Comparison:")
    print("=" * 100)
    print(f"{'Model':<20} {'Success':<10} {'Iterations':<10} {'Tokens':<10} {'Time (s)':<10} {'Complexity':<10}")
    print("-" * 100)

    for metrics in comparison:
        print(
            f"{metrics['model']:<20} {str(metrics['success']):<10} {metrics['iterations']:<10} {metrics['total_tokens']:<10} {metrics['generation_time']:.2f}{'s':<6} {metrics.get('complexity', 'N/A'):<10}")