#!/usr/bin/env python3
"""
Run comparison between adaptive and non-adaptive approaches.
"""

import argparse
import logging
import json
import time
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from solution.leetcode_solution_pipeline import LeetCodeSolutionPipeline
from data.leetcode_dataloader import LeetCodeDataLoader
from utils.adaptive_statistics import AdaptiveStatisticsTracker


def run_comparison(args):
    """Run comparison between adaptive and non-adaptive approaches."""
    # Load configuration
    config = Config(args.config)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create results directory
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data loader
    data_loader = LeetCodeDataLoader(config)

    # Get problems to test
    if args.problems:
        problem_ids = args.problems.split(",")
    else:
        problems = data_loader.list_problems(difficulty=args.difficulty, limit=args.limit)
        problem_ids = [p["problem_id"] for p in problems]

    results = {
        "adaptive": [],
        "non_adaptive": []
    }

    for problem_id in problem_ids:
        logging.info(f"Processing problem {problem_id}")

        problem_data = data_loader.load_problem_by_id(problem_id)
        if not problem_data:
            logging.error(f"Problem {problem_id} not found")
            continue

        # Test with non-adaptive approach
        config["leetcode"]["adaptive_mode"] = False
        pipeline_non_adaptive = LeetCodeSolutionPipeline(config, args.model)

        start_time = time.time()
        result_non_adaptive = pipeline_non_adaptive.solve_problem(problem_data)
        result_non_adaptive["approach"] = "non_adaptive"
        results["non_adaptive"].append(result_non_adaptive)

        logging.info(
            f"Non-adaptive: {result_non_adaptive['status']} in "
            f"{result_non_adaptive['processing_time']:.2f}s"
        )

        # Test with adaptive approach
        config["leetcode"]["adaptive_mode"] = True
        pipeline_adaptive = LeetCodeSolutionPipeline(config, args.model)

        start_time = time.time()
        result_adaptive = pipeline_adaptive.solve_problem(problem_data)
        result_adaptive["approach"] = "adaptive"
        results["adaptive"].append(result_adaptive)

        logging.info(
            f"Adaptive: {result_adaptive['status']} in "
            f"{result_adaptive['processing_time']:.2f}s"
        )

    # Save comparison results
    comparison_file = results_dir / "comparison_results.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate comparison summary
    generate_comparison_summary(results, results_dir)

    # Save global adaptive statistics
    if pipeline_adaptive.adaptive_stats_tracker:
        pipeline_adaptive.adaptive_stats_tracker.save_global_summary()


def generate_comparison_summary(results, results_dir):
    """Generate summary comparing adaptive vs non-adaptive."""
    summary = {
        "total_problems": len(results["adaptive"]),
        "adaptive": {
            "success_rate": 0,
            "avg_time": 0,
            "avg_candidates": 0
        },
        "non_adaptive": {
            "success_rate": 0,
            "avg_time": 0,
            "avg_candidates": 0
        },
        "improvements": {
            "time_reduction": 0,
            "candidate_reduction": 0,
            "success_rate_change": 0
        }
    }

    # Calculate metrics for each approach
    for approach in ["adaptive", "non_adaptive"]:
        approach_results = results[approach]
        if approach_results:
            successes = sum(1 for r in approach_results if r["status"] == "solved")
            summary[approach]["success_rate"] = successes / len(approach_results)
            summary[approach]["avg_time"] = sum(r["processing_time"] for r in approach_results) / len(approach_results)
            summary[approach]["avg_candidates"] = sum(r["total_candidates"] for r in approach_results) / len(
                approach_results)

    # Calculate improvements
    if summary["non_adaptive"]["avg_time"] > 0:
        summary["improvements"]["time_reduction"] = (
                (summary["non_adaptive"]["avg_time"] - summary["adaptive"]["avg_time"]) /
                summary["non_adaptive"]["avg_time"]
        )

    if summary["non_adaptive"]["avg_candidates"] > 0:
        summary["improvements"]["candidate_reduction"] = (
                (summary["non_adaptive"]["avg_candidates"] - summary["adaptive"]["avg_candidates"]) /
                summary["non_adaptive"]["avg_candidates"]
        )

    summary["improvements"]["success_rate_change"] = (
            summary["adaptive"]["success_rate"] - summary["non_adaptive"]["success_rate"]
    )

    # Save summary
    summary_file = results_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"Total problems: {summary['total_problems']}")
    print("\nNon-Adaptive:")
    print(f"  Success rate: {summary['non_adaptive']['success_rate'] * 100:.1f}%")
    print(f"  Avg time: {summary['non_adaptive']['avg_time']:.2f}s")
    print(f"  Avg candidates: {summary['non_adaptive']['avg_candidates']:.1f}")
    print("\nAdaptive:")
    print(f"  Success rate: {summary['adaptive']['success_rate'] * 100:.1f}%")
    print(f"  Avg time: {summary['adaptive']['avg_time']:.2f}s")
    print(f"  Avg candidates: {summary['adaptive']['avg_candidates']:.1f}")
    print("\nImprovements:")
    print(f"  Time reduction: {summary['improvements']['time_reduction'] * 100:.1f}%")
    print(f"  Candidate reduction: {summary['improvements']['candidate_reduction'] * 100:.1f}%")
    print(f"  Success rate change: {summary['improvements']['success_rate_change'] * 100:+.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare adaptive vs non-adaptive approaches")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--problems", type=str, help="Comma-separated problem IDs")
    parser.add_argument("--difficulty", type=str, help="Problem difficulty")
    parser.add_argument("--limit", type=int, default=5, help="Number of problems")
    parser.add_argument("--output", type=str, default="results/comparison", help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    run_comparison(args)
