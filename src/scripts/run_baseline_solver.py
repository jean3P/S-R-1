# src/scripts/run_baseline_solver.py

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config.config import Config
from src.data.leetcode_dataloader import LeetCodeDataLoader
from src.solution.leetcode_baseline_pipeline import LeetCodeBaselinePipeline


def main():
    parser = argparse.ArgumentParser(description="Run baseline LeetCode solver (single attempt)")

    parser.add_argument("--config", type=str, default="configs/experiments/leetcode_solver.yaml")
    parser.add_argument("--model", type=str, default="deepseek-r1-distill")
    parser.add_argument("--difficulty", type=str, choices=["Easy", "Medium", "Hard"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/baseline")
    parser.add_argument("--use-code-eval", action="store_true")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Load config
    config = Config(args.config)

    # Update config with args
    config["evaluation"]["results_dir"] = args.output
    if args.use_code_eval:
        config["evaluation"]["use_code_eval"] = True

    # Initialize data loader
    data_loader = LeetCodeDataLoader(config)

    # Get problems
    problems = data_loader.list_problems(difficulty=args.difficulty, limit=args.limit)
    problem_ids = [p["problem_id"] for p in problems]

    logging.info(f"Running baseline solver on {len(problem_ids)} problems")

    # Initialize pipeline
    pipeline = LeetCodeBaselinePipeline(config, args.model)

    # Process each problem
    results = []
    solved_count = 0

    for problem_id in problem_ids:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Processing: {problem_id}")
        logging.info(f"{'=' * 60}")

        problem_data = data_loader.load_problem_by_id(problem_id)

        if not problem_data:
            logging.error(f"Problem {problem_id} not found")
            continue

        result = pipeline.solve_problem(problem_data)
        results.append(result)

        if result["status"] == "solved":
            solved_count += 1

        logging.info(f"Status: {result['status']} | Time: {result['processing_time']:.2f}s")

    # Summary
    total = len(results)
    solve_rate = (solved_count / total * 100) if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"BASELINE RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total problems: {total}")
    print(f"Solved: {solved_count} ({solve_rate:.1f}%)")
    print(f"Failed: {total - solved_count}")

    # Save summary
    summary = {
        "model": args.model,
        "total_problems": total,
        "solved": solved_count,
        "solve_rate": solve_rate,
        "difficulty": args.difficulty,
        "results": results
    }

    summary_file = Path(args.output) / "baseline_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"\nResults saved to: {summary_file}")


if __name__ == "__main__":
    main()
