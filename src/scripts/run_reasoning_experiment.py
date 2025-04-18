# src/scripts/run_reasoning_experiment.py

"""
Self-reasoning experiment runner for SWE-Bench.

This script orchestrates experiments that test self-reasoning abilities
on SWE-Bench problems, with analysis of reasoning processes and outcomes.
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.utils.logging import setup_logging
from src.config import ConfigManager
from src.agents import get_agent
from src.datasets import get_dataset
from src.models import get_model
from src.prompts import get_prompt
from src.evaluators import get_evaluator
from src.utils.swe_bench_tester import SWEBenchTester
from src.utils.swe_bench_results import SWEBenchResultsAnalyzer
from src.utils.swe_bench_analysis import RepoDiffAnalyzer


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Self-Reasoning Experiment for SWE-Bench")

    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--dataset", type=str, default="swe_bench_verified",
                        help="Dataset name")
    parser.add_argument("--agent", type=str, default="improved_code_refinement",
                        help="Agent ID")
    parser.add_argument("--model", type=str, required=True,
                        help="Model ID")
    parser.add_argument("--prompt", type=str, default="swe_bench_prompt",
                        help="Prompt template ID")
    parser.add_argument("--evaluator", type=str, default="swe_bench_eval",
                        help="Evaluator ID")
    parser.add_argument("--instance-id", type=str,
                        help="Specific instance ID to test")
    parser.add_argument("--max-instances", type=int, default=1,
                        help="Maximum number of instances to test (0 for all)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum number of self-reasoning iterations")
    parser.add_argument("--output-dir", type=str,
                        help="Base directory to save results")
    parser.add_argument("--repos-dir", type=str, default="data/repositories",
                        help="Directory to store repositories")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout for test execution in seconds")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline test with original patches")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze results after running tests")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    return parser.parse_args()


def run_reasoning_experiment(
        name: str,
        dataset_name: str,
        agent_id: str,
        model_id: str,
        prompt_id: str,
        evaluator_id: str,
        instance_id: Optional[str] = None,
        max_instances: int = 1,
        max_iterations: int = 3,
        output_dir: Optional[str] = None,
        repos_dir: str = "data/repositories",
        timeout: int = 300,
        baseline: bool = False,
        analyze: bool = True,
        log_level: str = "INFO"
) -> Dict[str, Any]:
    """
    Run a self-reasoning experiment on SWE-Bench.

    Args:
        name: Experiment name
        dataset_name: Dataset name
        agent_id: Agent ID
        model_id: Model ID
        prompt_id: Prompt template ID
        evaluator_id: Evaluator ID
        instance_id: Specific instance ID to test (optional)
        max_instances: Maximum number of instances to test
        max_iterations: Maximum number of self-reasoning iterations
        output_dir: Base directory to save results (optional)
        repos_dir: Directory to store repositories
        timeout: Timeout for test execution in seconds
        baseline: Whether to run baseline test with original patches
        analyze: Whether to analyze results after running tests
        log_level: Logging level

    Returns:
        Experiment results
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{name}_{timestamp}"

    # Create output directory
    if output_dir is None:
        output_dir = os.path.join("results", "experiments", experiment_name)

    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_dir, f"{experiment_name}.log")
    logger = setup_logging(log_dir=os.path.dirname(log_file),
                           log_level=log_level,
                           log_to_file=True)

    logger.info(f"Starting self-reasoning experiment: {experiment_name}")
    logger.info(f"Model: {model_id}, Agent: {agent_id}, Prompt: {prompt_id}")

    # Save experiment configuration
    config = {
        "name": name,
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "agent": agent_id,
        "model": model_id,
        "prompt": prompt_id,
        "evaluator": evaluator_id,
        "instance_id": instance_id,
        "max_instances": max_instances,
        "max_iterations": max_iterations,
        "baseline": baseline,
        "repos_dir": repos_dir,
        "timeout": timeout
    }

    config_file = os.path.join(output_dir, "experiment_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # Get dataset
    dataset = get_dataset(dataset_name)
    logger.info(f"Using dataset: {dataset_name} with {len(dataset)} instances")

    # Create agent if not baseline
    agent = None
    if not baseline:
        logger.info(f"Creating agent {agent_id} with model {model_id}")

        try:
            agent = get_agent(
                agent_id=agent_id,
                model_id=model_id,
                prompt_id=prompt_id,
                evaluator_id=evaluator_id,
                config={
                    "max_iterations": max_iterations
                }
            )
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            raise
    else:
        logger.info("Running baseline test with original patches")

    # Create SWE-Bench tester
    results_dir = os.path.join(output_dir, "instance_results")
    os.makedirs(results_dir, exist_ok=True)

    tester = SWEBenchTester(
        repos_dir=repos_dir,
        results_dir=results_dir,
        timeout=timeout,
        max_iterations=max_iterations,
        use_cache=True
    )

    # Determine instances to test
    if instance_id:
        # Test a specific instance
        instance = dataset.get_problem_by_instance_id(instance_id)
        if not instance:
            logger.error(f"Instance {instance_id} not found")
            return {"success": False, "error": f"Instance {instance_id} not found"}

        instances = [instance]
    else:
        # Test all instances (or up to max_instances)
        instances = list(dataset)
        if max_instances > 0:
            instances = instances[:max_instances]

    logger.info(f"Testing {len(instances)} instances")

    # Process instances
    results = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "total_instances": len(instances),
        "successful_instances": 0,
        "failed_instances": 0,
        "error_instances": 0,
        "instance_results": []
    }

    for i, instance in enumerate(instances):
        instance_id = instance.get("instance_id")
        logger.info(f"Testing instance {i + 1}/{len(instances)}: {instance_id}")

        try:
            # Test instance
            instance_result = tester.test_instance(instance, agent)

            # Record results
            results["instance_results"].append({
                "instance_id": instance_id,
                "success": instance_result.get("success", False),
                "iterations": len(instance_result.get("iterations", [])),
                "status": instance_result.get("status")
            })

            # Update counters
            if instance_result.get("success", False):
                results["successful_instances"] += 1
                logger.info(f"✅ Instance {instance_id} succeeded")
            elif instance_result.get("status") == "error":
                results["error_instances"] += 1
                logger.info(f"❌ Instance {instance_id} had an error")
            else:
                results["failed_instances"] += 1
                logger.info(f"❌ Instance {instance_id} failed")

        except Exception as e:
            logger.error(f"Error testing instance {instance_id}: {str(e)}")
            results["error_instances"] += 1
            results["instance_results"].append({
                "instance_id": instance_id,
                "success": False,
                "status": "error",
                "error": str(e)
            })

    # Calculate success rate
    if results["total_instances"] > 0:
        results["success_rate"] = results["successful_instances"] / results["total_instances"]
    else:
        results["success_rate"] = 0.0

    # Save overall results
    results_file = os.path.join(output_dir, "experiment_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Experiment completed with success rate: {results['success_rate']:.2%}")

    # Analyze results if requested
    if analyze:
        try:
            logger.info("Analyzing results...")
            analyzer = SWEBenchResultsAnalyzer(results_dir)

            # Generate summary
            summary = analyzer.summarize_all_results(
                output_file=os.path.join(output_dir, "results_summary.json")
            )

            # Generate report
            report_file = analyzer.generate_report(
                output_dir=os.path.join(output_dir, "reports")
            )

            logger.info(f"Analysis complete. Report saved to {report_file}")

            # Add analysis to results
            results["analysis"] = {
                "summary": summary,
                "report": report_file
            }

        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")

    return results


def compare_with_baseline(experiment_dir: str, baseline_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare experiment results with baseline.

    Args:
        experiment_dir: Directory containing experiment results
        baseline_dir: Directory containing baseline results
        output_dir: Directory to save comparison results (optional)

    Returns:
        Comparison results
    """
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, "analysis")

    os.makedirs(output_dir, exist_ok=True)

    analyzer = SWEBenchResultsAnalyzer()
    comparison = analyzer.compare_runs(
        [experiment_dir, baseline_dir],
        output_file=os.path.join(output_dir, "baseline_comparison.json")
    )

    return comparison


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        # Run the experiment
        results = run_reasoning_experiment(
            name=args.name,
            dataset_name=args.dataset,
            agent_id=args.agent,
            model_id=args.model,
            prompt_id=args.prompt,
            evaluator_id=args.evaluator,
            instance_id=args.instance_id,
            max_instances=args.max_instances,
            max_iterations=args.max_iterations,
            output_dir=args.output_dir,
            repos_dir=args.repos_dir,
            timeout=args.timeout,
            baseline=args.baseline,
            analyze=args.analyze,
            log_level=args.log_level
        )

        # If successful, report summary
        if results.get("success_rate", 0) > 0:
            print(f"\nExperiment completed successfully!")
            print(f"Success rate: {results.get('success_rate', 0):.2%}")
            print(f"Total instances: {results.get('total_instances', 0)}")
            print(f"Successful instances: {results.get('successful_instances', 0)}")

            if args.analyze and "analysis" in results:
                print(f"\nAnalysis report available at: {results['analysis'].get('report', '')}")

    except Exception as e:
        print(f"Error running experiment: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
