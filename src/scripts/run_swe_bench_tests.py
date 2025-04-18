# src/scripts/run_swe_bench_tests.py

"""
Enhanced SWE-Bench testing script.

This script provides an improved approach to testing SWE-Bench dataset rows,
with support for self-reasoning agents, batched testing, and detailed reporting.
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional

from src.utils.logging import setup_logging
from src.config import ConfigManager
from src.agents import get_agent
from src.datasets import get_dataset
from src.utils.swe_bench_tester import SWEBenchTester


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced SWE-Bench Testing")

    parser.add_argument("--experiment", type=str, default="swe_bench_experiment",
                        help="Experiment configuration name")
    parser.add_argument("--dataset", type=str, default="swe_bench_verified",
                        help="Dataset name")
    parser.add_argument("--instance-id", type=str,
                        help="Specific instance ID to test")
    parser.add_argument("--output-dir", type=str, default="results/swe_bench",
                        help="Directory to save results")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--max-instances", type=int, default=0,
                        help="Maximum number of instances to test (0 for all)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Maximum number of self-reasoning iterations")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout for test execution in seconds")
    parser.add_argument("--repos-dir", type=str, default="data/repositories",
                        help="Directory to store repositories")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="Disable self-reasoning (use original patches)")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use cached repositories")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for testing (0 for all)")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        run_swe_bench_tests(
            experiment_name=args.experiment,
            dataset_name=args.dataset,
            instance_id=args.instance_id,
            output_dir=args.output_dir,
            log_level=args.log_level,
            max_instances=args.max_instances,
            max_iterations=args.max_iterations,
            timeout=args.timeout,
            repos_dir=args.repos_dir,
            use_reasoning=not args.no_reasoning,
            use_cache=args.use_cache,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


def run_swe_bench_tests(
        experiment_name: str,
        dataset_name: str,
        instance_id: Optional[str] = None,
        output_dir: str = "results/swe_bench",
        log_level: str = "INFO",
        max_instances: int = 0,
        max_iterations: int = 3,
        timeout: int = 300,
        repos_dir: str = "data/repositories",
        use_reasoning: bool = True,
        use_cache: bool = True,
        batch_size: int = 1
) -> Dict[str, Any]:
    """
    Run enhanced SWE-Bench tests.

    Args:
        experiment_name: Name of the experiment configuration
        dataset_name: Name of the dataset
        instance_id: Specific instance ID to test (optional)
        output_dir: Directory to save results
        log_level: Logging level
        max_instances: Maximum number of instances to test (0 for all)
        max_iterations: Maximum number of self-reasoning iterations
        timeout: Timeout for test execution in seconds
        repos_dir: Directory to store repositories
        use_reasoning: Whether to use self-reasoning or original patches
        use_cache: Whether to use cached repositories
        batch_size: Batch size for testing (0 for all)

    Returns:
        Test results
    """
    # Set up logging
    logger = setup_logging(log_level=log_level)
    logger.info(f"Running enhanced SWE-Bench tests with experiment: {experiment_name}")

    # Load experiment configuration
    config_manager = ConfigManager()
    experiment_config = config_manager.load_experiment_config(experiment_name)

    # Override output directory if specified
    if output_dir:
        experiment_config["output_dir"] = output_dir
        logger.info(f"Output directory set to: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create SWE-Bench tester
    tester = SWEBenchTester(
        repos_dir=repos_dir,
        results_dir=output_dir,
        timeout=timeout,
        max_iterations=max_iterations,
        use_cache=use_cache
    )

    # Get the dataset
    dataset = get_dataset(dataset_name)
    logger.info(f"Using dataset: {dataset_name} with {len(dataset)} instances")

    # Create agent if using reasoning
    agent = None
    if use_reasoning:
        # Extract component IDs
        agent_id = experiment_config["agent"]["id"]
        model_id = experiment_config["model"]["id"]
        prompt_id = experiment_config["prompt"]["id"]
        evaluator_id = experiment_config["evaluator"]["id"]

        # Get component instances
        logger.info(f"Using agent: {agent_id}")
        logger.info(f"Using model: {model_id}")
        logger.info(f"Using prompt: {prompt_id}")
        logger.info(f"Using evaluator: {evaluator_id}")

        agent = get_agent(
            agent_id=agent_id,
            model_id=model_id,
            prompt_id=prompt_id,
            evaluator_id=evaluator_id,
            config=experiment_config["agent"].get("config", {})
        )
    else:
        logger.info("Self-reasoning disabled, using original patches")

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
    if batch_size > 1:
        # Batch processing
        return tester.test_batch(instances, agent, max_instances)
    else:
        # Process one by one
        results = {
            "instance_results": [],
            "total_instances": len(instances),
            "successful_instances": 0,
            "failed_instances": 0,
            "error_instances": 0
        }

        for i, instance in enumerate(instances):
            logger.info(f"Testing instance {i + 1}/{len(instances)}: {instance.get('instance_id')}")

            try:
                instance_result = tester.test_instance(instance, agent)

                # Record results
                results["instance_results"].append({
                    "instance_id": instance.get("instance_id"),
                    "success": instance_result.get("success", False),
                    "iterations": len(instance_result.get("iterations", [])),
                    "status": instance_result.get("status")
                })

                # Update counters
                if instance_result.get("success", False):
                    results["successful_instances"] += 1
                elif instance_result.get("status") == "error":
                    results["error_instances"] += 1
                else:
                    results["failed_instances"] += 1

            except Exception as e:
                logger.error(f"Error testing instance {instance.get('instance_id')}: {str(e)}")
                results["error_instances"] += 1
                results["instance_results"].append({
                    "instance_id": instance.get("instance_id"),
                    "success": False,
                    "status": "error",
                    "error": str(e)
                })

        # Calculate success rate
        if results["total_instances"] > 0:
            results["success_rate"] = results["successful_instances"] / results["total_instances"]
        else:
            results["success_rate"] = 0.0

        return results
