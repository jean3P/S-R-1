#!/usr/bin/env python3
"""
Run SWE-bench evaluation.

This script runs evaluation on the SWE-bench Lite dataset.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any

from src.config import ConfigManager
from src.utils.logging import setup_logging
from src.agents import get_agent
from src.datasets import get_dataset


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SWE-bench Evaluation")

    parser.add_argument("--experiment", type=str, default="swe_bench_experiment",
                        help="Experiment configuration name")
    parser.add_argument("--dataset", type=str, default="swe_bench_lite",
                        help="Dataset name")
    parser.add_argument("--instance-id", type=str,
                        help="Specific instance ID to evaluate. If not provided, evaluate all instances.")
    parser.add_argument("--output-dir", type=str, default="results/swe_bench",
                        help="Directory to save results")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--max-instances", type=int, default=0,
                        help="Maximum number of instances to evaluate (0 for all)")

    return parser.parse_args()


def run_swe_bench_evaluation(
        experiment_name: str,
        dataset_name: str,
        instance_id: str = None,
        output_dir: str = "results/swe_bench",
        log_level: str = "INFO",
        max_instances: int = 0
) -> Dict[str, Any]:
    """
    Run SWE-bench evaluation.

    Args:
        experiment_name: Name of the experiment configuration
        dataset_name: Name of the dataset
        instance_id: Specific instance ID to evaluate (optional)
        output_dir: Directory to save results
        log_level: Logging level
        max_instances: Maximum number of instances to evaluate (0 for all)

    Returns:
        Evaluation results
    """
    # Set up logging
    logger = setup_logging(log_level=log_level)
    logger.info(f"Running SWE-bench evaluation with experiment: {experiment_name}")

    # Load experiment configuration
    config_manager = ConfigManager()
    experiment_config = config_manager.load_experiment_config(experiment_name)

    # Override output directory if specified
    if output_dir:
        experiment_config["output_dir"] = output_dir
        logger.info(f"Output directory set to: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

    # Get the dataset
    dataset = get_dataset(dataset_name)
    logger.info(f"Using dataset: {dataset_name} with {len(dataset)} instances")

    # Create agent
    agent = get_agent(
        agent_id=agent_id,
        model_id=model_id,
        prompt_id=prompt_id,
        evaluator_id=evaluator_id,
        config=experiment_config["agent"].get("config", {})
    )

    # Determine instances to evaluate
    if instance_id:
        # Evaluate a specific instance
        instances = [dataset.get_problem_by_instance_id(instance_id)]
        if not instances[0]:
            logger.error(f"Instance {instance_id} not found")
            return {"success": False, "error": f"Instance {instance_id} not found"}
    else:
        # Evaluate all instances (or up to max_instances)
        instances = list(dataset)
        if max_instances > 0:
            instances = instances[:max_instances]

    # Results container
    results = {
        "experiment": experiment_name,
        "dataset": dataset_name,
        "model": model_id,
        "total_instances": len(instances),
        "successful_instances": 0,
        "instances": []
    }

    # Process each instance
    for i, instance in enumerate(instances):
        instance_id = instance.get("instance_id")
        logger.info(f"Processing instance {i + 1}/{len(instances)}: {instance_id}")

        # Prepare the task for the agent
        try:
            task = dataset.prepare_for_agent(instance_id)
        except Exception as e:
            logger.error(f"Error preparing task for instance {instance_id}: {str(e)}")
            results["instances"].append({
                "instance_id": instance_id,
                "success": False,
                "error": f"Error preparing task: {str(e)}"
            })
            continue

        # Run the agent
        try:
            instance_results = agent.reflect(task["initial_prompt"], task)

            # Determine success
            instance_success = instance_results.get("success", False)

            # Update results
            if instance_success:
                results["successful_instances"] += 1

            # Store instance results
            results["instances"].append({
                "instance_id": instance_id,
                "success": instance_success,
                "iterations": len(instance_results.get("iterations", [])),
                "best_solution": instance_results.get("best_solution")
            })

            # Save detailed instance results
            instance_output_path = os.path.join(output_dir, f"{instance_id}.json")
            with open(instance_output_path, 'w') as f:
                json.dump(instance_results, f, indent=2)

            logger.info(f"Instance {instance_id} completed. Success: {instance_success}")

        except Exception as e:
            logger.error(f"Error processing instance {instance_id}: {str(e)}")
            results["instances"].append({
                "instance_id": instance_id,
                "success": False,
                "error": str(e)
            })

    # Calculate success rate
    results["success_rate"] = results["successful_instances"] / results["total_instances"] if results[
                                                                                                  "total_instances"] > 0 else 0.0

    # Save overall results
    overall_output_path = os.path.join(output_dir, "overall_results.json")
    with open(overall_output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation completed. Success rate: {results['success_rate']:.2%}")
    logger.info(f"Results saved to {output_dir}")

    return results


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        run_swe_bench_evaluation(
            experiment_name=args.experiment,
            dataset_name=args.dataset,
            instance_id=args.instance_id,
            output_dir=args.output_dir,
            log_level=args.log_level,
            max_instances=args.max_instances
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
