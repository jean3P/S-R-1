#!/usr/bin/env python3
"""
Main entry point for the AI system.

This script initializes the system, parses command-line arguments,
and executes the requested operation.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, List, Optional

# Import system components
from src.config import ConfigManager
from src.utils.logging import setup_logging
from src.agents import get_agent
from src.models import get_model
from src.prompts import get_prompt
from src.evaluators import get_evaluator
from src.datasets import get_dataset
from src.utils import tokenizer_config
from src.scripts.run_swe_bench import run_swe_bench_evaluation



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AI Self-Reflection System")

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Run experiment command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--config", type=str, required=True, help="Path to experiment configuration file")
    run_parser.add_argument("--output-dir", type=str, help="Directory to save output")
    run_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                            help="Logging level")

    # List available components command
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument("--type", type=str, required=True,
                             choices=["agents", "models", "prompts", "evaluators", "datasets", "experiments"],
                             help="Component type to list")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a solution for a single problem")
    generate_parser.add_argument("--model", type=str, required=True, help="Model ID to use")
    generate_parser.add_argument("--prompt", type=str, required=True, help="Prompt template ID to use")
    generate_parser.add_argument("--evaluator", type=str, required=True, help="Evaluator ID to use")
    generate_parser.add_argument("--problem", type=str, help="Path to problem file or problem text")
    generate_parser.add_argument("--iterations", type=int, default=3, help="Number of reflection iterations")
    generate_parser.add_argument("--output", type=str, help="Path to save output")

    # Create an experiment configuration command
    create_parser = subparsers.add_parser("create", help="Create an experiment configuration")
    create_parser.add_argument("--name", type=str, required=True, help="Experiment name")
    create_parser.add_argument("--agent", type=str, required=True, help="Agent ID to use")
    create_parser.add_argument("--model", type=str, required=True, help="Model ID to use")
    create_parser.add_argument("--prompt", type=str, required=True, help="Prompt template ID to use")
    create_parser.add_argument("--evaluator", type=str, required=True, help="Evaluator ID to use")
    create_parser.add_argument("--task", type=str, required=True, help="Task description or path to task file")
    create_parser.add_argument("--output", type=str, help="Path to save experiment configuration")

    # SWE-bench command
    swe_bench_parser = subparsers.add_parser("swe-bench", help="Run SWE-bench evaluation")
    swe_bench_parser.add_argument("--experiment", type=str, default="swe_bench_experiment",
                                  help="Experiment configuration name")
    swe_bench_parser.add_argument("--dataset", type=str, default="swe_bench_lite",
                                  help="Dataset name")
    swe_bench_parser.add_argument("--instance-id", type=str,
                                  help="Specific instance ID to evaluate")
    swe_bench_parser.add_argument("--output-dir", type=str, default="results/swe_bench",
                                  help="Directory to save results")
    swe_bench_parser.add_argument("--max-instances", type=int, default=0,
                                  help="Maximum number of instances to evaluate (0 for all)")

    swe_bench_parser.add_argument("--log-level", type=str, default="INFO",
                                  choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                                  help="Logging level to use for swe-bench runs")
    swe_bench_parser.add_argument("--batch-size", type=int, default=1,
                                  help="Batch size for processing instances")


    return parser.parse_args()


def run_experiment(config_path: str, output_dir: Optional[str] = None, log_level: str = "INFO") -> Dict[str, Any]:
    """
    Run an experiment using the specified configuration.

    Args:
        config_path: Path to the experiment configuration file
        output_dir: Directory to save output (overrides configuration)
        log_level: Logging level

    Returns:
        Experiment results
    """
    # Set up logging
    logger = setup_logging(log_level=log_level)
    logger.info(f"Running experiment with configuration: {config_path}")

    # Load experiment configuration
    config_manager = ConfigManager()
    experiment_config = config_manager.load_experiment_config(os.path.splitext(os.path.basename(config_path))[0])

    # Override output directory if specified
    if output_dir:
        experiment_config["output_dir"] = output_dir
        logger.info(f"Output directory overridden to: {output_dir}")

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

    # Get task details
    task = experiment_config["task"]
    initial_prompt = task["initial_prompt"]

    logger.info(f"Starting experiment: {experiment_config.get('name', 'unnamed')}")

    # Run the agent
    try:
        results = agent.reflect(initial_prompt, task)
        logger.info(f"Experiment completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        raise


def list_components(component_type: str) -> None:
    """
    List available components of a specific type.

    Args:
        component_type: Type of components to list
    """
    # Set up logging
    logger = setup_logging()

    if component_type == "agents":
        from src.agents import list_available_agents
        components = list_available_agents()
    elif component_type == "models":
        from src.models import list_available_models
        components = list_available_models()
    elif component_type == "prompts":
        from src.prompts import list_available_prompts
        components = list_available_prompts()
    elif component_type == "evaluators":
        from src.evaluators import list_available_evaluators
        components = list_available_evaluators()
    elif component_type == "datasets":
        from src.datasets import list_available_datasets
        components = list_available_datasets()
    elif component_type == "experiments":
        config_manager = ConfigManager()
        components = config_manager.get_experiment_configs()
    else:
        logger.error(f"Unknown component type: {component_type}")
        return

    if not components:
        print(f"No {component_type} available.")
        return

    print(f"Available {component_type}:")
    for name, description in components.items():
        print(f"- {name}: {description}")


def generate_solution(model_id: str, prompt_id: str, evaluator_id: str, problem: str,
                      iterations: int = 3, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a solution for a single problem.

    Args:
        model_id: Model ID to use
        prompt_id: Prompt template ID to use
        evaluator_id: Evaluator ID to use
        problem: Problem text or path to problem file
        iterations: Number of reflection iterations
        output_path: Path to save output

    Returns:
        Generation results
    """
    # Set up logging
    logger = setup_logging()
    logger.info(f"Generating solution for problem")

    # Load problem
    if os.path.exists(problem):
        with open(problem, 'r') as f:
            problem_text = f.read()
    else:
        problem_text = problem

    # Create a temporary task
    task = {
        "name": "single_generation",
        "language": "python",
        "initial_prompt": problem_text
    }

    # Create an agent
    agent = get_agent(
        agent_id="code_refinement",
        model_id=model_id,
        prompt_id=prompt_id,
        evaluator_id=evaluator_id,
        config={"max_iterations": iterations}
    )

    # Generate solution
    try:
        results = agent.reflect(problem_text, task)

        # Save results if output path is specified
        if output_path:
            from src.utils.file_utils import save_json
            save_json(results, output_path)
            logger.info(f"Results saved to {output_path}")

        logger.info(f"Solution generation completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error generating solution: {str(e)}")
        raise


def create_experiment_config(name: str, agent_id: str, model_id: str, prompt_id: str,
                             evaluator_id: str, task: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create an experiment configuration.

    Args:
        name: Experiment name
        agent_id: Agent ID to use
        model_id: Model ID to use
        prompt_id: Prompt template ID to use
        evaluator_id: Evaluator ID to use
        task: Task description or path to task file
        output_path: Path to save experiment configuration

    Returns:
        Experiment configuration
    """
    # Set up logging
    logger = setup_logging()
    logger.info(f"Creating experiment configuration: {name}")

    # Load task
    if os.path.exists(task):
        with open(task, 'r') as f:
            task_text = f.read()
    else:
        task_text = task

    # Create configuration manager
    config_manager = ConfigManager()

    # Create task configuration
    task_config = {
        "name": name.lower().replace(" ", "_"),
        "language": "python",
        "initial_prompt": task_text
    }

    # Create experiment configuration
    experiment_config = config_manager.create_experiment_config(
        name=name,
        description=f"Experiment: {name}",
        agent_id=agent_id,
        model_id=model_id,
        prompt_id=prompt_id,
        evaluator_id=evaluator_id,
        task=task_config
    )

    # Save configuration if output path is specified
    if output_path:
        from src.utils.file_utils import save_yaml
        save_yaml(experiment_config, output_path)
        logger.info(f"Experiment configuration saved to {output_path}")
    else:
        # Save to default location
        config_id = task_config["name"]
        config_manager.save_experiment_config(config_id, experiment_config)
        logger.info(f"Experiment configuration saved with ID: {config_id}")

    return experiment_config


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_arguments()

    try:
        # Execute the requested command
        if args.command == "run":
            run_experiment(args.config, args.output_dir, args.log_level)
        elif args.command == "list":
            list_components(args.type)
        elif args.command == "generate":
            generate_solution(
                args.model, args.prompt, args.evaluator, args.problem,
                args.iterations, args.output
            )
        elif args.command == "create":
            create_experiment_config(
                args.name, args.agent, args.model, args.prompt,
                args.evaluator, args.task, args.output
            )
        elif args.command == "swe-bench":
            run_swe_bench_evaluation(
                experiment_name=args.experiment,
                dataset_name=args.dataset,
                instance_id=args.instance_id,
                output_dir=args.output_dir,
                log_level=args.log_level,
                max_instances=args.max_instances
            )
        else:
            print("No command specified. Use --help for usage information.")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
