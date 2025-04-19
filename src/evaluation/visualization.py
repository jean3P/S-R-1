# src/evaluation/visualization.py

import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Visualizer for evaluation results.
    """

    def __init__(self, config):
        """
        Initialize the visualizer.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.results_dir = Path(config["evaluation"]["results_dir"])

        # Create results directory if it doesn't exist
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)

    def visualize_model_comparison(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Visualize comparison between models across multiple issues.

        Args:
            results: List of results dictionaries.
            output_path: Optional path to save the visualization.

        Returns:
            Path to the saved visualization.
        """
        # Extract model names
        all_models = set()
        for result in results:
            if "solutions" in result:
                all_models.update(result["solutions"].keys())

        # Convert to sorted list
        models = sorted(list(all_models))

        # Extract metrics for each model
        metrics = ["success_rate", "code_quality", "patch_quality", "overall_score"]

        # Initialize data structure
        model_metrics = {model: {metric: [] for metric in metrics} for model in models}

        # Collect data
        for result in results:
            if "solutions" not in result:
                continue

            for model in models:
                if model not in result["solutions"]:
                    continue

                model_solutions = result["solutions"][model]

                # Average metrics across iterations
                for metric in metrics:
                    values = []
                    for solution in model_solutions:
                        if "evaluation" in solution and metric in solution["evaluation"]:
                            values.append(solution["evaluation"][metric])

                    if values:
                        model_metrics[model][metric].append(np.mean(values))

        # Create the visualization
        plt.figure(figsize=(12, 8))

        # For each metric, create a subplot
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)

            # Prepare data for box plot
            data = [model_metrics[model][metric] for model in models]

            # Create box plot
            plt.boxplot(data, labels=models)
            plt.title(f"{metric.replace('_', ' ').title()}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        # Save the figure
        if output_path is None:
            output_path = self.results_dir / "model_comparison.png"

        plt.savefig(output_path)
        plt.close()

        return str(output_path)

    def visualize_iteration_improvement(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Visualize improvement across iterations for each model.

        Args:
            results: List of results dictionaries.
            output_path: Optional path to save the visualization.

        Returns:
            Path to the saved visualization.
        """
        # Extract model names
        all_models = set()
        for result in results:
            if "solutions" in result:
                all_models.update(result["solutions"].keys())

        # Convert to sorted list
        models = sorted(list(all_models))

        # Extract metrics
        metric = "overall_score"  # Focus on overall score

        # Initialize data structure
        iteration_scores = {model: {1: [], 2: [], 3: []} for model in models}

        # Collect data
        for result in results:
            if "solutions" not in result:
                continue

            for model in models:
                if model not in result["solutions"]:
                    continue

                model_solutions = result["solutions"][model]

                for solution in model_solutions:
                    if "evaluation" in solution and metric in solution["evaluation"]:
                        iteration = solution.get("iteration", 1)
                        iteration_scores[model][iteration].append(solution["evaluation"][metric])

        # Create the visualization
        plt.figure(figsize=(10, 6))

        # Prepare data for plotting
        iterations = [1, 2, 3]

        for model in models:
            # Calculate mean scores for each iteration
            means = [np.mean(iteration_scores[model][i]) if iteration_scores[model][i] else 0 for i in iterations]

            # Plot line
            plt.plot(iterations, means, marker='o', label=model)

        plt.title(f"Improvement Across Iterations ({metric.replace('_', ' ').title()})")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.xticks(iterations)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the figure
        if output_path is None:
            output_path = self.results_dir / "iteration_improvement.png"

        plt.savefig(output_path)
        plt.close()

        return str(output_path)
