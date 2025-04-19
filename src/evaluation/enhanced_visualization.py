# src/evaluation/enhanced_visualization.py

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class EnhancedVisualizer:
    """
    Enhanced visualization for SWE-bench benchmark results.
    """

    def __init__(self, config):
        """
        Initialize the enhanced visualizer.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.results_dir = Path(config["evaluation"]["results_dir"])
        self.viz_dir = self.results_dir / "visualizations"

        # Create visualization directory if it doesn't exist
        if not self.viz_dir.exists():
            self.viz_dir.mkdir(parents=True)

        # Set up matplotlib styling
        plt.style.use('ggplot')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']

    def visualize_all(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate all visualizations for the benchmark results.

        Args:
            results: List of benchmark results.

        Returns:
            Dictionary mapping visualization names to file paths.
        """
        visualization_paths = {}

        # Generate all visualizations
        visualization_paths["model_comparison"] = self.visualize_model_comparison(results)
        visualization_paths["iteration_improvement"] = self.visualize_iteration_improvement(results)
        visualization_paths["performance_metrics"] = self.visualize_performance_metrics(results)
        visualization_paths["success_rate_boxplot"] = self.visualize_success_rate_boxplot(results)
        visualization_paths["execution_time"] = self.visualize_execution_time(results)
        visualization_paths["performance_heatmap"] = self.visualize_performance_heatmap(results)
        visualization_paths["code_quality_scatter"] = self.visualize_code_quality_scatter(results)
        visualization_paths["reflection_effectiveness"] = self.visualize_reflection_effectiveness(results)

        return visualization_paths

    def visualize_model_comparison(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Visualize comparison between models across multiple issues.

        Args:
            results: List of results dictionaries.
            output_path: Optional path to save the visualization.

        Returns:
            Path to the saved visualization.
        """
        # Extract model names and metrics
        all_models = set()
        for result in results:
            if "solutions" in result:
                all_models.update(result["solutions"].keys())

        models = sorted(list(all_models))
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
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        for i, (ax, metric) in enumerate(zip(axes.flatten(), metrics)):
            data = [model_metrics[model][metric] for model in models]

            # Create box plot
            ax.boxplot(data, labels=models, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'),
                       medianprops=dict(color='red'))

            # Add data points
            for j, d in enumerate(data):
                x = np.random.normal(j + 1, 0.04, size=len(d))
                ax.scatter(x, d, alpha=0.6, s=30, c=self.colors[j % len(self.colors)])

            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        if output_path is None:
            output_path = self.viz_dir / "model_comparison.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        plt.figure(figsize=(12, 8))

        # Prepare data for plotting
        iterations = [1, 2, 3]

        # Calculate statistics
        model_means = {}
        model_stds = {}

        for model in models:
            # Calculate mean scores for each iteration
            means = [np.mean(iteration_scores[model][i]) if iteration_scores[model][i] else 0 for i in iterations]
            stds = [np.std(iteration_scores[model][i]) if iteration_scores[model][i] and len(
                iteration_scores[model][i]) > 1 else 0 for i in iterations]

            model_means[model] = means
            model_stds[model] = stds

            # Plot line with error shading
            plt.plot(iterations, means, marker='o', linewidth=2, label=model,
                     color=self.colors[models.index(model) % len(self.colors)])
            plt.fill_between(iterations,
                             [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)],
                             alpha=0.2, color=self.colors[models.index(model) % len(self.colors)])

        plt.title(f"Improvement Across Self-Reflection Iterations\n({metric.replace('_', ' ').title()})", fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(iterations, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add improvement percentages
        for i, model in enumerate(models):
            means = model_means[model]
            if means[0] > 0:  # Avoid division by zero
                improvement = (means[-1] - means[0]) / means[0] * 100
                plt.annotate(f"{improvement:.1f}%",
                             xy=(iterations[-1], means[-1]),
                             xytext=(10, (-1) ** i * 15),  # Alternate above/below
                             textcoords="offset points",
                             fontsize=10,
                             color=self.colors[i % len(self.colors)],
                             weight='bold')

        plt.tight_layout()

        # Save the figure
        if output_path is None:
            output_path = self.viz_dir / "iteration_improvement.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def visualize_performance_metrics(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Visualize performance metrics for each model.

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

        models = sorted(list(all_models))
        metrics = ["success_rate", "code_quality", "patch_quality", "overall_score"]

        # Initialize data structure
        model_data = []

        # Collect data (using final iteration only)
        for result in results:
            if "solutions" not in result:
                continue

            for model in models:
                if model not in result["solutions"]:
                    continue

                model_solutions = result["solutions"][model]

                # Get the final iteration solution
                final_solutions = [s for s in model_solutions if s.get("iteration") == 3]
                if not final_solutions:
                    final_solution = max(model_solutions, key=lambda s: s.get("iteration", 0))
                else:
                    final_solution = final_solutions[0]

                # Extract metrics
                if "evaluation" in final_solution:
                    eval_data = final_solution["evaluation"]

                    for metric in metrics:
                        if metric in eval_data:
                            model_data.append({
                                "Model": model,
                                "Metric": metric.replace('_', ' ').title(),
                                "Score": eval_data[metric]
                            })

        # Convert to DataFrame
        df = pd.DataFrame(model_data)

        # Create the visualization
        plt.figure(figsize=(14, 8))

        # Create grouped bar chart
        sns.barplot(x="Model", y="Score", hue="Metric", data=df, palette="viridis")

        plt.title("Performance Metrics by Model", fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.legend(title="Metric", fontsize=12, title_fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha="right", fontsize=12)

        plt.tight_layout()

        # Save the figure
        if output_path is None:
            output_path = self.viz_dir / "performance_metrics.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def visualize_success_rate_boxplot(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Visualize success rate distribution for each model.

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

        models = sorted(list(all_models))

        # Initialize data structure
        success_rates = {model: [] for model in models}

        # Collect data (using final iteration only)
        for result in results:
            if "solutions" not in result:
                continue

            for model in models:
                if model not in result["solutions"]:
                    continue

                model_solutions = result["solutions"][model]

                # Get the final iteration solution
                final_solutions = [s for s in model_solutions if s.get("iteration") == 3]
                if not final_solutions:
                    final_solution = max(model_solutions, key=lambda s: s.get("iteration", 0))
                else:
                    final_solution = final_solutions[0]

                # Extract success rate
                if "evaluation" in final_solution and "success_rate" in final_solution["evaluation"]:
                    success_rates[model].append(final_solution["evaluation"]["success_rate"])

        # Create the visualization
        plt.figure(figsize=(12, 8))

        # Create violin plot with swarm overlay
        data = [success_rates[model] for model in models]

        # Box plot base
        box_parts = plt.boxplot(data, labels=models, patch_artist=True,
                                boxprops=dict(facecolor='lightblue', alpha=0.6),
                                medianprops=dict(color='darkred', linewidth=2),
                                showfliers=False)

        # Add swarm plot
        for i, model in enumerate(models):
            # Add jitter
            x = np.random.normal(i + 1, 0.05, size=len(success_rates[model]))
            plt.scatter(x, success_rates[model], alpha=0.7, s=30,
                        color=self.colors[i % len(self.colors)], zorder=3)

            # Add mean marker
            mean_val = np.mean(success_rates[model]) if success_rates[model] else 0
            plt.scatter(i + 1, mean_val, color='white', edgecolor='black',
                        s=80, zorder=4, marker='*')

            # Add mean annotation
            plt.annotate(f"μ={mean_val:.2f}", (i + 1, mean_val),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=10, fontweight='bold')

        plt.title("Success Rate Distribution by Model", fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Success Rate', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha="right", fontsize=12)

        plt.tight_layout()

        # Save the figure
        if output_path is None:
            output_path = self.viz_dir / "success_rate_boxplot.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def visualize_execution_time(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Visualize execution time for each model.

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

        models = sorted(list(all_models))

        # Initialize data structure
        execution_times = {model: [] for model in models}

        # Collect data
        for result in results:
            if "solutions" not in result:
                continue

            for model in models:
                if model not in result["solutions"]:
                    continue

                model_solutions = result["solutions"][model]

                for solution in model_solutions:
                    if "execution_time" in solution:
                        execution_times[model].append(solution["execution_time"])

        # Calculate statistics
        model_stats = {}
        for model in models:
            if execution_times[model]:
                model_stats[model] = {
                    "mean": np.mean(execution_times[model]),
                    "median": np.median(execution_times[model]),
                    "std": np.std(execution_times[model]),
                    "total": np.sum(execution_times[model])
                }
            else:
                model_stats[model] = {
                    "mean": 0,
                    "median": 0,
                    "std": 0,
                    "total": 0
                }

        # Create the visualization - dual axis plot (mean time and total time)
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Mean times (bar chart)
        x = np.arange(len(models))
        bar_width = 0.35

        # Plot mean times with error bars
        bars = ax1.bar(x, [model_stats[model]["mean"] for model in models],
                       width=bar_width,
                       yerr=[model_stats[model]["std"] for model in models],
                       capsize=5, alpha=0.7,
                       color=[self.colors[i % len(self.colors)] for i in range(len(models))])

        ax1.set_xlabel('Model', fontsize=14)
        ax1.set_ylabel('Mean Execution Time (s)', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create second y-axis for total time (line chart)
        ax2 = ax1.twinx()
        ax2.plot(x, [model_stats[model]["total"] for model in models],
                 'ro-', linewidth=2, markersize=8)
        ax2.set_ylabel('Total Execution Time (s)', color='red', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='red')

        # Add annotations for mean times
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}s',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=10)

        plt.title("Execution Time Analysis by Model", fontsize=16)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Add legend
        ax1.legend(['Mean Time'], loc='upper left')
        ax2.legend(['Total Time'], loc='upper right')

        plt.tight_layout()

        # Save the figure
        if output_path is None:
            output_path = self.viz_dir / "execution_time.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def visualize_performance_heatmap(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Visualize performance heatmap across models and issues.

        Args:
            results: List of results dictionaries.
            output_path: Optional path to save the visualization.

        Returns:
            Path to the saved visualization.
        """
        # Extract model names and issue IDs
        all_models = set()
        all_issues = set()

        for result in results:
            if "solutions" in result and "issue_id" in result:
                all_models.update(result["solutions"].keys())
                all_issues.add(result["issue_id"])

        models = sorted(list(all_models))
        issues = sorted(list(all_issues))

        # Limit to top 20 issues if there are too many
        if len(issues) > 20:
            issues = issues[:20]

        # Initialize data matrix
        performance_matrix = np.zeros((len(models), len(issues)))

        # Collect data
        for i, model in enumerate(models):
            for j, issue_id in enumerate(issues):
                # Find the result for this issue
                issue_result = next((r for r in results if r.get("issue_id") == issue_id), None)

                if not issue_result or "solutions" not in issue_result or model not in issue_result["solutions"]:
                    performance_matrix[i, j] = np.nan  # No data
                    continue

                solutions = issue_result["solutions"][model]

                # Get the final solution
                final_solutions = [s for s in solutions if s.get("iteration") == 3]
                if not final_solutions:
                    final_solution = max(solutions, key=lambda s: s.get("iteration", 0))
                else:
                    final_solution = final_solutions[0]

                # Get overall score
                if "evaluation" in final_solution and "overall_score" in final_solution["evaluation"]:
                    performance_matrix[i, j] = final_solution["evaluation"]["overall_score"]

        # Create the visualization
        plt.figure(figsize=(max(12, len(issues) * 0.5), max(8, len(models) * 0.5)))

        # Create custom colormap
        cmap = plt.cm.get_cmap('viridis')

        # Create heatmap
        im = plt.imshow(performance_matrix, cmap=cmap, aspect='auto', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Overall Score', rotation=270, labelpad=20, fontsize=12)

        # Add labels
        plt.title("Performance Heatmap: Models vs. Issues", fontsize=16)
        plt.xlabel('Issues', fontsize=14)
        plt.ylabel('Models', fontsize=14)

        # Customize ticks
        plt.xticks(np.arange(len(issues)), issues, rotation=90, fontsize=10)
        plt.yticks(np.arange(len(models)), models, fontsize=10)

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(issues)):
                value = performance_matrix[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value < 0.5 else 'black'
                    plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)
                else:
                    plt.text(j, i, "N/A", ha="center", va="center", color='red', fontsize=9)

        plt.tight_layout()

        # Save the figure
        if output_path is None:
            output_path = self.viz_dir / "performance_heatmap.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def visualize_code_quality_scatter(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Visualize code quality vs. patch quality scatter plot.

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
                models = sorted(list(all_models))

                # Initialize data structure
                model_data = []

                # Collect data
                for result in results:
                    if "solutions" not in result or "issue_id" not in result:
                        continue

                    issue_id = result["issue_id"]

                    for model in models:
                        if model not in result["solutions"]:
                            continue

                        model_solutions = result["solutions"][model]

                        # Get the final solution
                        final_solutions = [s for s in model_solutions if s.get("iteration") == 3]
                        if not final_solutions:
                            final_solution = max(model_solutions, key=lambda s: s.get("iteration", 0))
                        else:
                            final_solution = final_solutions[0]

                        # Extract metrics
                        if "evaluation" in final_solution:
                            eval_data = final_solution["evaluation"]

                            if "code_quality" in eval_data and "patch_quality" in eval_data:
                                model_data.append({
                                    "Model": model,
                                    "Issue": issue_id,
                                    "Code Quality": eval_data["code_quality"],
                                    "Patch Quality": eval_data["patch_quality"],
                                    "Overall Score": eval_data.get("overall_score", 0)
                                })

                # Create the visualization
                plt.figure(figsize=(12, 10))

                # Create separate scatter plots for each model
                for i, model in enumerate(models):
                    model_subset = [d for d in model_data if d["Model"] == model]

                    if not model_subset:
                        continue

                    x = [d["Code Quality"] for d in model_subset]
                    y = [d["Patch Quality"] for d in model_subset]
                    sizes = [d["Overall Score"] * 100 + 20 for d in model_subset]  # Scale for visibility

                    plt.scatter(x, y, s=sizes, alpha=0.7, label=model,
                                c=[self.colors[i % len(self.colors)]])

                # Add diagonal line (code quality = patch quality)
                lims = [
                    np.min([plt.xlim()[0], plt.ylim()[0]]),
                    np.max([plt.xlim()[1], plt.ylim()[1]])
                ]
                plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

                plt.title("Code Quality vs. Patch Quality by Model", fontsize=16)
                plt.xlabel('Code Quality', fontsize=14)
                plt.ylabel('Patch Quality', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(title="Model", fontsize=12, title_fontsize=14)

                # Add annotations for quadrants
                plt.annotate("High Quality Code\nHigh Quality Patch",
                             xy=(0.75, 0.75), xycoords='axes fraction',
                             ha='center', va='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.6))

                plt.annotate("Low Quality Code\nLow Quality Patch",
                             xy=(0.25, 0.25), xycoords='axes fraction',
                             ha='center', va='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.6))

                plt.annotate("Low Quality Code\nHigh Quality Patch",
                             xy=(0.25, 0.75), xycoords='axes fraction',
                             ha='center', va='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.6))

                plt.annotate("High Quality Code\nLow Quality Patch",
                             xy=(0.75, 0.25), xycoords='axes fraction',
                             ha='center', va='center', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.6))

                plt.tight_layout()

                # Save the figure
                if output_path is None:
                    output_path = self.viz_dir / "code_quality_scatter.png"

                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

                return str(output_path)

    def visualize_reflection_effectiveness(self, results: List[Dict[str, Any]],
                                           output_path: Optional[str] = None) -> str:
        """
        Visualize the effectiveness of self-reflection across models.

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

        models = sorted(list(all_models))

        # Initialize data structures
        improvement_data = []

        # Collect data
        for result in results:
            if "solutions" not in result or "issue_id" not in result:
                continue

            issue_id = result["issue_id"]

            for model in models:
                if model not in result["solutions"]:
                    continue

                model_solutions = result["solutions"][model]

                # Need at least 2 iterations to measure improvement
                if len(model_solutions) < 2:
                    continue

                # Sort solutions by iteration
                sorted_solutions = sorted(model_solutions, key=lambda s: s.get("iteration", 0))

                # Get first and last iteration solutions
                first_solution = sorted_solutions[0]
                last_solution = sorted_solutions[-1]

                # Calculate improvement
                if ("evaluation" in first_solution and "overall_score" in first_solution["evaluation"] and
                        "evaluation" in last_solution and "overall_score" in last_solution["evaluation"]):

                    initial_score = first_solution["evaluation"]["overall_score"]
                    final_score = last_solution["evaluation"]["overall_score"]

                    if initial_score > 0:  # Avoid division by zero
                        improvement_pct = (final_score - initial_score) / initial_score * 100
                    else:
                        improvement_pct = float('inf') if final_score > 0 else 0

                    improvement_data.append({
                        "Model": model,
                        "Issue": issue_id,
                        "Initial Score": initial_score,
                        "Final Score": final_score,
                        "Absolute Improvement": final_score - initial_score,
                        "Improvement %": improvement_pct
                    })

        # Create the visualization - combination of bar chart and boxplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 1. Bar chart for average improvement by model
        model_avg_improvement = {}
        for model in models:
            model_improvements = [d["Improvement %"] for d in improvement_data if d["Model"] == model]
            if model_improvements:
                model_avg_improvement[model] = np.mean(model_improvements)
            else:
                model_avg_improvement[model] = 0

        # Sort models by improvement
        sorted_models = sorted(model_avg_improvement.keys(),
                               key=lambda m: model_avg_improvement[m],
                               reverse=True)

        # Plot average improvements
        bars = ax1.bar(sorted_models,
                       [model_avg_improvement[m] for m in sorted_models],
                       color=[self.colors[models.index(m) % len(self.colors)] for m in sorted_models])

        ax1.set_title("Average Improvement % by Model", fontsize=14)
        ax1.set_xlabel("Model", fontsize=12)
        ax1.set_ylabel("Average Improvement %", fontsize=12)
        ax1.set_xticklabels(sorted_models, rotation=45, ha="right")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            text_color = 'black'
            if height < 0:
                text_color = 'red'
            ax1.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3 if height >= 0 else -15),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         color=text_color,
                         fontsize=10)

        # 2. Box plot for improvement distribution
        improvement_by_model = {model: [] for model in models}
        for d in improvement_data:
            improvement_by_model[d["Model"]].append(d["Improvement %"])

        # Prepare data for boxplot
        bp_data = [improvement_by_model[model] for model in models]

        # Create box plot
        box_colors = [self.colors[i % len(self.colors)] for i in range(len(models))]
        box_parts = ax2.boxplot(bp_data, patch_artist=True, labels=models)

        # Customize box colors
        for patch, color in zip(box_parts['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add individual points
        for i, model in enumerate(models):
            if improvement_by_model[model]:
                # Add jitter
                x = np.random.normal(i + 1, 0.05, size=len(improvement_by_model[model]))
                ax2.scatter(x, improvement_by_model[model], alpha=0.7,
                            s=30, color=box_colors[i])

        ax2.set_title("Improvement % Distribution by Model", fontsize=14)
        ax2.set_xlabel("Model", fontsize=12)
        ax2.set_ylabel("Improvement %", fontsize=12)
        ax2.set_xticklabels(models, rotation=45, ha="right")
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # Add horizontal line at y=0
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)

        plt.tight_layout()

        # Save the figure
        if output_path is None:
            output_path = self.viz_dir / "reflection_effectiveness.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)
