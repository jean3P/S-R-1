# statistics/benchmark_report.py
import os
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from .statistical_tests import StatisticalTests
from .correlation_analysis import CorrelationAnalysis

logger = logging.getLogger(__name__)


class BenchmarkReport:
    """
    Generate comprehensive benchmark reports.
    """

    def __init__(self, config, results_data):
        """
        Initialize benchmark report generator.

        Args:
            config: Configuration object.
            results_data: Results data from experiments.
        """
        self.config = config
        self.results_data = results_data
        self.report_dir = Path(config["evaluation"]["results_dir"]) / "reports"

        # Create report directory if it doesn't exist
        if not self.report_dir.exists():
            self.report_dir.mkdir(parents=True)

    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate a full benchmark report.

        Returns:
            Dictionary containing the full report.
        """
        logger.info("Generating full benchmark report")

        # Extract models from the results
        models = self._extract_models()

        # Generate individual reports
        model_performance = self._analyze_model_performance(models)
        comparative_analysis = self._perform_comparative_analysis(models)
        reasoning_analysis = self._analyze_reasoning_methods()
        reflection_analysis = self._analyze_reflection_effectiveness()
        issue_complexity_analysis = self._analyze_issue_complexity()

        # Compile the full report
        report = {
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "num_issues": len(self.results_data),
            "model_performance": model_performance,
            "comparative_analysis": comparative_analysis,
            "reasoning_analysis": reasoning_analysis,
            "reflection_analysis": reflection_analysis,
            "issue_complexity_analysis": issue_complexity_analysis,
            "summary": self._generate_summary(model_performance, comparative_analysis)
        }

        # Save the report
        report_path = self.report_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Benchmark report saved to {report_path}")

        return report

    def generate_model_report(self, model_name: str) -> Dict[str, Any]:
        """
        Generate a report for a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary containing the model report.
        """
        logger.info(f"Generating report for model: {model_name}")

        # Extract data for the specified model
        model_data = self._extract_model_data(model_name)

        if not model_data:
            return {"error": f"No data found for model {model_name}"}

        # Analyze model performance
        performance = self._analyze_single_model_performance(model_name, model_data)

        # Analyze reflection effectiveness for this model
        reflection = self._analyze_model_reflection(model_name, model_data)

        # Analyze issue performance
        issue_analysis = self._analyze_model_by_issue(model_name, model_data)

        # Compile the model report
        report = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "num_issues": len(model_data),
            "performance": performance,
            "reflection_analysis": reflection,
            "issue_analysis": issue_analysis,
            "summary": self._generate_model_summary(model_name, performance, reflection)
        }

        # Save the report
        report_path = self.report_dir / f"{model_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Model report saved to {report_path}")

        return report

    def _extract_models(self) -> List[str]:
        """Extract the list of model names from the results."""
        models = set()

        for result in self.results_data:
            if "solutions" in result:
                models.update(result["solutions"].keys())

        return sorted(list(models))

    def _extract_model_data(self, model_name: str) -> List[Dict[str, Any]]:
        """Extract data for a specific model from the results."""
        model_data = []

        for result in self.results_data:
            if "solutions" in result and model_name in result["solutions"]:
                issue_id = result.get("issue_id", "unknown")
                solutions = result["solutions"][model_name]

                model_data.append({
                    "issue_id": issue_id,
                    "solutions": solutions
                })

        return model_data

    def _analyze_model_performance(self, models: List[str]) -> Dict[str, Any]:
        """Analyze the performance of all models."""
        performance = {}

        for model in models:
            model_data = self._extract_model_data(model)
            performance[model] = self._analyze_single_model_performance(model, model_data)

        return performance

    def _analyze_single_model_performance(self, model_name: str, model_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the performance of a single model."""
        if not model_data:
            return {"error": "No data available"}

        # Collect metrics across all issues
        overall_scores = []
        success_rates = []
        code_quality_scores = []
        patch_quality_scores = []
        execution_times = []

        for issue in model_data:
            solutions = issue.get("solutions", [])

            if not solutions:
                continue

            # Get the final solution (after all iterations)
            final_solutions = [s for s in solutions if s.get("iteration") == 3]
            if not final_solutions:
                final_solution = max(solutions, key=lambda s: s.get("iteration", 0))
            else:
                final_solution = final_solutions[0]

            # Extract metrics
            if "evaluation" in final_solution:
                eval_data = final_solution["evaluation"]
                overall_scores.append(eval_data.get("overall_score", 0))
                success_rates.append(eval_data.get("success_rate", 0))
                code_quality_scores.append(eval_data.get("code_quality", 0))
                patch_quality_scores.append(eval_data.get("patch_quality", 0))

            # Extract execution time
            execution_times.append(final_solution.get("execution_time", 0))

        # Calculate statistics
        return {
            "num_issues_solved": len(overall_scores),
            "overall_score": {
                "mean": np.mean(overall_scores) if overall_scores else None,
                "median": np.median(overall_scores) if overall_scores else None,
                "std": np.std(overall_scores) if overall_scores else None,
                "min": np.min(overall_scores) if overall_scores else None,
                "max": np.max(overall_scores) if overall_scores else None
            },
            "success_rate": {
                "mean": np.mean(success_rates) if success_rates else None,
                "median": np.median(success_rates) if success_rates else None,
                "std": np.std(success_rates) if success_rates else None
            },
            "code_quality": {
                "mean": np.mean(code_quality_scores) if code_quality_scores else None,
                "median": np.median(code_quality_scores) if code_quality_scores else None,
                "std": np.std(code_quality_scores) if code_quality_scores else None
            },
            "patch_quality": {
                "mean": np.mean(patch_quality_scores) if patch_quality_scores else None,
                "median": np.median(patch_quality_scores) if patch_quality_scores else None,
                "std": np.std(patch_quality_scores) if patch_quality_scores else None
            },
            "execution_time": {
                "mean": np.mean(execution_times) if execution_times else None,
                "median": np.median(execution_times) if execution_times else None,
                "std": np.std(execution_times) if execution_times else None,
                "total": np.sum(execution_times) if execution_times else None
            }
        }

    def _perform_comparative_analysis(self, models: List[str]) -> Dict[str, Any]:
        """Perform comparative analysis between models."""
        if len(models) < 2:
            return {"error": "Need at least two models for comparative analysis"}

        # Get overall scores for each model
        model_scores = {}

        for model in models:
            model_scores[model] = []

            for result in self.results_data:
                if "solutions" not in result or model not in result["solutions"]:
                    continue

                solutions = result["solutions"][model]

                # Get the final solution
                final_solutions = [s for s in solutions if s.get("iteration") == 3]
                if not final_solutions:
                    final_solution = max(solutions, key=lambda s: s.get("iteration", 0))
                else:
                    final_solution = final_solutions[0]

                # Extract overall score
                if "evaluation" in final_solution:
                    overall_score = final_solution["evaluation"].get("overall_score", 0)
                    model_scores[model].append(overall_score)

        # Perform ANOVA test
        anova_results = StatisticalTests.anova_test(model_scores)

        # Perform Tukey HSD test if ANOVA is significant
        tukey_results = None
        if anova_results.get("significant", False):
            tukey_results = StatisticalTests.tukey_hsd(model_scores)

        # Perform pairwise t-tests
        pairwise_results = []

        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                # Find issues solved by both models
                common_issues = []

                for a_score, b_score in zip(model_scores[model_a], model_scores[model_b]):
                    common_issues.append((a_score, b_score))

                if common_issues:
                    a_scores = [a for a, _ in common_issues]
                    b_scores = [b for _, b in common_issues]

                    # Perform t-test
                    t_test = StatisticalTests.paired_ttest(a_scores, b_scores, model_a, model_b)

                    # Perform Wilcoxon test
                    wilcoxon_test = StatisticalTests.wilcoxon_test(a_scores, b_scores, model_a, model_b)

                    pairwise_results.append({
                        "model_a": model_a,
                        "model_b": model_b,
                        "num_common_issues": len(common_issues),
                        "t_test": t_test,
                        "wilcoxon_test": wilcoxon_test
                    })

        return {
            "anova": anova_results,
            "tukey_hsd": tukey_results,
            "pairwise_tests": pairwise_results
        }

    def _analyze_reasoning_methods(self) -> Dict[str, Any]:
        """Analyze the performance of different reasoning methods."""
        # This would require results to include the reasoning method used
        # For now, we'll just return a placeholder
        return {
            "note": "Reasoning method analysis requires the reasoning method to be recorded in the results"
        }

    def _analyze_reflection_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of self-reflection across models."""
        reflection_results = {}

        for model in self._extract_models():
            reflection_results[model] = self._analyze_model_reflection(model)

        return reflection_results

    def _analyze_model_reflection(self, model_name: str, model_data: Optional[List[Dict[str, Any]]] = None) -> Dict[
        str, Any]:
        """Analyze the effectiveness of self-reflection for a specific model."""
        if model_data is None:
            model_data = self._extract_model_data(model_name)

        if not model_data:
            return {"error": "No data available"}

        # Collect metrics across all issues
        improvement_metrics = []

        for issue in model_data:
            solutions = issue.get("solutions", [])

            if len(solutions) < 2:
                continue

            # Sort solutions by iteration
            sorted_solutions = sorted(solutions, key=lambda s: s.get("iteration", 0))

            # Extract overall scores for each iteration
            iterations = []
            scores = []

            for solution in sorted_solutions:
                if "evaluation" in solution:
                    iterations.append(solution.get("iteration", 0))
                    scores.append(solution["evaluation"].get("overall_score", 0))

            if len(iterations) >= 2:
                # Analyze reflection effectiveness
                effectiveness = CorrelationAnalysis.analyze_reflection_effectiveness(iterations, scores)
                improvement_metrics.append(effectiveness)

        # Calculate aggregate statistics
        improvements = [m.get("improvement", 0) for m in improvement_metrics if "improvement" in m]
        improvement_percentages = [m.get("improvement_percentage", 0) for m in improvement_metrics if
                                   "improvement_percentage" in m]

        return {
            "num_issues_analyzed": len(improvement_metrics),
            "improvements": {
                "mean": np.mean(improvements) if improvements else None,
                "median": np.median(improvements) if improvements else None,
                "std": np.std(improvements) if improvements else None,
                "positive_improvements": sum(1 for imp in improvements if imp > 0) if improvements else 0,
                "negative_improvements": sum(1 for imp in improvements if imp < 0) if improvements else 0,
                "percentage_positive": (sum(1 for imp in improvements if imp > 0) / len(
                    improvements) * 100) if improvements else None
            },
            "improvement_percentages": {
                "mean": np.mean(improvement_percentages) if improvement_percentages else None,
                "median": np.median(improvement_percentages) if improvement_percentages else None,
                "std": np.std(improvement_percentages) if improvement_percentages else None
            }
        }

    def _analyze_issue_complexity(self) -> Dict[str, Any]:
        """Analyze the relationship between issue complexity and model performance."""
        # Calculate complexity metrics for each issue
        issue_metrics = []

        for result in self.results_data:
            if "issue_id" not in result or "solutions" not in result:
                continue

            issue_id = result["issue_id"]

            # Calculate complexity (e.g., based on diff size)
            # This is a placeholder and should be replaced with actual complexity calculation
            complexity = 1.0

            # Calculate average performance across models
            model_performances = []
            for model, solutions in result["solutions"].items():
                final_solutions = [s for s in solutions if s.get("iteration") == 3]
                if not final_solutions:
                    final_solution = max(solutions, key=lambda s: s.get("iteration", 0))
                else:
                    final_solution = final_solutions[0]

                if "evaluation" in final_solution:
                    overall_score = final_solution["evaluation"].get("overall_score", 0)
                    model_performances.append(overall_score)

            avg_performance = np.mean(model_performances) if model_performances else 0

            issue_metrics.append({
                "issue_id": issue_id,
                "complexity": complexity,
                "avg_performance": avg_performance
            })

        # Analyze the relationship between complexity and performance
        complexity_scores = [m["complexity"] for m in issue_metrics]
        performance_scores = [m["avg_performance"] for m in issue_metrics]

        complexity_analysis = CorrelationAnalysis.analyze_issue_complexity_impact(
            complexity_scores, performance_scores
        )

        return complexity_analysis

    def _analyze_model_by_issue(self, model_name: str, model_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model performance by issue."""
        issue_results = []

        for issue in model_data:
            issue_id = issue.get("issue_id", "unknown")
            solutions = issue.get("solutions", [])

            if not solutions:
                continue

            # Get the final solution
            final_solutions = [s for s in solutions if s.get("iteration") == 3]
            if not final_solutions:
                final_solution = max(solutions, key=lambda s: s.get("iteration", 0))
            else:
                final_solution = final_solutions[0]

            # Extract evaluation metrics
            if "evaluation" in final_solution:
                eval_data = final_solution["evaluation"]

                issue_results.append({
                    "issue_id": issue_id,
                    "overall_score": eval_data.get("overall_score", 0),
                    "success_rate": eval_data.get("success_rate", 0),
                    "code_quality": eval_data.get("code_quality", 0),
                    "patch_quality": eval_data.get("patch_quality", 0),
                    "execution_time": final_solution.get("execution_time", 0)
                })

        # Sort issues by overall score
        sorted_issues = sorted(issue_results, key=lambda x: x.get("overall_score", 0), reverse=True)

        return {
            "best_issues": sorted_issues[:5] if len(sorted_issues) >= 5 else sorted_issues,
            "worst_issues": sorted_issues[-5:] if len(sorted_issues) >= 5 else sorted_issues[::-1],
            "all_issues": sorted_issues
        }

    def _generate_summary(self, model_performance: Dict[str, Any], comparative_analysis: Dict[str, Any]) -> Dict[
        str, Any]:
        """Generate a summary of the benchmark results."""
        # Determine the best model based on overall score
        best_model = None
        best_score = -float('inf')

        for model, perf in model_performance.items():
            if "overall_score" in perf and perf["overall_score"].get("mean") is not None:
                score = perf["overall_score"]["mean"]
                if score > best_score:
                    best_score = score
                    best_model = model

        # Get significant differences from comparative analysis
        significant_differences = []

        if "pairwise_tests" in comparative_analysis:
            for test in comparative_analysis["pairwise_tests"]:
                if "t_test" in test and test["t_test"].get("significant"):
                    significant_differences.append({
                        "model_a": test["model_a"],
                        "model_b": test["model_b"],
                        "better_model": test["t_test"].get("better_model"),
                        "p_value": test["t_test"].get("p_value")
                    })

        return {
            "best_model": best_model,
            "best_model_score": best_score if best_model else None,
            "significant_differences": significant_differences,
            "num_significant_differences": len(significant_differences)
        }

    def _generate_model_summary(self, model_name: str, performance: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[
        str, Any]:
        """Generate a summary for a specific model."""
        return {
            "model_name": model_name,
            "overall_score": performance.get("overall_score", {}).get("mean"),
            "success_rate": performance.get("success_rate", {}).get("mean"),
            "reflection_improvement": reflection.get("improvements", {}).get("mean"),
            "reflection_effectiveness": reflection.get("improvements", {}).get("percentage_positive")
        }
