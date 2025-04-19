# statistics/correlation_analysis.py
import numpy as np
from scipy import stats
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class CorrelationAnalysis:
    """
    Correlation analysis for model performance.
    """

    @staticmethod
    def pearson_correlation(x: List[float], y: List[float], x_name: str, y_name: str) -> Dict[str, Any]:
        """
        Calculate Pearson correlation between two variables.

        Args:
            x: First variable values.
            y: Second variable values.
            x_name: Name of first variable.
            y_name: Name of second variable.

        Returns:
            Dictionary with correlation results.
        """
        if len(x) != len(y):
            raise ValueError("Variable lists must have the same length for correlation")

        if len(x) < 2:
            return {
                "x_variable": x_name,
                "y_variable": y_name,
                "correlation": None,
                "p_value": None,
                "significant": None,
                "strength": None,
                "error": "Not enough samples"
            }

        try:
            corr, p_value = stats.pearsonr(x, y)

            # Interpret the correlation strength
            strength = "None"
            if abs(corr) < 0.1:
                strength = "Negligible"
            elif abs(corr) < 0.3:
                strength = "Weak"
            elif abs(corr) < 0.5:
                strength = "Moderate"
            elif abs(corr) < 0.7:
                strength = "Strong"
            else:
                strength = "Very Strong"

            return {
                "x_variable": x_name,
                "y_variable": y_name,
                "correlation": corr,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "strength": strength,
                "direction": "Positive" if corr > 0 else "Negative" if corr < 0 else "None"
            }
        except Exception as e:
            logger.error(f"Error in Pearson correlation: {str(e)}")
            return {
                "x_variable": x_name,
                "y_variable": y_name,
                "error": str(e)
            }

    @staticmethod
    def spearman_correlation(x: List[float], y: List[float], x_name: str, y_name: str) -> Dict[str, Any]:
        """
        Calculate Spearman rank correlation between two variables.

        Args:
            x: First variable values.
            y: Second variable values.
            x_name: Name of first variable.
            y_name: Name of second variable.

        Returns:
            Dictionary with correlation results.
        """
        if len(x) != len(y):
            raise ValueError("Variable lists must have the same length for correlation")

        if len(x) < 2:
            return {
                "x_variable": x_name,
                "y_variable": y_name,
                "correlation": None,
                "p_value": None,
                "significant": None,
                "strength": None,
                "error": "Not enough samples"
            }

        try:
            corr, p_value = stats.spearmanr(x, y)

            # Interpret the correlation strength
            strength = "None"
            if abs(corr) < 0.1:
                strength = "Negligible"
            elif abs(corr) < 0.3:
                strength = "Weak"
            elif abs(corr) < 0.5:
                strength = "Moderate"
            elif abs(corr) < 0.7:
                strength = "Strong"
            else:
                strength = "Very Strong"

            return {
                "x_variable": x_name,
                "y_variable": y_name,
                "correlation": corr,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "strength": strength,
                "direction": "Positive" if corr > 0 else "Negative" if corr < 0 else "None"
            }
        except Exception as e:
            logger.error(f"Error in Spearman correlation: {str(e)}")
            return {
                "x_variable": x_name,
                "y_variable": y_name,
                "error": str(e)
            }

    @staticmethod
    def analyze_reflection_effectiveness(iterations: List[int], scores: List[float]) -> Dict[str, Any]:
        """
        Analyze the effectiveness of self-reflection iterations.

        Args:
            iterations: List of iteration numbers.
            scores: List of corresponding scores.

        Returns:
            Dictionary with analysis results.
        """
        if len(iterations) != len(scores):
            raise ValueError("Iteration and score lists must have the same length")

        if len(iterations) < 2:
            return {
                "correlation": None,
                "improvement": None,
                "error": "Not enough samples"
            }

        try:
            # Calculate correlation
            corr_results = CorrelationAnalysis.pearson_correlation(
                iterations, scores, "Iterations", "Scores"
            )

            # Calculate improvement from first to last iteration
            improvement = scores[-1] - scores[0]
            improvement_percentage = (improvement / scores[0]) * 100 if scores[0] != 0 else float('inf')

            return {
                "correlation_analysis": corr_results,
                "improvement": improvement,
                "improvement_percentage": improvement_percentage,
                "initial_score": scores[0],
                "final_score": scores[-1],
                "effective": improvement > 0
            }
        except Exception as e:
            logger.error(f"Error in reflection effectiveness analysis: {str(e)}")
            return {
                "error": str(e)
            }

    @staticmethod
    def analyze_issue_complexity_impact(complexity_scores: List[float], performance_scores: List[float]) -> Dict[
        str, Any]:
        """
        Analyze the impact of issue complexity on model performance.

        Args:
            complexity_scores: List of issue complexity scores.
            performance_scores: List of corresponding performance scores.

        Returns:
            Dictionary with analysis results.
        """
        if len(complexity_scores) != len(performance_scores):
            raise ValueError("Complexity and performance lists must have the same length")

        if len(complexity_scores) < 2:
            return {
                "correlation": None,
                "error": "Not enough samples"
            }

        try:
            # Calculate correlation
            corr_results = CorrelationAnalysis.spearman_correlation(
                complexity_scores, performance_scores, "Issue Complexity", "Model Performance"
            )

            # Group issues by complexity
            complexity_array = np.array(complexity_scores)
            performance_array = np.array(performance_scores)

            # Define complexity bins
            q1 = np.percentile(complexity_array, 25)
            q2 = np.percentile(complexity_array, 50)
            q3 = np.percentile(complexity_array, 75)

            # Get performance by complexity group
            low_complexity = performance_array[complexity_array <= q1]
            medium_low_complexity = performance_array[(complexity_array > q1) & (complexity_array <= q2)]
            medium_high_complexity = performance_array[(complexity_array > q2) & (complexity_array <= q3)]
            high_complexity = performance_array[complexity_array > q3]

            return {
                "correlation_analysis": corr_results,
                "performance_by_complexity": {
                    "low": np.mean(low_complexity) if len(low_complexity) > 0 else None,
                    "medium_low": np.mean(medium_low_complexity) if len(medium_low_complexity) > 0 else None,
                    "medium_high": np.mean(medium_high_complexity) if len(medium_high_complexity) > 0 else None,
                    "high": np.mean(high_complexity) if len(high_complexity) > 0 else None
                }
            }
        except Exception as e:
            logger.error(f"Error in issue complexity impact analysis: {str(e)}")
            return {
                "error": str(e)
            }
