# statistics/statistical_tests.py
import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class StatisticalTests:
    """
    Statistical tests for model comparison.
    """

    @staticmethod
    def paired_ttest(model_a_scores: List[float], model_b_scores: List[float], model_a_name: str, model_b_name: str) -> \
    Dict[str, Any]:
        """
        Perform a paired t-test between two models.

        Args:
            model_a_scores: List of scores for model A.
            model_b_scores: List of scores for model B.
            model_a_name: Name of model A.
            model_b_name: Name of model B.

        Returns:
            Dictionary with test results.
        """
        if len(model_a_scores) != len(model_b_scores):
            raise ValueError("Score lists must have the same length for paired t-test")

        if len(model_a_scores) < 2:
            return {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "t_statistic": None,
                "p_value": None,
                "mean_difference": None,
                "significant": None,
                "better_model": None,
                "error": "Not enough samples"
            }

        try:
            t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)

            mean_diff = np.mean(model_a_scores) - np.mean(model_b_scores)
            significant = p_value < 0.05
            better_model = model_a_name if mean_diff > 0 else model_b_name if mean_diff < 0 else "Equal"

            return {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "t_statistic": t_stat,
                "p_value": p_value,
                "mean_difference": mean_diff,
                "significant": significant,
                "better_model": better_model if significant else "No significant difference"
            }
        except Exception as e:
            logger.error(f"Error in paired t-test: {str(e)}")
            return {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "error": str(e)
            }

    @staticmethod
    def anova_test(model_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform ANOVA test across multiple models.

        Args:
            model_scores: Dictionary mapping model names to score lists.

        Returns:
            Dictionary with ANOVA results.
        """
        # Convert to format needed for scipy's f_oneway
        groups = list(model_scores.values())
        model_names = list(model_scores.keys())

        if len(groups) < 2:
            return {
                "f_statistic": None,
                "p_value": None,
                "significant": None,
                "error": "Need at least two models for ANOVA"
            }

        try:
            f_stat, p_value = stats.f_oneway(*groups)

            return {
                "models": model_names,
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        except Exception as e:
            logger.error(f"Error in ANOVA test: {str(e)}")
            return {
                "models": model_names,
                "error": str(e)
            }

    @staticmethod
    def tukey_hsd(model_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform Tukey's HSD post-hoc test following ANOVA.

        Args:
            model_scores: Dictionary mapping model names to score lists.

        Returns:
            Dictionary with Tukey HSD results.
        """
        try:
            # Create a DataFrame in long format for statsmodels
            data = []
            for model_name, scores in model_scores.items():
                for score in scores:
                    data.append({"model": model_name, "score": score})

            df = pd.DataFrame(data)

            # Import here to avoid making statsmodels a main dependency
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            # Perform the Tukey HSD test
            tukey_results = pairwise_tukeyhsd(df["score"], df["model"], alpha=0.05)

            # Convert to a more usable format
            result_list = []
            for i, (group1, group2, reject, _, _, _) in enumerate(zip(
                    tukey_results.groupsunique[tukey_results.data[0]],
                    tukey_results.groupsunique[tukey_results.data[1]],
                    tukey_results.reject,
                    tukey_results.meandiffs,
                    tukey_results.confint[:, 0],
                    tukey_results.confint[:, 1]
            )):
                result_list.append({
                    "model_a": group1,
                    "model_b": group2,
                    "significant_difference": bool(reject),
                    "mean_difference": float(tukey_results.meandiffs[i]),
                    "conf_interval_lower": float(tukey_results.confint[i, 0]),
                    "conf_interval_upper": float(tukey_results.confint[i, 1])
                })

            return {
                "pairwise_comparisons": result_list
            }
        except Exception as e:
            logger.error(f"Error in Tukey HSD test: {str(e)}")
            return {
                "error": str(e)
            }

    @staticmethod
    def wilcoxon_test(model_a_scores: List[float], model_b_scores: List[float], model_a_name: str, model_b_name: str) -> \
    Dict[str, Any]:
        """
        Perform a Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            model_a_scores: List of scores for model A.
            model_b_scores: List of scores for model B.
            model_a_name: Name of model A.
            model_b_name: Name of model B.

        Returns:
            Dictionary with test results.
        """
        if len(model_a_scores) != len(model_b_scores):
            raise ValueError("Score lists must have the same length for Wilcoxon test")

        if len(model_a_scores) < 2:
            return {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "statistic": None,
                "p_value": None,
                "significant": None,
                "better_model": None,
                "error": "Not enough samples"
            }

        try:
            statistic, p_value = stats.wilcoxon(model_a_scores, model_b_scores)

            # Check which model performs better
            mean_diff = np.mean(model_a_scores) - np.mean(model_b_scores)
            significant = p_value < 0.05
            better_model = model_a_name if mean_diff > 0 else model_b_name if mean_diff < 0 else "Equal"

            return {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "statistic": statistic,
                "p_value": p_value,
                "mean_difference": mean_diff,
                "significant": significant,
                "better_model": better_model if significant else "No significant difference"
            }
        except Exception as e:
            logger.error(f"Error in Wilcoxon test: {str(e)}")
            return {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "error": str(e)
            }







