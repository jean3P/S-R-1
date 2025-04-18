# src/evaluators/base_evaluator.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseEvaluator(ABC):
    """Abstract base class for all code evaluators."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.

        Args:
            config: Evaluator configuration
        """
        from src.utils.logging import get_logger

        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Initialize evaluator-specific metrics
        self.metrics = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_execution_time": 0.0
        }

    @abstractmethod
    def evaluate(self, code: str) -> Tuple[str, str]:
        """
        Evaluate the code.

        Args:
            code: Code to evaluate

        Returns:
            Tuple of (output, errors)
        """
        pass

    def run_test_cases(self, code: str) -> Dict[str, Any]:
        """
        Run test cases on the code.

        Args:
            code: Code to test

        Returns:
            Test results
        """
        test_cases = self.config.get("test_cases", [])
        if not test_cases:
            return {"tested": False, "message": "No test cases defined"}

        # This is a default implementation
        # Subclasses should override this method for proper testing
        self.logger.info(f"Running {len(test_cases)} test cases")

        results = {
            "tested": True,
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }

        for test_case in test_cases:
            test_result = {
                "input": test_case.get("input"),
                "expected": test_case.get("expected"),
                "passed": False,
                "actual": None,
                "error": None
            }

            # Subclasses should implement the actual testing logic
            results["details"].append(test_result)

        return results

    def _record_evaluation(self, success: bool, execution_time: float) -> None:
        """
        Record metrics for an evaluation.

        Args:
            success: Whether the evaluation was successful
            execution_time: Time taken for execution in seconds
        """
        self.metrics["total_evaluations"] += 1

        if success:
            self.metrics["successful_evaluations"] += 1
        else:
            self.metrics["failed_evaluations"] += 1

        # Update average execution time
        if self.metrics["total_evaluations"] > 1:
            prev_avg = self.metrics["average_execution_time"]
            prev_count = self.metrics["total_evaluations"] - 1

            # Calculate new average
            self.metrics["average_execution_time"] = (
                                                             (prev_avg * prev_count) + execution_time
                                                     ) / self.metrics["total_evaluations"]
        else:
            self.metrics["average_execution_time"] = execution_time

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current evaluator metrics.

        Returns:
            Dictionary of evaluator metrics
        """
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset the evaluator metrics."""
        self.metrics = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_execution_time": 0.0
        }
