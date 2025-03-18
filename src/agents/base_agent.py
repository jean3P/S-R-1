# src/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import time
from datetime import datetime
from src.models.registry import get_model
from src.prompts.registry import get_prompt
from src.evaluators.registry import get_evaluator
from src.utils.logging import get_logger


class BaseAgent(ABC):
    """Abstract base class for all self-reflection agents."""

    def __init__(self, model_id: str, prompt_id: str, evaluator_id: str, config: Dict[str, Any]):
        """
        Initialize the agent.

        Args:
            model_id: ID of the model to use
            prompt_id: ID of the prompt template to use
            evaluator_id: ID of the evaluator to use
            config: Agent configuration
        """

        self.model = get_model(model_id)
        self.prompt = get_prompt(prompt_id)
        self.evaluator = get_evaluator(evaluator_id)
        self.config = config
        self.max_iterations = config.get("max_iterations", 3)
        self.logger = get_logger(self.__class__.__name__)

        # Initialize agent-specific metrics
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "total_iterations": 0,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "total_tokens_used": 0,
            "average_generation_time": 0
        }

    @abstractmethod
    def reflect(self, initial_prompt: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the self-reflection loop.

        Args:
            initial_prompt: Initial prompt to the model
            task: Task details

        Returns:
            Dict containing the results of the self-reflection process
        """
        pass

    def _start_metrics(self) -> None:
        """Initialize metrics for a new reflection session."""
        self.metrics["start_time"] = datetime.now().isoformat()
        self.metrics["total_iterations"] = 0
        self.metrics["successful_iterations"] = 0
        self.metrics["failed_iterations"] = 0
        self.metrics["total_tokens_used"] = 0
        self.metrics["generation_times"] = []

    def _end_metrics(self) -> None:
        """Finalize metrics for the reflection session."""
        self.metrics["end_time"] = datetime.now().isoformat()

        # Calculate average generation time
        if self.metrics.get("generation_times"):
            total_time = sum(self.metrics["generation_times"])
            count = len(self.metrics["generation_times"])
            self.metrics["average_generation_time"] = total_time / count

            # Remove detailed generation times from final metrics
            del self.metrics["generation_times"]

    def _record_generation(self, tokens_used: int, generation_time: float, success: bool) -> None:
        """
        Record metrics for a generation step.

        Args:
            tokens_used: Number of tokens used
            generation_time: Time taken for generation in seconds
            success: Whether the generation was successful
        """
        self.metrics["total_iterations"] += 1
        self.metrics["total_tokens_used"] += tokens_used

        if not self.metrics.get("generation_times"):
            self.metrics["generation_times"] = []

        self.metrics["generation_times"].append(generation_time)

        if success:
            self.metrics["successful_iterations"] += 1
        else:
            self.metrics["failed_iterations"] += 1

    def _measure_execution(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure the execution time of a function.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tuple of (result, execution_time)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        return result, execution_time
    