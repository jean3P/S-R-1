import time
from typing import Dict, Any, Tuple, List
from src.agents.base_agent import BaseAgent
from src.models.registry import get_model
from src.prompts.registry import get_prompt
from src.evaluators.registry import get_evaluator
from src.utils.logging import get_logger


class TreeOfThoughtDiagnosticAgent(BaseAgent):
    """
    Diagnostic Agent that uses a 'Tree of Thought' approach to investigate
    issues in code—focusing on matrix dimensionality, operator applications,
    and data structure representations.
    """

    def __init__(self, model_id: str, prompt_id: str, evaluator_id: str, config: Dict[str, Any]):
        """
        Initialize the TreeOfThoughtDiagnosticAgent.

        Args:
            model_id: ID of the model to use
            prompt_id: ID of the prompt to use
            evaluator_id: ID of the evaluator to use
            config: Configuration dictionary
        """
        super().__init__(model_id, prompt_id, evaluator_id, config)
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("TreeOfThoughtDiagnosticAgent initialized.")

        # Additional config or initialization if needed
        self.depth_limit = config.get("depth_limit", 3)
        self.debug_mode = config.get("debug_mode", True)

    def reflect(self, initial_prompt: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a 'tree of thought' diagnostic process.

        This method will iteratively:
          1. Generate high-level hypotheses about the potential bug (e.g., matrix dimension mismatch).
          2. For each hypothesis, gather deeper code context or logs.
          3. Narrow down the root cause based on repeated analysis or hints from an evaluator.
          4. Provide a recommended fix.

        Args:
            initial_prompt: Description of the bug or problem
            task: Additional task details (e.g. references to relevant code or logs)

        Returns:
            A dictionary summarizing the final root cause and recommended fix.
        """
        self.logger.info(f"Starting Tree-of-Thought diagnostic for bug:\n{initial_prompt}")
        self._start_metrics()

        # 1. Outline high-level hypotheses
        #    For demonstration, we’ll just hardcode some “typical” categories of mistakes
        #    that might occur in a “separability matrix” scenario:
        hypotheses = [
            "Operator application mismatch (multiplying the wrong dimension)",
            "Matrix dimension mismatch (shape error in code path)",
            "Data structure representation conflict (e.g., row-major vs col-major order)",
        ]

        # 2. Conduct deeper analyses on each hypothesis
        #    In a real scenario, you'd gather code from file_summaries or logs from the evaluator
        deeper_analyses = []
        for i, hypothesis in enumerate(hypotheses, start=1):
            analysis = self._deep_analysis_of_hypothesis(hypothesis, i)
            deeper_analyses.append(analysis)

            # Early exit if we find a conclusive cause or if depth exceeded
            if analysis["conclusive"]:
                break
            if i >= self.depth_limit:
                break

        # 3. Identify root cause from analyses
        root_cause = None
        for analysis in deeper_analyses:
            if analysis["conclusive"]:
                root_cause = analysis["root_cause"]
                break

        # If no single conclusive cause emerged, pick the best guess
        if not root_cause:
            root_cause = "Matrix dimension mismatch: shape mismatch in the operator call"

        # 4. Provide recommended fix
        recommended_fix = "Adjust the matrix dimensions to match the operator’s shape requirements (e.g., ensure row vs. column alignment)."

        # 5. Summarize final results
        self._end_metrics()

        # Example results dictionary
        results = {
            "task": task,
            "agent": self.__class__.__name__,
            "initial_prompt": initial_prompt,
            "root_cause": root_cause,
            "recommended_fix": recommended_fix,
            "analysis": deeper_analyses,
            "metrics": self.metrics,
        }

        self.logger.info(f"Diagnostic completed. Root cause: {root_cause} | Fix: {recommended_fix}")
        return results

    def _deep_analysis_of_hypothesis(self, hypothesis: str, level: int) -> Dict[str, Any]:
        """
        Analyze a single hypothesis in more detail.

        In a real scenario, you might:
        - Inspect code references or logs.
        - Call the evaluator to test a small patch or gather error details.
        - Use the model to refine the hypothesis.

        Returns:
            A dictionary with whether the hypothesis is conclusive and the identified root cause.
        """
        # [Pseudo-code for demonstration]
        self.logger.debug(f"[Level {level} Analysis] Checking hypothesis: {hypothesis}")

        # Here we pretend we discovered that 'Matrix dimension mismatch' is indeed the culprit
        # after seeing an error “ValueError: shapes (32,64) and (32,64) not aligned...”
        conclusive = False
        root_cause = None

        # If hypothesis matches typical dimension mismatch patterns, we conclude
        if "dimension mismatch" in hypothesis.lower():
            conclusive = True
            root_cause = (
                "Dimension mismatch in operator application. The code tries to multiply a (m x n) matrix with a (m x "
                "p) matrix, but the shapes do not align for typical matrix multiplication."
            )

        return {
            "hypothesis": hypothesis,
            "conclusive": conclusive,
            "root_cause": root_cause,
        }

