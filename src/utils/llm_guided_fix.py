# src/solution/llm_guided_fix.py

import logging
import re
from typing import Dict, List, Any
from ..utils.llm_guidance import LLMCodeLocationGuidance
from ..models import create_model
from ..utils.patch_validator import PatchValidator

logger = logging.getLogger(__name__)


class LLMGuidedFix:
    """
    Uses the LLM Code Location Guidance Framework to guide models to fix issues.
    """

    def __init__(self, config, model_name=None):
        """
        Initialize the LLM guided fix processor.

        Args:
            config: Configuration object.
            model_name: Name of the model to use, or None to use default.
        """
        self.config = config
        self.model_name = model_name or config.get("default_model", "deepseek-r1-distill")
        self.model = create_model(self.model_name, config)
        self.guidance = LLMCodeLocationGuidance(config)
        self.patch_validator = PatchValidator(config)

    def fix_issue(self, issue: Dict[str, Any],
                  suspected_files: List[str] = None,
                  error_output: str = None,
                  max_iterations: int = 3) -> Dict[str, Any]:
        """
        Fix an issue using guided LLM.

        Args:
            issue: Issue dictionary.
            suspected_files: Optional list of suspected files.
            error_output: Optional error output or test info.
            max_iterations: Maximum iterations for progressive refinement.

        Returns:
            Dictionary with fix results.
        """
        # Create initial guidance prompt
        prompt = self.guidance.create_guidance_prompt(
            issue,
            suspected_files=suspected_files,
            error_output=error_output
        )

        # Track iterations
        iterations = []
        current_prompt = prompt

        for i in range(max_iterations):
            logger.info(f"LLM Guided Fix: Iteration {i + 1}/{max_iterations}")

            # Generate solution using model
            response = self.model.generate(current_prompt)

            # Extract patch from response
            patch = self._extract_patch(response)

            # Validate patch
            validation = self.patch_validator.validate_patch(patch, issue.get("instance_id", "unknown"))

            # Record iteration
            iterations.append({
                "iteration": i + 1,
                "prompt": current_prompt,
                "response": response,
                "patch": patch,
                "validation": validation
            })

            # If patch is valid, we're done
            if validation.get("success", False):
                logger.info("Valid patch generated, stopping iterations")
                break

            # If patch is invalid, refine prompt with feedback
            feedback = f"The proposed patch has issues: {validation.get('feedback', 'Unknown error')}"
            current_prompt = self.guidance.apply_feedback(prompt, response, feedback)

        # Return results
        final_iteration = iterations[-1]
        return {
            "issue_id": issue.get("issue_id", "unknown"),
            "model_name": self.model_name,
            "iterations": iterations,
            "final_patch": final_iteration["patch"],
            "patch_validation": final_iteration["validation"],
            "success": final_iteration["validation"].get("success", False)
        }

    def _extract_patch(self, response: str) -> str:
        """
        Extract patch from model response.

        Args:
            response: Model response text.

        Returns:
            Extracted patch string.
        """
        # Look for diff sections
        diff_pattern = r'(diff --git.*?)(?=^```|\Z)'
        diff_match = re.search(diff_pattern, response, re.MULTILINE | re.DOTALL)

        if diff_match:
            return diff_match.group(1).strip()

        # Look for code blocks that might contain patches
        code_block_pattern = r'```(?:diff|patch)?\n(.*?)```'
        code_match = re.search(code_block_pattern, response, re.MULTILINE | re.DOTALL)

        if code_match:
            return code_match.group(1).strip()

        # If no clear patch format is found, return empty string
        return ""
