# src/context/memory_manager.py

import time
from typing import Dict, List, Any, Optional
from src.utils.logging import get_logger
from src.utils.tokenization import count_tokens


class MemoryManager:
    """
    Manages memory of previous interactions for continuity across iterations.
    """

    def __init__(self, max_history_items: int = 10, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}
        self.max_history_items = max_history_items
        self.model_name = self.config.get("model_name", "default")
        self.interaction_history = []
        self.insights = []

    def store_interaction(self,
                          prompt: str,
                          response: str,
                          context: Dict[str, Any],
                          result: Optional[Dict[str, Any]] = None) -> None:
        """
        Store an interaction for future reference.

        Args:
            prompt: The prompt used
            response: The model's response
            context: The context provided to the model
            result: The result of evaluating the solution (optional)
        """
        # Create a summary of the context to avoid storing the full context
        context_summary = self._summarize_context(context)

        # Extract insights from the response
        new_insights = self._extract_insights(response)

        # Create interaction record
        interaction = {
            "timestamp": time.time(),
            "prompt_summary": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response_summary": response[:200] + "..." if len(response) > 200 else response,
            "context_summary": context_summary,
            "success": result.get("success", False) if result else None,
            "insights": new_insights
        }

        # Add to history
        self.interaction_history.append(interaction)

        # Add to insights
        self.insights.extend(new_insights)

        # Limit history size
        if len(self.interaction_history) > self.max_history_items:
            self.interaction_history = self.interaction_history[-self.max_history_items:]

        self.logger.info(f"Stored interaction with {len(new_insights)} new insights")

    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a compact summary of the context.

        Args:
            context: Full context

        Returns:
            Summary of context
        """
        # This is a simple implementation - in practice, would be more sophisticated
        summary = {}

        if "segments" in context:
            segment_count = len(context["segments"])
            segment_types = {}
            for segment in context["segments"]:
                segment_type = segment.get("type", "unknown")
                segment_types[segment_type] = segment_types.get(segment_type, 0) + 1

            summary["segments"] = {
                "count": segment_count,
                "types": segment_types
            }

        if "token_usage" in context:
            summary["token_usage"] = context["token_usage"]

        return summary

    def _extract_insights(self, response: str) -> List[str]:
        """
        Extract insights from a model response.

        Args:
            response: Model response

        Returns:
            List of extracted insights
        """
        # This is a placeholder implementation
        # In a real implementation, this would use more sophisticated
        # techniques to extract insights from the response

        insights = []

        # Simple extraction of lines that might contain insights
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines that seem to contain insights
            if any(term in line.lower() for term in
                   ["insight", "observation", "found", "discovered", "problem is",
                    "issue is", "key point", "important", "critical"]):
                insights.append(line)

        return insights

    def retrieve_relevant_interactions(self, current_prompt: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant past interactions based on the current prompt.

        Args:
            current_prompt: The current prompt

        Returns:
            List of relevant past interactions
        """
        # This is a simplified implementation - in practice, would use semantic similarity

        # Calculate rough relevance scores based on term matching
        scored_interactions = []

        for interaction in self.interaction_history:
            score = 0
            # Check prompt overlap
            common_terms = set(current_prompt.lower().split()) & set(interaction["prompt_summary"].lower().split())
            score += len(common_terms) * 0.5

            # Successful solutions get a boost
            if interaction.get("success"):
                score += 2.0

            # More recent interactions get a boost
            recency_boost = max(0, 1.0 - (time.time() - interaction["timestamp"]) / 3600)  # Within the last hour
            score += recency_boost

            scored_interactions.append((interaction, score))

        # Sort by score
        scored_interactions.sort(key=lambda x: x[1], reverse=True)

        # Return top interactions
        return [interaction for interaction, _ in scored_interactions[:3]]

    def build_continuity_context(self, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Build a context for continuity across iterations.

        Args:
            max_tokens: Maximum tokens for continuity context

        Returns:
            Dictionary with continuity context
        """
        # Start with insights
        continuity_context = {
            "insights": self.insights[-10:],  # Last 10 insights
            "interaction_summaries": []
        }

        # Add summaries of recent interactions
        for interaction in reversed(self.interaction_history[-3:]):  # Last 3 interactions
            summary = {
                "prompt_summary": interaction["prompt_summary"],
                "success": interaction.get("success", False),
            }
            continuity_context["interaction_summaries"].append(summary)

        # Calculate token usage
        context_str = str(continuity_context)
        token_count = count_tokens(context_str, self.model_name)

        # If we're over budget, trim insights
        if token_count > max_tokens and continuity_context["insights"]:
            # Start removing insights until we're under budget
            while token_count > max_tokens and continuity_context["insights"]:
                continuity_context["insights"].pop(0)
                context_str = str(continuity_context)
                token_count = count_tokens(context_str, self.model_name)

            # If still over budget, remove interaction summaries
            if token_count > max_tokens and continuity_context["interaction_summaries"]:
                continuity_context["interaction_summaries"] = continuity_context["interaction_summaries"][:1]
                context_str = str(continuity_context)
                token_count = count_tokens(context_str, self.model_name)

        continuity_context["token_count"] = token_count
        return continuity_context
