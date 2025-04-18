# src/context/context_manager.py

from typing import Dict, List, Any, Optional
import re
from src.utils.logging import get_logger
from src.utils.tokenization import count_tokens
from src.utils.relevance import calculate_relevance_score


class ContextManager:
    """
    Manages the context provided to the LLM by intelligently selecting
    relevant code segments and managing token budget.
    """

    def __init__(self, max_tokens: int = 4000, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}
        self.max_tokens = max_tokens
        self.model_name = self.config.get("model_name", "default")

    def get_relevant_context(self,
                             prompt: str,
                             task: Dict[str, Any],
                             file_summaries: Dict[str, Any] = None,
                             iteration: int = 1) -> Dict[str, Any]:
        """
        Get the most relevant context for the current prompt and task.

        Args:
            prompt: Current prompt
            task: Task details
            file_summaries: Dictionary of file summaries
            iteration: Current iteration number

        Returns:
            Dictionary with relevant context
        """
        # Extract key terms from the prompt
        key_terms = self._extract_key_terms(prompt)
        self.logger.info(f"Extracted {len(key_terms)} key terms from prompt")

        # Get available file summaries
        if file_summaries is None:
            file_summaries = task.get("file_summaries", {})

        # Score code segments by relevance
        segments = []

        # Extract segments from file summaries
        for file_path, summary in file_summaries.items():
            # Add file-level segment
            file_segment = {
                "type": "file",
                "path": file_path,
                "name": summary.get("file_name", ""),
                "imports": summary.get("imports", []),
                "content": None,  # Full content not included in summary
            }
            segments.append(file_segment)

            # Add class-level segments
            for cls in summary.get("classes", []):
                class_segment = {
                    "type": "class",
                    "path": file_path,
                    "name": cls.get("name", ""),
                    "docstring": cls.get("docstring", ""),
                    "bases": cls.get("bases", []),
                }
                segments.append(class_segment)

                # Add method-level segments
                for method in cls.get("methods", []):
                    method_segment = {
                        "type": "method",
                        "path": file_path,
                        "class": cls.get("name", ""),
                        "name": method.get("name", ""),
                        "docstring": method.get("docstring", ""),
                        "args": method.get("args", []),
                        "returns": method.get("returns", None),
                    }
                    segments.append(method_segment)

            # Add function-level segments
            for func in summary.get("functions", []):
                func_segment = {
                    "type": "function",
                    "path": file_path,
                    "name": func.get("name", ""),
                    "docstring": func.get("docstring", ""),
                    "args": func.get("args", []),
                    "returns": func.get("returns", None),
                }
                segments.append(func_segment)

        # Calculate relevance scores
        scored_segments = []
        for segment in segments:
            score = calculate_relevance_score(segment, key_terms)
            scored_segments.append((segment, score))

        # Prioritize segments
        prioritized_segments = self.prioritize_code_segments(scored_segments)

        # Manage token budget
        context = self.manage_token_budget(prompt, prioritized_segments, self.max_tokens)

        return context

    def _extract_key_terms(self, prompt: str) -> List[str]:
        """
        Extract key terms from the prompt.

        Args:
            prompt: Current prompt

        Returns:
            List of key terms
        """
        # Extract potential code identifiers (functions, classes, variables)
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', prompt)

        # Filter common words and short terms
        common_words = {'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'with', 'as',
                        'def', 'class', 'return', 'import', 'from', 'print', 'True', 'False', 'None',
                        'and', 'or', 'not', 'in', 'is', 'the', 'a', 'an', 'of', 'to', 'for', 'in'}

        filtered_terms = [term for term in identifiers
                          if term not in common_words and len(term) > 2]

        # Add exact quoted terms which might be more important
        quoted_terms = re.findall(r'[\'"`](.*?)[\'"`]', prompt)

        # Add terms inside code blocks which might be more important
        code_blocks = re.findall(r'```(?:.*?)\n(.*?)```', prompt, re.DOTALL)
        code_terms = []
        for block in code_blocks:
            code_terms.extend(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', block))

        # Combine all terms
        all_terms = filtered_terms + quoted_terms + code_terms

        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in all_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    def prioritize_code_segments(self,
                                 scored_segments: List[tuple]) -> List[Dict[str, Any]]:
        """
        Prioritize code segments based on relevance scores.

        Args:
            scored_segments: List of (segment, score) tuples

        Returns:
            List of prioritized segments
        """
        # Sort by score in descending order
        sorted_segments = sorted(scored_segments, key=lambda x: x[1], reverse=True)

        # Extract just the segments
        prioritized_segments = [segment for segment, _ in sorted_segments]

        return prioritized_segments

    def manage_token_budget(self,
                            prompt: str,
                            segments: List[Dict[str, Any]],
                            max_tokens: int) -> Dict[str, Any]:
        """
        Manage token budget to optimize context usage.

        Args:
            prompt: Current prompt
            segments: Prioritized segments
            max_tokens: Maximum tokens allowed

        Returns:
            Dictionary with optimized context
        """
        # Calculate tokens used by the prompt
        prompt_tokens = count_tokens(prompt, self.model_name)

        # Calculate available tokens for context
        available_tokens = max_tokens - prompt_tokens - 500  # Reserve 500 tokens for response

        # Ensure we have at least some tokens available
        if available_tokens <= 0:
            self.logger.warning("No tokens available for context")
            return {"segments": [], "token_usage": 0}

        # Select segments that fit within the token budget
        selected_segments = []
        token_usage = 0

        for segment in segments:
            # Serialize segment to calculate token count
            segment_str = str(segment)
            segment_tokens = count_tokens(segment_str, self.model_name)

            if token_usage + segment_tokens <= available_tokens:
                selected_segments.append(segment)
                token_usage += segment_tokens
            else:
                break

        return {
            "segments": selected_segments,
            "token_usage": token_usage,
            "total_segments": len(segments),
            "selected_segments": len(selected_segments)
        }
    