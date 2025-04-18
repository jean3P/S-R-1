# src/utils/relevance.py (complete file)

from typing import Dict, List, Any


def calculate_relevance_score(segment: Dict[str, Any], key_terms: List[str]) -> float:
    """
    Calculate a relevance score for a code segment based on key terms.

    Args:
        segment: Code segment
        key_terms: List of key terms

    Returns:
        Relevance score (0.0 to 1.0)
    """
    score = 0.0

    # Get all text from the segment
    segment_text = ""

    # Add name
    if "name" in segment:
        segment_text += segment["name"] + " "

    # Add docstring
    if "docstring" in segment and segment["docstring"]:
        segment_text += segment["docstring"] + " "

    # Add arguments
    if "args" in segment:
        segment_text += " ".join(segment["args"]) + " "

    # Add returns
    if "returns" in segment and segment["returns"]:
        segment_text += segment["returns"] + " "

    # Add class (for methods)
    if "class" in segment:
        segment_text += segment["class"] + " "

    # Add bases (for classes)
    if "bases" in segment and segment["bases"]:
        segment_text += " ".join(segment["bases"]) + " "

    # Add imports (for files)
    if "imports" in segment and segment["imports"]:
        segment_text += " ".join(segment["imports"]) + " "

    # Count matches
    segment_text = segment_text.lower()
    match_count = 0
    term_weights = {}

    for term in key_terms:
        # Count occurrences
        term_lower = term.lower()
        count = segment_text.count(term_lower)

        if count > 0:
            # Weight by term specificity and match location
            weight = 1.0

            # Exact name match gets high weight
            if "name" in segment and term_lower == segment["name"].lower():
                weight = 3.0

            # Class match for methods gets medium weight
            elif "class" in segment and term_lower == segment["class"].lower():
                weight = 2.0

            # Argument match gets lower weight
            elif "args" in segment and term_lower in [arg.lower() for arg in segment["args"]]:
                weight = 1.5

            term_weights[term] = count * weight
            match_count += count

    # Calculate base score from match count
    if match_count > 0:
        # Log scale to avoid overly high scores for repeated terms
        import math
        score = math.log(1 + match_count) / 5.0  # Normalize to 0-1 range

        # Boost by term weights
        weight_boost = sum(term_weights.values()) / (len(key_terms) * 3.0)  # Normalize
        score += weight_boost

        # Cap at 1.0
        score = min(1.0, score)

    # Add type-specific boosts
    if segment.get("type") == "function":
        score += 0.1  # Slight boost for functions
    elif segment.get("type") == "class":
        score += 0.15  # Higher boost for classes

    return score
