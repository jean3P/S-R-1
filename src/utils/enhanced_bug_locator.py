# src/utils/enhanced_bug_locator.py

import logging
import os
import re
import torch
from typing import Dict, List, Any, Optional, Tuple

from .patch_validator import PatchValidator
from .bug_fixer import BugFixer

logger = logging.getLogger(__name__)


class EnhancedBugLocator:
    """
    Enhanced utility for precisely locating bugs in code with integrated validation.
    Implements the first phase of the optimized bug fixing pipeline.
    """

    def __init__(self, model, config):
        """Initialize with a language model and config for validation."""
        self.model = model
        self.config = config
        self.patch_validator = PatchValidator(config)
        self.bug_fixer = BugFixer(model, config)
        self.location_metrics = {
            "file_verification": 0.0,
            "function_verification": 0.0,
            "symptom_correlation": 0.0,
            "specificity_score": 0.0
        }

    def locate_bug_with_validation(
            self,
            issue_id: str,
            issue_description: str,
            repository_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Locate the specific bug with integrated validation.

        Args:
            issue_id: ID of the issue for validation
            issue_description: Description of the issue
            repository_data: Repository exploration results

        Returns:
            Dictionary with bug location information and validation results
        """
        logger.info("Locating bug with integrated validation approach")

        # First pass: Standard bug location
        bug_location = self._perform_initial_location(issue_description, repository_data)

        # Calculate initial location specificity
        specificity = self._calculate_location_specificity(bug_location)
        logger.info(f"Initial bug location specificity: {specificity:.2f}")

        # Track validated patches
        validated_patches = []

        # Verify the location
        verification_result = self._verify_location(bug_location, repository_data)

        # If specificity is high enough to try generating a fix
        if specificity >= 0.6:
            logger.info("Attempting to generate fix from initial bug location")
            # Attempt to generate a patch
            patch = self.bug_fixer.generate_fix(bug_location, issue_description, repository_data)

            if patch:
                # Validate the patch
                validation_result = self.patch_validator.validate_patch(patch, issue_id)
                validated_patches.append({
                    "patch": patch,
                    "validation": validation_result,
                    "source": "initial_location"
                })

                # If patch is valid, return early
                if validation_result.get("success", False):
                    logger.info("Found valid patch from initial bug location")
                    return {
                        "bug_location": bug_location,
                        "specificity": specificity,
                        "verification": verification_result,
                        "valid_patch_found": True,
                        "validated_patches": validated_patches,
                        "metrics": self.location_metrics
                    }

        # Second pass: Apply self-reflection to improve initial location
        improved_location = self._refine_location_with_reflection(
            bug_location,
            issue_description,
            repository_data,
            verified=verification_result.get("verified", False)
        )

        # Calculate improved specificity
        improved_specificity = self._calculate_location_specificity(improved_location)
        logger.info(f"Improved bug location specificity: {improved_specificity:.2f}")

        # Verify the improved location
        improved_verification = self._verify_location(improved_location, repository_data)

        # If improved location is better and specific enough
        if improved_specificity > specificity and improved_specificity >= 0.7:
            logger.info("Attempting to generate fix from improved bug location")
            # Attempt to generate a patch
            patch = self.bug_fixer.generate_fix(improved_location, issue_description, repository_data)

            if patch:
                # Validate the patch
                validation_result = self.patch_validator.validate_patch(patch, issue_id)
                validated_patches.append({
                    "patch": patch,
                    "validation": validation_result,
                    "source": "improved_location"
                })

                # If patch is valid, use improved location
                if validation_result.get("success", False):
                    logger.info("Found valid patch from improved bug location")
                    return {
                        "bug_location": improved_location,
                        "specificity": improved_specificity,
                        "verification": improved_verification,
                        "valid_patch_found": True,
                        "validated_patches": validated_patches,
                        "metrics": self.location_metrics
                    }

        # Return the best location (original or improved)
        if improved_specificity > specificity:
            return {
                "bug_location": improved_location,
                "specificity": improved_specificity,
                "verification": improved_verification,
                "valid_patch_found": False,
                "validated_patches": validated_patches,
                "metrics": self.location_metrics
            }
        else:
            return {
                "bug_location": bug_location,
                "specificity": specificity,
                "verification": verification_result,
                "valid_patch_found": False,
                "validated_patches": validated_patches,
                "metrics": self.location_metrics
            }

    def _perform_initial_location(self, issue_description: str, repository_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform initial bug location using two-phase retrieval approach.

        Args:
            issue_description: Description of the issue
            repository_data: Repository exploration results

        Returns:
            Bug location information
        """
        # Extract key entities from the issue description
        entities = self._extract_entities(issue_description)

        # Get relevant files
        relevant_files = repository_data.get("relevant_files", [])

        # Create detailed prompt for bug location
        location_prompt = f"""
        You are an expert code analyzer tasked with precisely locating a bug. 

        ISSUE DESCRIPTION:
        {issue_description}

        RELEVANT FILES:
        {', '.join(relevant_files[:5])}

        Your task is to identify the exact location of the bug. Provide:
        1. The specific file path
        2. The specific function or method containing the bug
        3. The line numbers where the bug is located
        4. A clear technical explanation of the bug

        Focus on the following entities mentioned in the issue:
        - Functions: {', '.join(entities.get('functions', [])[:5])}
        - Classes: {', '.join(entities.get('classes', [])[:5])}
        - Technical terms: {', '.join(entities.get('technical_terms', [])[:5])}

        Format your answer as follows:
        FILE: [file path]
        FUNCTION: [function name]
        LINE NUMBERS: [start-end or specific line]
        ISSUE: [clear technical explanation of the bug]
        """

        # Generate the bug location analysis
        response = self.model.generate(location_prompt)

        # Parse the response to extract location details
        return self._parse_bug_location(response)

    def _refine_location_with_reflection(
            self,
            initial_location: Dict[str, Any],
            issue_description: str,
            repository_data: Dict[str, Any],
            verified: bool = False
    ) -> Dict[str, Any]:
        """
        Apply self-reflection to refine the bug location.

        Args:
            initial_location: Initial bug location
            issue_description: Issue description
            repository_data: Repository exploration data
            verified: Whether the initial location was verified

        Returns:
            Refined bug location
        """
        # Create the self-reflection prompt
        reflection_prompt = f"""
        You are an expert software engineer reviewing a bug location analysis. The goal is to make the bug location more precise.

        ISSUE DESCRIPTION:
        {issue_description}

        INITIAL BUG LOCATION:
        FILE: {initial_location.get('file', 'Not specified')}
        FUNCTION: {initial_location.get('function', 'Not specified')}
        LINE NUMBERS: {initial_location.get('line_numbers', 'Not specified')}
        ISSUE: {initial_location.get('issue', 'Not specified')}

        {'The initial bug location was successfully verified against the repository.' if verified else 'The initial bug location needs improvement to be more specific and accurate.'}

        Your task is to reflect on this analysis and provide a more precise bug location by:
        1. Making sure the file path is complete and correct
        2. Identifying the exact function/method name
        3. Narrowing down the line number range as much as possible
        4. Providing a more detailed technical explanation of the bug

        Format your improved bug location as follows:
        FILE: [specific file path]
        FUNCTION: [specific function name]
        LINE NUMBERS: [specific line numbers or narrow range]
        ISSUE: [detailed technical explanation]
        """

        # Generate reflection
        reflection_response = self.model.generate(reflection_prompt)

        # Parse the refined bug location
        return self._parse_bug_location(reflection_response)

    def _parse_bug_location(self, response: str) -> Dict[str, Any]:
        """
        Parse the model's response to extract bug location details.

        Args:
            response: Model response text

        Returns:
            Dictionary with bug location details
        """
        bug_location = {
            "file": None,
            "function": None,
            "line_numbers": None,
            "issue": None
        }

        # Extract file path
        file_match = re.search(r'FILE:\s*(.+?)(?:\n|$)', response)
        if file_match:
            bug_location["file"] = file_match.group(1).strip()

        # Extract function name
        function_match = re.search(r'FUNCTION:\s*(.+?)(?:\n|$)', response)
        if function_match:
            bug_location["function"] = function_match.group(1).strip()

        # Extract line numbers
        line_match = re.search(r'LINE NUMBERS?:\s*(.+?)(?:\n|$)', response)
        if line_match:
            bug_location["line_numbers"] = line_match.group(1).strip()

        # Extract issue description
        issue_match = re.search(r'ISSUE:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
        if issue_match:
            bug_location["issue"] = issue_match.group(1).strip()

        return bug_location

    def _calculate_location_specificity(self, bug_location: Dict[str, Any]) -> float:
        """
        Calculate location specificity score (0-1).

        Args:
            bug_location: Bug location information

        Returns:
            Specificity score between 0 and 1
        """
        score = 0.0

        # File identification (0.3)
        if bug_location.get("file"):
            score += 0.3
            self.location_metrics["file_verification"] = 0.3

        # Function identification (0.3)
        if bug_location.get("function"):
            score += 0.3
            self.location_metrics["function_verification"] = 0.3

        # Line number identification (0.2)
        if bug_location.get("line_numbers"):
            # Check if it's a specific line or narrow range
            line_numbers = bug_location.get("line_numbers", "")
            if "-" in line_numbers:
                # Range of lines
                try:
                    start, end = map(int, line_numbers.split("-"))
                    range_size = end - start

                    # Score based on range size
                    if range_size <= 5:
                        score += 0.2  # Very specific
                    elif range_size <= 10:
                        score += 0.15  # Fairly specific
                    else:
                        score += 0.1  # Broad range
                except ValueError:
                    score += 0.1  # Couldn't parse but has something
            else:
                # Single line or comma-separated lines
                score += 0.2

        # Issue explanation (0.2)
        if bug_location.get("issue") and len(bug_location.get("issue", "")) > 20:
            # More points for detailed explanations
            explanation_length = len(bug_location.get("issue", ""))
            if explanation_length > 200:
                score += 0.2
                self.location_metrics["symptom_correlation"] = 0.2
            elif explanation_length > 100:
                score += 0.15
                self.location_metrics["symptom_correlation"] = 0.15
            else:
                score += 0.1
                self.location_metrics["symptom_correlation"] = 0.1

        self.location_metrics["specificity_score"] = min(1.0, score)
        return min(1.0, score)

    def _verify_location(self, bug_location: Dict[str, Any], repository_data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Verify the bug location against repository data.

        Args:
            bug_location: Bug location to verify
            repository_data: Repository exploration data

        Returns:
            Dictionary with verification results
        """
        verification = {
            "verified": False,
            "file_exists": False,
            "function_exists": False,
            "line_range_valid": False
        }

        file_path = bug_location.get("file")
        function_name = bug_location.get("function")

        if not file_path:
            return verification

        # Check if file exists in repository data
        file_contents = repository_data.get("file_contents", {})
        verification["file_exists"] = file_path in file_contents

        if verification["file_exists"] and function_name:
            # Check if function exists in file
            file_info = file_contents[file_path]
            functions = file_info.get("functions", {})

            if isinstance(functions, dict):
                verification["function_exists"] = function_name in functions
            elif isinstance(functions, list):
                verification["function_exists"] = any(f.get("name") == function_name for f in functions)

        # Verify line range if possible
        if verification["file_exists"] and bug_location.get("line_numbers"):
            # Get total lines in file
            file_info = file_contents[file_path]
            if "content" in file_info:
                total_lines = file_info["content"].count("\n") + 1

                # Parse line numbers
                line_numbers = bug_location.get("line_numbers", "")
                try:
                    if "-" in line_numbers:
                        start, end = map(int, line_numbers.split("-"))
                        verification["line_range_valid"] = 1 <= start <= end <= total_lines
                    else:
                        # Could be a single number or comma-separated list
                        for line_str in line_numbers.split(","):
                            line = int(line_str.strip())
                            if not (1 <= line <= total_lines):
                                break
                        else:
                            verification["line_range_valid"] = True
                except (ValueError, TypeError):
                    # Couldn't parse line numbers
                    pass

        # Overall verification
        verification["verified"] = (
                verification["file_exists"] and
                (not function_name or verification["function_exists"]) and
                (not bug_location.get("line_numbers") or verification["line_range_valid"])
        )

        return verification

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract key entities from the issue description.

        Args:
            text: Issue description text

        Returns:
            Dictionary of extracted entities by category
        """
        entities = {
            "functions": [],
            "classes": [],
            "variables": [],
            "technical_terms": [],
            "files": []
        }

        # Remove code blocks for cleaner analysis
        text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

        # Extract function names
        func_patterns = [
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # function calls
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # function definitions
        ]

        for pattern in func_patterns:
            for match in re.finditer(pattern, text_without_code):
                func_name = match.group(1).strip()
                if func_name and len(func_name) > 2 and func_name not in entities["functions"]:
                    entities["functions"].append(func_name)

        # Extract class names
        class_patterns = [
            r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # class definitions
            r'\b([A-Z][a-zA-Z0-9_]*)\b',  # CamelCase potential class names
        ]

        for pattern in class_patterns:
            for match in re.finditer(pattern, text_without_code):
                class_name = match.group(1).strip()
                if class_name and len(class_name) > 2 and class_name not in entities["classes"]:
                    entities["classes"].append(class_name)

        # Extract variable names
        var_pattern = r'\b([a-z_][a-zA-Z0-9_]*)\s*='
        for match in re.finditer(var_pattern, text_without_code):
            var_name = match.group(1).strip()
            if var_name and len(var_name) > 2 and var_name not in entities["variables"]:
                entities["variables"].append(var_name)

        # Extract file paths
        file_patterns = [
            r'\b([a-zA-Z0-9_\-./]+\.(?:py|java|js|c|cpp|h))\b',  # common code file extensions
            r'(?:in|at|from|to)\s+file\s+([a-zA-Z0-9_\-./]+)',  # file references
        ]

        for pattern in file_patterns:
            for match in re.finditer(pattern, text_without_code):
                file_path = match.group(1).strip()
                if file_path and file_path not in entities["files"]:
                    entities["files"].append(file_path)

        # Extract technical terms
        tech_patterns = [
            r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:_[a-zA-Z0-9_]+)+)\b',  # snake_case
            r'\b([a-z]+[A-Z][a-zA-Z0-9]*)\b',  # camelCase
        ]

        for pattern in tech_patterns:
            for match in re.finditer(pattern, text_without_code):
                term = match.group(1).strip()
                if term and len(term) > 3 and term not in entities["technical_terms"]:
                    entities["technical_terms"].append(term)

        return entities
