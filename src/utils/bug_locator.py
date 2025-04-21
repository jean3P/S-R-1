# src/utils/bug_locator.py
import logging
import os
import re
import torch
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class BugLocator:
    """
    Utility for precisely locating bugs in code based on issue descriptions.
    """

    def __init__(self, model):
        """Initialize with a language model for analysis."""
        self.model = model

    def locate_bug(self, issue_description: str, repository_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Locate the specific bug based on issue description and repository data.
        Modified to use a two-phase RAG-like approach for reduced memory usage.

        Args:
            issue_description: Description of the issue.
            repository_data: Repository exploration results.

        Returns:
            Dictionary with bug location information.
        """
        logger.info("Locating bug with memory-efficient approach")

        # Phase 1: Work with code summaries to identify suspicious components
        suspicious_components = self._identify_suspicious_components(issue_description, repository_data)
        logger.info(f"Identified {len(suspicious_components)} suspicious components")

        # Free up memory before retrieving detailed code
        torch.cuda.empty_cache()

        # Phase 2: Retrieve and analyze only the suspicious components
        detailed_code = self._retrieve_detailed_code(repository_data, suspicious_components)
        logger.info(f"Retrieved detailed code for {len(detailed_code)} components")

        # Free up memory again
        torch.cuda.empty_cache()

        # Phase 3: Final analysis with focused code
        bug_location = self._analyze_focused_code(issue_description, detailed_code)
        logger.info(f"Bug location identified: {bug_location.get('file')}, {bug_location.get('function')}")

        return bug_location

    def _identify_suspicious_components(self, issue_description: str, repository_data: Dict[str, Any]) -> List[
        Dict[str, str]]:
        """
        Identify suspicious components based on code summaries.

        Args:
            issue_description: Description of the issue.
            repository_data: Repository exploration results with code summaries.

        Returns:
            List of suspicious components with file and component name.
        """
        # Format the summaries for the model
        formatted_summaries = self._format_summaries_for_model(repository_data)

        # Create prompt focused on identifying suspicious components
        prompt = f"""
        You are an expert code analyzer. Based on the issue description, identify which components (functions or classes)
        might contain a bug. DO NOT analyze the code in detail yet, just identify suspicious components.

        ISSUE DESCRIPTION:
        {issue_description}

        CODE SUMMARIES:
        {formatted_summaries}

        Identify up to 5 components that might contain the bug. For each component, provide:
        1. The file path
        2. The component name (function or class)
        3. A brief reason why this component might contain the bug

        Format your response as follows:
        COMPONENT 1:
        FILE: [file path]
        NAME: [component name]
        REASON: [brief reason]

        COMPONENT 2:
        FILE: [file path]
        NAME: [component name]
        REASON: [brief reason]

        ... and so on.
        """

        # Clear memory before generating
        torch.cuda.empty_cache()

        # Generate the analysis
        response = self.model.generate(prompt)

        # Parse the response to extract suspicious components
        components = []

        component_pattern = r'COMPONENT\s+\d+:\s*\n\s*FILE:\s*(.*?)\s*\n\s*NAME:\s*(.*?)\s*\n\s*REASON:'
        matches = re.finditer(component_pattern, response, re.DOTALL)

        for match in matches:
            file_path = match.group(1).strip()
            component_name = match.group(2).strip()

            components.append({
                "file": file_path,
                "component": component_name
            })

        # If no components were found with the pattern, attempt a fallback
        if not components:
            # Look for file and function names mentioned in the response
            file_mentions = re.findall(r'([/\w]+\.py)', response)
            func_mentions = re.findall(r'`(\w+)`|function\s+(\w+)|method\s+(\w+)|class\s+(\w+)', response)

            # Combine the most likely file with the most likely function
            if file_mentions and func_mentions:
                component_name = next((name for group in func_mentions for name in group if name), "")
                if component_name:
                    components.append({
                        "file": file_mentions[0],
                        "component": component_name
                    })

        return components

    def _format_summaries_for_model(self, repository_data: Dict[str, Any]) -> str:
        """Format code summaries in a readable way for the model."""
        formatted_text = ""

        file_summaries = repository_data.get("file_summaries", {})
        for file_path, file_info in file_summaries.items():
            formatted_text += f"\n### FILE: {file_path}\n"

            if "error" in file_info:
                formatted_text += f"Error: {file_info['error']}\n"
                continue

            summaries = file_info.get("summaries", [])
            for summary in summaries:
                summary_type = summary.get("type", "unknown")
                name = summary.get("name", "unnamed")
                line = summary.get("line", "?")
                summary_text = summary.get("summary", "No summary available")

                formatted_text += f"- {summary_type} `{name}` (Line {line}): {summary_text}\n"

                # Add method names for classes
                if summary_type == "class" and "methods" in summary:
                    methods = summary.get("methods", [])
                    if methods:
                        formatted_text += f"  Methods: {', '.join(methods)}\n"

            formatted_text += "\n"

        return formatted_text

    def _retrieve_detailed_code(self, repository_data: Dict[str, Any], suspicious_components: List[Dict[str, str]]) -> \
    List[Dict[str, Any]]:
        """
        Retrieve detailed code for suspicious components.

        Args:
            repository_data: Repository exploration results.
            suspicious_components: List of suspicious components.

        Returns:
            List of detailed code information for each component.
        """
        detailed_code = []
        repo_explorer = repository_data.get("repo_explorer", None)

        if not repo_explorer:
            # Create a minimal repository explorer
            from ..config.config import Config
            from ..utils.repository_explorer import RepositoryExplorer
            config = Config()
            repo_explorer = RepositoryExplorer(config)

        repo_path = repository_data.get("repo_path", "")

        for component in suspicious_components:
            file_path = component.get("file", "")
            component_name = component.get("component", "")

            # Skip if file path or component name is missing
            if not file_path or not component_name:
                continue

            # Retrieve full code for the component
            code_info = repo_explorer.retrieve_full_code(
                os.path.join(repo_path, file_path) if repo_path else file_path,
                component_name
            )

            if "error" not in code_info:
                detailed_code.append(code_info)

            # Clear memory after each component
            torch.cuda.empty_cache()

        return detailed_code

    def _analyze_focused_code(self, issue_description: str, detailed_code: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform final analysis with focused code to locate the bug.

        Args:
            issue_description: Description of the issue.
            detailed_code: List of detailed code information.

        Returns:
            Dictionary with bug location information.
        """
        # Format the detailed code for the model
        formatted_code = self._format_detailed_code(detailed_code)

        # Create prompt focused only on bug localization
        prompt = f"""
        You are an expert code analyzer. Based on the issue description and the detailed code, identify the EXACT location 
        of the bug. DO NOT suggest fixes yet, just locate the bug precisely.

        ISSUE DESCRIPTION:
        {issue_description}

        DETAILED CODE:
        {formatted_code}

        Your task is to:
        1. Identify which file contains the bug
        2. Identify which function contains the bug
        3. Identify the specific line or lines that need to be changed
        4. Explain why this is a bug (what incorrect behavior it causes)

        Provide your analysis in this format:
        FILE: [file path]
        FUNCTION: [function name]
        LINE NUMBERS: [start-end or specific line]
        ISSUE: [explanation of the bug]
        """

        # Clear memory before generating
        torch.cuda.empty_cache()

        # Generate the bug location analysis
        response = self.model.generate(prompt)

        # Parse the response to extract bug location details
        return self._parse_bug_location(response)

    def _format_detailed_code(self, detailed_code: List[Dict[str, Any]]) -> str:
        """Format detailed code in a readable way for the model."""
        formatted_text = ""

        for code_info in detailed_code:
            file_path = code_info.get("file_path", "unknown")
            component_type = code_info.get("type", "unknown")
            name = code_info.get("name", "unnamed")
            start_line = code_info.get("start_line", "?")
            end_line = code_info.get("end_line", "?")
            code = code_info.get("code", "# No code available")

            formatted_text += f"\n### FILE: {file_path}\n"
            formatted_text += f"### {component_type.upper()}: {name} (Lines {start_line}-{end_line})\n"
            formatted_text += "```python\n"
            formatted_text += code
            formatted_text += "\n```\n\n"

        return formatted_text

    def _parse_bug_location(self, response: str) -> Dict[str, Any]:
        """Parse the model's response to extract bug location details."""
        result = {
            "file": None,
            "function": None,
            "line_numbers": None,
            "issue": None
        }

        # Extract file path
        file_match = re.search(r'FILE:\s*(.+?)(?:\n|$)', response)
        if file_match:
            result["file"] = file_match.group(1).strip()

        # Extract function name
        function_match = re.search(r'FUNCTION:\s*(.+?)(?:\n|$)', response)
        if function_match:
            result["function"] = function_match.group(1).strip()

        # Extract line numbers
        line_match = re.search(r'LINE NUMBERS?:\s*(.+?)(?:\n|$)', response)
        if line_match:
            result["line_numbers"] = line_match.group(1).strip()

        # Extract issue description
        issue_match = re.search(r'ISSUE:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
        if issue_match:
            result["issue"] = issue_match.group(1).strip()

        return result
