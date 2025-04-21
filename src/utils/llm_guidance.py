# src/utils/llm_guidance.py

import re
import logging
from pathlib import Path
from typing import Dict, List, Any
from ..utils.repository_explorer import RepositoryExplorer

logger = logging.getLogger(__name__)


class LLMCodeLocationGuidance:
    """
    A framework for guiding LLMs to the correct code location for bug fixing.
    Implements a comprehensive approach with multiple strategies.
    """

    def __init__(self, config):
        """
        Initialize the LLM guidance framework.

        Args:
            config: Configuration object containing paths and settings.
        """
        self.config = config
        self.repo_path = Path(config["data"]["repositories"])
        self.cache_dir = Path(config["data"]["cache_dir"])

    def create_guidance_prompt(self, issue: Dict[str, Any],
                               suspected_files: List[str] = None,
                               error_output: str = None) -> str:
        """
        Create a comprehensive guidance prompt using all strategies.

        Args:
            issue: Issue dictionary containing metadata.
            suspected_files: List of files that might contain the bug.
            error_output: Error output or test failure information.

        Returns:
            A formatted prompt string designed to guide the LLM.
        """
        prompt_parts = []

        # Step 1: Initial Context Setting
        context = self._create_context_setting(issue)
        prompt_parts.append(f"## Initial Context\n{context}")

        # Step 2: Problem Description with Evidence
        evidence = self._create_problem_evidence(issue, error_output)
        prompt_parts.append(f"## Problem Description and Evidence\n{evidence}")

        # Step 3: Systematic Code Analysis
        analysis_guidance = self._create_analysis_guidance(issue, suspected_files)
        prompt_parts.append(f"## Code Analysis Instructions\n{analysis_guidance}")

        # Step 4: Guided Debugging Walkthrough
        debug_steps = self._create_debug_walkthrough(issue)
        prompt_parts.append(f"## Debugging Walkthrough\n{debug_steps}")

        # Step 5: Progressive Refinement
        refinement = self._create_progressive_refinement(issue)
        prompt_parts.append(f"## Refinement Steps\n{refinement}")

        # Final Instructions for Solution Development
        solution_guidance = self._create_solution_guidance(issue)
        prompt_parts.append(f"## Solution Development Guidelines\n{solution_guidance}")

        return "\n\n".join(prompt_parts)

    def _create_context_setting(self, issue: Dict[str, Any]) -> str:
        """
        Create the initial context setting section with improved repository exploration.

        Args:
            issue: Issue dictionary.
            suspected_files: List of files that might contain the bug.

        Returns:
            Formatted context string.
        """
        # Get repository name and path
        repo = issue.get("repo", "")

        # Get repository structure using either provided exploration or by exploring
        if "repository_exploration" in issue and issue["repository_exploration"]:
            repo_exploration = issue["repository_exploration"]
        else:
            repo_explorer = RepositoryExplorer(self.config)  # Create an instance
            repo_exploration = repo_explorer.explore_repository(issue)  # Call on the instance

        # Get structure and relevant files
        structure = "Files found to be relevant for this issue:"
        for file_path in repo_exploration.get("relevant_files", []):
            structure += f"\n- {file_path}"

        # Get code snippets from relevant files
        code_snippets = ""
        for file_path in repo_exploration.get("relevant_files", [])[:3]:  # Limit to top 3 files
            file_content = repo_exploration.get("file_contents", {}).get(file_path, {})
            if "content" in file_content:
                code_snippets += f"\n### File: {file_path}\n"
                content = file_content["content"]
                # Add line numbers to the code
                lines = content.split('\n')
                numbered_lines = [f"{i + 1}: {line}" for i, line in enumerate(lines)]
                code_snippets += '\n'.join(numbered_lines[:100])  # First 100 lines
                if len(lines) > 100:
                    code_snippets += "\n... (truncated)"

        context = f"""
            REPOSITORY: {repo}
            MODULE PATH: {repo_exploration.get("module_path", repo_exploration.get("repo_path", "Unknown"))}
            
            RELEVANT FILES:
            {structure}
            
            KEY TERMS FROM ISSUE:
            {', '.join(repo_exploration.get("key_terms", []))}
            
            FUNCTIONS MENTIONED IN ISSUE:
            {', '.join(repo_exploration.get("functions", []))}
            
            RELEVANT CODE SNIPPETS:
            {code_snippets}
            """
        return context


    def _create_problem_evidence(self, issue: Dict[str, Any], error_output: str = None) -> str:
        """
        Create the problem description with evidence section.

        Args:
            issue: Issue dictionary.
            error_output: Error output or test failure information.

        Returns:
            Formatted problem evidence string.
        """
        description = issue.get("issue_description", "No description available.")

        # Extract test cases if available
        test_cases = self._extract_test_cases(description)

        # Format error output if available
        error_info = f"ERROR OUTPUT:\n{error_output}" if error_output else ""

        return f"""
        ISSUE DESCRIPTION:
        {description}
        
        {test_cases}
        
        {error_info}
        """

    def _extract_test_cases(self, description: str) -> str:
        """
        Extract test cases from the issue description.

        Args:
            description: Issue description text.

        Returns:
            Formatted test cases string.
        """
        # Look for code blocks in the description
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', description, re.DOTALL)

        if not code_blocks:
            return "No explicit test cases found in the description."

        # Format test cases
        test_cases = "TEST CASES:\n"
        for i, block in enumerate(code_blocks):
            test_cases += f"Test Case {i + 1}:\n```python\n{block}\n```\n"

        return test_cases

    def _create_analysis_guidance(self, issue: Dict[str, Any],
                                  suspected_files: List[str] = None) -> str:
        """
        Create the systematic code analysis section.

        Args:
            issue: Issue dictionary.
            suspected_files: List of files that might contain the bug.

        Returns:
            Formatted analysis guidance string.
        """
        # Create tailored analysis instructions
        if suspected_files:
            file_list = "\n".join([f"- {file}" for file in suspected_files])
            specific_guidance = f"""
            Please analyze the following files to identify the bug location:
            {file_list}
            
            For each file:
            1. Identify the key functions and their purposes
            2. Analyze the data flow between functions
            3. Identify any suspicious patterns or inconsistencies
            4. Look for areas where nested structures might be handled incorrectly
            """
        else:
            specific_guidance = """
            Based on the issue description, please:
            1. Identify which files might contain the bug
            2. Determine which functions are likely involved
            3. Analyze their interactions and data flow
            4. Look for code patterns that might cause the described behavior
            """

        return f"""
        ANALYSIS INSTRUCTIONS:
        {specific_guidance}
        
        Pay special attention to:
        - How compound or nested structures are processed
        - Functions that handle matrices or collections
        - Recursive function calls or lack thereof
        - How data is transformed or combined between functions
        
        Please map the logical flow of the code before suggesting a fix.
        """

    def _create_debug_walkthrough(self, issue: Dict[str, Any]) -> str:
        """
        Create the guided debugging walkthrough section.

        Args:
            issue: Issue dictionary.

        Returns:
            Formatted debug walkthrough string.
        """
        # Extract relevant details from issue description
        description = issue.get("issue_description", "")
        has_examples = "```" in description

        walkthrough = """
        DEBUGGING WALKTHROUGH:
        Please trace the execution of the code for the provided examples:
        
        1. First, identify the entry point of the code (main function called by the user)
        2. Follow the execution path step by step, noting how data is transformed
        3. For each function in the call stack:
           a. What are its inputs and expected outputs?
           b. What operations does it perform on the data?
           c. How does it handle different types of inputs?
        4. Compare the execution paths for the working and non-working examples
        5. Identify exactly where the behavior diverges
        
        Focus on how nested structures are handled differently from flat structures.
        """

        if has_examples:
            walkthrough += """
            TRACE BOTH CASES:
            - Trace the execution for the working case
            - Trace the execution for the failing case 
            - Identify exactly where and why the behavior diverges
            """

        return walkthrough

    def _create_progressive_refinement(self, issue: Dict[str, Any]) -> str:
        """
        Create the progressive refinement section.

        Args:
            issue: Issue dictionary.

        Returns:
            Formatted progressive refinement string.
        """
        return """
            REFINEMENT PROCESS:
            Start with a high-level analysis of the issue, then progressively narrow down to the specific bug:
            
            1. First, identify which module or component contains the bug
            2. Next, identify which specific function(s) are involved
            3. Within those functions, identify suspicious sections of code
            4. Finally, pinpoint the exact line(s) that need to be modified
            
            As you refine your analysis, consider:
            - Are there any assumptions in the code that don't hold for nested structures?
            - Are there helper functions that might be handling the special case incorrectly?
            - Is there a simple change that would align the behavior across all cases?
            
            Iteratively refine your understanding until you've identified the minimum change required.
            """

    def _create_solution_guidance(self, issue: Dict[str, Any]) -> str:
        """
        Create the solution development guidance section.

        Args:
            issue: Issue dictionary.

        Returns:
            Formatted solution guidance string.
        """
        # Get issue-specific information that might help tailor the guidance
        issue_type = "bug fix" if "bug" in issue.get("issue_description", "").lower() else "enhancement"
        has_tests = "test" in issue.get("issue_description", "").lower()

        # Add issue-specific guidance
        specific_guidance = ""
        if issue_type == "bug fix":
            specific_guidance += "Focus on minimal changes to fix the bug without introducing new features.\n"
        if has_tests:
            specific_guidance += "Make sure your solution passes all the test cases mentioned in the issue.\n"

        return f"""
    SOLUTION DEVELOPMENT:
    Once you've identified the exact location of the bug, follow these guidelines:

    1. Propose the minimal change necessary to fix the issue
    2. Explain why this change addresses the root cause
    3. Validate that your solution works for all the test cases
    4. Consider potential edge cases or side effects
    5. Format your solution as a proper Git patch with the correct file paths

    {specific_guidance}

    Your final solution should include:
    - The exact file path where the change is made
    - The specific line number(s) being modified
    - The complete patch in Git diff format
    - A clear explanation of why this fix works

    Remember that the simplest solution is often the best one.
    """

    def apply_feedback(self, initial_prompt: str, model_response: str,
                       user_feedback: str) -> str:
        """
        Apply user feedback to refine the guidance prompt.

        Args:
            initial_prompt: The original guidance prompt.
            model_response: The model's response to the initial prompt.
            user_feedback: User feedback on the model's response.

        Returns:
            Refined guidance prompt.
        """
        # Extract what the model focused on
        focus_areas = self._extract_focus_areas(model_response)

        # Create refinement based on user feedback
        refinement = f"""
            FEEDBACK ON PREVIOUS ANALYSIS:
            {user_feedback}
            
            AREAS YOU FOCUSED ON:
            {', '.join(focus_areas) if focus_areas else "No specific areas identified."}
            
            REFINED FOCUS:
            Based on the feedback, please focus your analysis on the following areas:
        """

        # Add specific guidance based on user feedback
        if "wrong file" in user_feedback.lower():
            refinement += "- You identified the wrong file. Please reconsider which file the bug might be in.\n"

        if "wrong function" in user_feedback.lower():
            refinement += ("- You focused on the wrong function. Look more closely at helper functions or utility "
                           "functions.\n")

        if "too complex" in user_feedback.lower():
            refinement += "- Your solution is too complex. The bug can be fixed with a minimal change.\n"

        if "didn't consider" in user_feedback.lower():
            refinement += "- You missed some important aspects of the problem. Review the issue description again.\n"

        # Combine the initial prompt with the refinement
        return f"{initial_prompt}\n\n{refinement}"

    def _extract_focus_areas(self, model_response: str) -> List[str]:
        """
        Extract the areas the model focused on in its response.

        Args:
            model_response: The model's response.

        Returns:
            List of focus areas.
        """
        focus_areas = []

        # Extract file paths
        file_paths = re.findall(r'(\w+[/\\]\w+\.py)', model_response)
        if file_paths:
            focus_areas.extend(file_paths)

        # Extract function names
        functions = re.findall(r'def (\w+)\(', model_response)
        if functions:
            focus_areas.extend([f"function: {func}" for func in functions])

        return focus_areas
