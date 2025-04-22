# src/utils/llm_guidance.py

import re
import logging
import os
from pathlib import Path
from typing import Dict, List, Any
from ..utils.repository_explorer import RepositoryExplorer
from ..data.data_loader import SWEBenchDataLoader
from ..utils.enhanced_patch_formatter import EnhancedPatchFormatter  # Import the new formatter

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
        self.data_loader = SWEBenchDataLoader(config)

        # Initialize the enhanced patch formatter
        self.patch_formatter = EnhancedPatchFormatter(config)
        self.current_issue = None
        self.current_repo = None

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
        self.current_issue = issue
        self.current_repo = issue.get('repo', '')
        # Get issue description directly from data loader to ensure consistency
        issue_description = self.data_loader.get_issue_description(issue)
        logger.info(f"Issue description length: {len(issue_description)}")

        prompt_parts = []

        # Step 1: Initial Context Setting
        context = self._create_context_setting(issue)
        prompt_parts.append(f"## Initial Context\n{context}")

        # Step 2: Problem Description with Evidence
        # Use the explicitly loaded issue description
        evidence = f"""
        ISSUE DESCRIPTION:
        {issue_description}

        {self._extract_test_cases(issue_description)}

        {(f"ERROR OUTPUT:"
          f"{error_output}") if error_output else ""}
        """
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

        # Add detailed patch formatting instructions
        prompt_parts.append(self._create_patch_formatting_guide())

        # Add a very explicit instruction for the model
        prompt_parts.append("""
        ## MOST IMPORTANT INSTRUCTIONS:
        1. You MUST provide a Git-formatted patch in your response.
        2. Your patch MUST start with "diff --git" and follow standard Git patch format.
        3. Include line numbers, context lines, and proper file paths.
        4. Even if you're not completely sure, provide your best attempt at a patch.
        5. DO NOT just describe the changes - actually create the patch.
        """)

        return "\n\n".join(prompt_parts)

    def _create_patch_formatting_guide(self) -> str:
        """Create a detailed guide for proper patch formatting."""
        return """
        ## PATCH FORMATTING REQUIREMENTS:

        Your patch MUST follow this exact format:

        ```
        diff --git a/path/to/file.py b/path/to/file.py
        --- a/path/to/file.py
        +++ b/path/to/file.py
        @@ -start_line,number_of_lines +start_line,number_of_lines @@ optional section info
         context line (starts with a SPACE)
         context line
        -removed line (starts with a MINUS)
        +added line (starts with a PLUS)
         context line
        ```

        Common formatting pitfalls to avoid:
        1. Always include the "diff --git" header line
        2. Always include both "---" and "+++" lines with proper file paths
        3. Use identical paths in "a/" and "b/" (unless creating/deleting files)
        4. Hunk headers (@@ lines) must contain line numbers with correct syntax
        5. Context lines MUST start with a SPACE character
        6. Removals MUST start with a MINUS character
        7. Additions MUST start with a PLUS character
        8. Ensure line numbers are correct relative to the original file
        9. Include a few lines of context before and after your changes

        This proper formatting is CRITICAL for the patch to be applied successfully.
        """

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
            # Handle both dictionary and list formats for file_contents
            file_content = {}
            file_contents = repo_exploration.get("file_contents", {})

            if isinstance(file_contents, dict):
                # Original dictionary format
                file_content = file_contents.get(file_path, {})
            elif isinstance(file_contents, list):
                # New list format from RepositoryRAG
                for chunk in file_contents:
                    if chunk.get("file_path") == file_path:
                        file_content = chunk
                        break

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
        description = self.data_loader.get_issue_description(issue)
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
        issue_type = "bug fix" if "bug" in self.data_loader.get_issue_description(issue).lower() else "enhancement"
        has_tests = "test" in self.data_loader.get_issue_description(issue).lower()

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

        # Add explicit patch guidance
        refinement += """
        PATCH FORMAT ISSUES:
        - Make sure your patch starts with "diff --git a/..." 
        - Ensure file paths are correct and match the repository structure
        - Use proper context lines (starting with space) for unchanged code
        - Mark removals with "-" and additions with "+"
        - Include hunk headers with line numbers (e.g., @@ -15,5 +15,7 @@)

        PREVIOUS RESPONSE HAD ISSUES, TRY AGAIN WITH A VALID PATCH.
        """

        # Add the enhanced patch formatting guide
        refinement += "\n" + self._create_patch_formatting_guide()

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
        file_paths = re.findall(r'(\w+[/\\]\w+\.\w+)', model_response)
        if file_paths:
            focus_areas.extend(list(set(file_paths)))

        # Extract function names
        functions = re.findall(r'def (\w+)\(', model_response)
        if functions:
            focus_areas.extend([f"function: {func}" for func in set(functions)])

        # Extract class names
        classes = re.findall(r'class (\w+)', model_response)
        if classes:
            focus_areas.extend([f"class: {cls}" for cls in set(classes)])

        return focus_areas

    def extract_patch_from_response(self, response: str) -> str:
        """
        Extract a Git patch from model response with improved pattern matching.

        Args:
            response: Model response text.

        Returns:
            Extracted patch string.
        """
        # Log the first 200 chars of the response
        logger.debug(f"Extracting patch from response starting with: {response[:200]}...")

        # First, use existing extraction logic to get the raw patch
        raw_patch = self._extract_raw_patch(response)

        # If no patch was found, return empty string
        if not raw_patch:
            logger.warning("No patch found in response")
            return ""

        # Then format the extracted patch using the enhanced formatter
        # Get repository name from the current issue being processed
        repo_name = self._get_current_repo_name()

        # Use the enhanced formatter to fix patch format issues
        formatted_patch = self.patch_formatter.format_patch(raw_patch, repo_name)

        if formatted_patch != raw_patch:
            logger.info("Patch was reformatted to fix common issues")
            logger.debug(f"Formatted patch: {formatted_patch[:200]}...")

        return formatted_patch

    def _extract_raw_patch(self, response: str) -> str:
        """Extract the raw patch from model response using existing patterns."""
        # Look for diff sections with Git format
        diff_pattern = r'(diff --git.*?)(?=^```|\Z)'
        diff_match = re.search(diff_pattern, response, re.MULTILINE | re.DOTALL)

        if diff_match:
            patch = diff_match.group(1).strip()
            logger.debug(f"Found patch using diff pattern, {len(patch)} chars")
            return patch

        # Look for code blocks that might contain patches
        code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
        code_blocks = re.findall(code_block_pattern, response, re.MULTILINE | re.DOTALL)

        # Try each code block
        for block in code_blocks:
            if block.strip().startswith("diff --git"):
                logger.debug(f"Found patch in code block, {len(block)} chars")
                return block.strip()

        # Try less restrictive pattern for code blocks with diff content
        for block in code_blocks:
            if "---" in block and "+++" in block and "@@" in block:
                logger.debug(f"Found diff-like content in code block, {len(block)} chars")
                return block.strip()

        # Look for sections that might be patches even if not in code blocks
        patch_sections = re.findall(r'(---\s+a/.*?\n\+\+\+\s+b/.*?(?:\n@@.*?@@.*?)(?:\n[-+\s].*?)+)',
                                    response, re.MULTILINE | re.DOTALL)

        if patch_sections:
            # Add diff header if missing
            first_section = patch_sections[0]
            file_match = re.search(r'---\s+a/(.*?)\n\+\+\+\s+b/(.*?)\n', first_section)
            if file_match:
                file_path = file_match.group(1)
                patch = f"diff --git a/{file_path} b/{file_path}\n{first_section}"
                logger.debug(f"Constructed patch from section, {len(patch)} chars")
                return patch

        # Last resort - look for any lines that look like they could be part of a patch
        patch_lines = []
        for line in response.split('\n'):
            line = line.rstrip()
            if (line.startswith('diff --git') or line.startswith('--- ') or
                    line.startswith('+++ ') or line.startswith('@@ ') or
                    (line and (line[0] in ['+', '-', ' ']) and len(line) > 1)):
                patch_lines.append(line)

        if patch_lines:
            patch = '\n'.join(patch_lines)
            logger.debug(f"Constructed patch from individual lines, {len(patch)} chars")
            return patch

        return ""

    def _get_current_repo_name(self) -> str:
        """Get the repository name for the current issue being processed."""
        # This method should be implemented to return the repository name
        # from the current issue context
        if hasattr(self, 'current_issue') and self.current_issue and 'repo' in self.current_issue:
            return self.current_issue.get('repo', '')

        # If no current issue context, try to extract repo name from other sources
        # This is a fallback that would need to be customized for your specific setup
        # For example, you might use a class variable that gets set during processing
        if hasattr(self, 'current_repo'):
            return self.current_repo

        return ""

    def create_minimal_patch_from_response(self, response: str, issue: Dict[str, Any]) -> str:
        """
        Create a minimal patch when extraction fails.

        Args:
            response: Model response text.
            issue: Issue dictionary with metadata.

        Returns:
            A minimal patch string.
        """
        # Try to identify file paths mentioned
        file_paths = re.findall(r'`([\w/\.-]+\.(?:py|java|js|c|cpp|h|rb))`', response)
        file_paths.extend(re.findall(r'in\s+([\w/\.-]+\.(?:py|java|js|c|cpp|h|rb))', response))

        # Look for functions or methods mentioned
        functions = re.findall(r'function\s+`?(\w+)`?', response)
        functions.extend(re.findall(r'method\s+`?(\w+)`?', response))
        functions.extend(re.findall(r'def\s+(\w+)', response))

        # Look for changes described
        changes = []
        change_blocks = re.findall(r'change\s+(.*?)\s+to\s+(.*?)(?:\.|$)', response, re.IGNORECASE | re.DOTALL)
        for old, new in change_blocks:
            changes.append((old.strip(), new.strip()))

        # If we have files from the issue, use those
        if "files_modified" in issue and not file_paths:
            file_paths = issue.get("files_modified", [])[:1]  # Use the first one

        if file_paths and (functions or changes):
            file_path = file_paths[0]

            # Create a minimal patch
            patch = f"diff --git a/{file_path} b/{file_path}\n"
            patch += f"--- a/{file_path}\n"
            patch += f"+++ b/{file_path}\n"
            patch += "@@ -1,5 +1,5 @@\n"

            if changes:
                old, new = changes[0]
                patch += f"-{old}\n+{new}\n"
            else:
                # Generic placeholder change
                patch += " # Existing line\n"
                patch += "-# Line with a bug\n"
                patch += "+# Fixed line\n"
                patch += " # Another existing line\n"

            # Apply the patch formatter to fix any issues
            repo_name = issue.get("repo", "")
            return self.patch_formatter.format_patch(patch, repo_name)

        return ""
