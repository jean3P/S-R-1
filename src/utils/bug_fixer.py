# src/utils/bug_fixer.py

import logging
import os
import re
import torch
from typing import Dict, Any
from ..utils.enhanced_patch_formatter import EnhancedPatchFormatter

logger = logging.getLogger(__name__)


class BugFixer:
    """
    Utility for generating targeted fixes for precisely located bugs.
    Modified to use memory-efficient approach and enhanced patch formatting.
    """

    def __init__(self, model, config=None):
        """Initialize with a language model for fix generation."""
        self.model = model
        self.config = config

        # Initialize enhanced patch formatter if config is provided
        self.patch_formatter = None
        if config:
            try:
                self.patch_formatter = EnhancedPatchFormatter(config)
            except ImportError as e:
                logger.warning(f"Could not initialize patch formatter: {e}")

    def generate_fix(self, bug_location: Dict[str, Any],
                     issue_description: str,
                     repository_data: Dict[str, Any]) -> str:
        """
        Generate a targeted fix for the bug with memory optimization.

        Args:
            bug_location: Information about where the bug is located.
            issue_description: Description of the issue.
            repository_data: Repository exploration results.

        Returns:
            String containing the Git-formatted patch.
        """
        # Extract only the minimal required code with context
        context_code = self._extract_minimal_code(bug_location, repository_data)

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create prompt focused on minimal bug fixing with enhanced patch guidance
        fix_prompt = f"""
        You are an expert software engineer. Fix the following bug with the MINIMUM possible change:

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        - File: {bug_location.get('file')}
        - Function: {bug_location.get('function')}
        - Line Numbers: {bug_location.get('line_numbers')}
        - Issue: {bug_location.get('issue')}

        RELEVANT CODE (with context):
        {context_code}

        INSTRUCTIONS:
        1. Make the MINIMAL change needed to fix the bug.
        2. Provide ONLY a Git-formatted patch that applies to the exact file.
        3. DO NOT rewrite entire functions - focus only on the problematic line(s).
        4. Ensure the patch has the correct paths, line numbers, and context.

        PATCH FORMAT REQUIREMENTS:
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

        Your response should contain a properly formatted Git patch that can be directly applied.
        """

        # Generate the fix
        response = self.model.generate(fix_prompt)

        # Clear memory again
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract the patch from the response
        raw_patch = self._extract_patch(response)

        # Format the patch if formatter is available
        if self.patch_formatter and repository_data.get("repo"):
            repo_name = repository_data.get("repo")
            formatted_patch = self.patch_formatter.format_patch(raw_patch, repo_name)
            return formatted_patch

        return raw_patch

    def _extract_minimal_code(self, bug_location: Dict[str, Any],
                              repository_data: Dict[str, Any]) -> str:
        """
        Extract only the minimal required code with context to save memory.

        Args:
            bug_location: Information about where the bug is located.
            repository_data: Repository exploration results.

        Returns:
            String containing the minimal relevant code.
        """
        file_path = bug_location.get("file")
        function_name = bug_location.get("function")
        line_numbers = bug_location.get("line_numbers")

        # Check if we need to retrieve the code
        if "file_summaries" in repository_data and not "file_contents" in repository_data:
            # Get the repository explorer
            repo_explorer = repository_data.get("repo_explorer")
            repo_path = repository_data.get("repo_path", "")

            if repo_explorer and function_name:
                # Retrieve just the specific function or class code
                full_file_path = os.path.join(repo_path, file_path) if repo_path else file_path
                component_info = repo_explorer.retrieve_full_code(full_file_path, function_name)

                if "code" in component_info:
                    return component_info["code"]

        # Fallback to old method if we can't use the optimized approach
        if "file_contents" in repository_data and file_path in repository_data.get("file_contents", {}):
            file_info = repository_data["file_contents"][file_path]

            if "error" in file_info:
                return f"Error accessing file: {file_info['error']}"

            # If function is specified, extract function code with context
            if function_name and "functions" in file_info:
                functions = file_info.get("functions", {})

                # Handle both dict and list formats for functions
                if isinstance(functions, dict) and function_name in functions:
                    return functions[function_name].get("code", "Function code not found.")
                elif isinstance(functions, list):
                    for func in functions:
                        if func.get("name") == function_name:
                            return func.get("code", "Function code not found.")

            # If we have line numbers but no function, extract that section
            if line_numbers and "content" in file_info:
                content = file_info["content"]
                lines = content.split('\n')

                # Parse line numbers (could be a range like "10-15" or a single number)
                try:
                    if '-' in line_numbers:
                        start, end = map(int, line_numbers.split('-'))
                    else:
                        start = max(1, int(line_numbers) - 5)  # 5 lines before
                        end = min(len(lines), int(line_numbers) + 5)  # 5 lines after

                    # Extract the relevant section with some context
                    return '\n'.join(lines[start - 1:end])
                except ValueError:
                    pass  # If parsing fails, fall back to full content

            # Fall back to returning a limited portion of the file content
            if "content" in file_info:
                content = file_info["content"]
                lines = content.split('\n')
                # Return first 100 lines as a reasonable default
                return '\n'.join(lines[:min(100, len(lines))])

        return "Could not extract relevant code for the bug location."

    def _extract_patch(self, response: str) -> str:
        """Extract the Git patch from the model's response with improved extraction."""
        # Look for a git diff format
        diff_pattern = r'(diff --git.*?)(?:\Z|(?=^```|\n\n\n))'
        diff_match = re.search(diff_pattern, response, re.MULTILINE | re.DOTALL)

        if diff_match:
            return diff_match.group(1).strip()

        # Look for content inside code blocks
        code_block_pattern = r'```(?:diff|patch|git)?\n(.*?)```'
        code_match = re.search(code_block_pattern, response, re.MULTILINE | re.DOTALL)

        if code_match:
            content = code_match.group(1).strip()
            # Check if it looks like a patch
            if content.startswith('diff --git') or ('---' in content and '+++' in content):
                return content

        # Look for unified diff format without the diff --git header
        unified_diff_pattern = r'(---\s+a/.*?\n\+\+\+\s+b/.*?\n(?:@@.*?@@.*?\n(?:[ \-+].*?\n)+)+)'
        unified_match = re.search(unified_diff_pattern, response, re.MULTILINE | re.DOTALL)

        if unified_match:
            content = unified_match.group(1).strip()
            # Try to add diff --git header by extracting file path
            file_match = re.search(r'---\s+a/(.*?)\n', content)
            if file_match:
                file_path = file_match.group(1)
                return f"diff --git a/{file_path} b/{file_path}\n{content}"
            return content

        # Try to extract anything that looks like a patch
        patch_lines = []
        capture = False
        for line in response.split('\n'):
            # Start capturing when we see a potential patch line
            if (line.startswith('diff --git') or line.startswith('--- ') or
                    line.startswith('+++ ') or line.startswith('@@ ')):
                capture = True
                patch_lines.append(line)
            # Continue capturing lines that look like they're part of a patch
            elif capture and line and line[0] in [' ', '+', '-']:
                patch_lines.append(line)
            # Stop capturing when we hit something that's not a patch line
            elif capture and line and line[0] not in [' ', '+', '-'] and not line.startswith('@@ '):
                # Only stop if we've accumulated some lines
                if len(patch_lines) > 3:
                    break

        if patch_lines:
            return '\n'.join(patch_lines)

        # Default to the entire response if no specific pattern is found
        # Just try to format it as a patch as a last resort
        if "def " in response or "class " in response:
            # This likely contains a code sample - let the formatter try to make a patch
            return response

        # Really no patch content found
        return ""

