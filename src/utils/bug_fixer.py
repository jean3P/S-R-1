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
    Integrates with bug detector output to focus fixes on the exact location.
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
                     issue_description: str) -> str:
        """
        Generate a targeted fix for the bug with memory optimization.

        Args:
            bug_location: Information from bug detector about where the bug is located.
            issue_description: Description of the issue.
            repository_data: Repository exploration results.

        Returns:
            String containing the Git-formatted patch.
        """
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract necessary information from bug_location
        file_path = bug_location.get('file')
        function_name = bug_location.get('function')
        bug_code = bug_location.get('code_content', '')

        # Simplified structure - no need to extract line numbers
        bug_type = bug_location.get('bug_type', 'Unknown')
        bug_description = bug_location.get('bug_description', 'Unknown issue')
        confidence = bug_location.get('confidence', 0.5)

        # Create prompt focused on minimal bug fixing with enhanced patch guidance
        fix_prompt = f"""
        You are an expert software engineer. Fix the following bug with the MINIMUM possible change:

        ISSUE DESCRIPTION:
        {issue_description}

        BUG LOCATION:
        - File: {file_path}
        - Function: {function_name}
        - Bug Type: {bug_type}
        - Bug Description: {bug_description}
        - Confidence: {confidence}

        CODE WITH BUG:
        ```python
        {bug_code}
        ```

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

        return raw_patch

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

