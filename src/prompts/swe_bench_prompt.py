# src/prompts/swe_bench_prompt.py

from typing import Dict, Any
from src.prompts.base_prompt import BasePrompt
import re


class SWEBenchPrompt(BasePrompt):
    """Prompt templates for SWE-bench problem solving."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SWE-bench prompt.

        Args:
            config: Prompt configuration
        """
        super().__init__(config)
        # self.logger.info(f"[SWE-BENCH PROMPT] Initializing SWE-bench prompt templates")

        # Default templates if not provided in config
        if "generation" not in self.templates:
            self.logger.info(f"[SWE-BENCH PROMPT] Using default generation template")
            self.templates["generation"] = (
                "# GitHub Issue: {issue_id}\n\n"
                "{problem_statement}\n\n"
                "{repository_context}\n\n"  # Add repository context placeholder
                "# Repository Information\n"
                "Repository: {repo}\n"
                "Base commit: {base_commit}\n\n"
                "# Task\n"
                "Your task is to create a patch that fixes this issue. The patch should be in the format of a git diff.\n"
                "Focus on creating a minimal change that addresses the issue while maintaining the code's integrity.\n"
                "Make sure to use the correct file paths based on the repository context provided.\n\n"
                "# Important Guidelines\n"
                "1. Use correct file paths matching the actual repository structure\n"
                "2. Use accurate line numbers for the files you modify\n"
                "3. Reference existing method names and variables exactly as they appear in the code\n"
                "4. Only modify files that actually exist in the repository\n"
                "5. Create a focused patch that addresses only the specific issue mentioned\n\n"
                "Please provide a patch in git diff format."
            )
        else:
            self.logger.info(f"[SWE-BENCH PROMPT] Using custom generation template from config")

        if "reflection" not in self.templates:
            # self.logger.info(f"[SWE-BENCH PROMPT] Using default reflection template")
            self.templates["reflection"] = (
                "# GitHub Issue: {issue_id}\n\n"
                "{problem_statement}\n\n"
                "{specific_guidance}\n\n"
                "# Your Previous Solution\n"
                "{solution}\n\n"
                "# Test Results\n"
                "{output}\n\n"
                "# Errors\n"
                "{errors}\n\n"
                "# Task\n"
                "Based on the test results above, please refine your solution. The patch should be in the format of a git diff.\n"
                "Focus on creating a minimal change that addresses the issue while maintaining the code's integrity.\n"
                "Make sure your solution passes all the required tests.\n\n"
                "# Important Guidelines\n"
                "1. Use correct file paths matching the actual repository structure\n"
                "2. Use accurate line numbers for the files you modify\n"
                "3. Reference existing method names and variables exactly as they appear in the code\n"
                "4. Only modify files that actually exist in the repository\n"
                "5. Create a focused patch that addresses only the specific issue mentioned\n\n"
                "Please provide your refined patch in git diff format."
            )
        else:
            self.logger.info(f"[SWE-BENCH PROMPT] Using custom reflection template from config")

    def format_generation(self, prompt: str, context: Dict[str, Any] = None, task: Dict[str, Any] = None) -> str:
        """
        Format the initial SWE-bench prompt with repository context.

        Args:
            prompt: Input prompt
            context: Additional context (optional)
            task: Task details (optional)

        Returns:
            Formatted prompt
        """
        self.logger.info(f"\n[SWE-BENCH PROMPT] Formatting generation prompt")
        self.logger.info(f"[SWE-BENCH PROMPT] Input prompt length: {len(prompt)} chars")
        # self.logger.info(f"[SWE-BENCH PROMPT] Task: {task.get('name', 'unnamed')}")
        task = task or {}

        # Extract task information
        repo_info = task.get("repo_info", {})
        repo_name = repo_info.get("repo", "")
        base_commit = repo_info.get("base_commit", "")

        self.logger.info(f"[SWE-BENCH PROMPT] Repository: {repo_name}")
        self.logger.info(f"[SWE-BENCH PROMPT] Base commit: {base_commit}")

        # Prepare variables for template
        variables = {
            "issue_id": task.get("name", ""),
            "problem_statement": prompt,
            "repo": repo_name,
            "base_commit": base_commit
        }

        if context:
            self.logger.info(f"[SWE-BENCH PROMPT] Adding context information")
            # You might want to add specific context elements to the prompt
            if context.get("expanded_details"):
                self.logger.info(f"[SWE-BENCH PROMPT] Adding expanded details from context")
                variables["expanded_details"] = self._format_expanded_details(context["expanded_details"])

        # Extract rule or file context from the problem
        # self.logger.info(f"[SWE-BENCH PROMPT] Extracting rule IDs and file paths from prompt")
        rule_ids = self._extract_rule_ids(prompt)
        file_paths = self._extract_file_paths(prompt)

        # self.logger.info(f"[SWE-BENCH PROMPT] Found {len(rule_ids)} rule IDs: {rule_ids}")
        # self.logger.info(f"[SWE-BENCH PROMPT] Found {len(file_paths)} file paths: {file_paths}")

        # Build repository context
        self.logger.info(f"[SWE-BENCH PROMPT] Building repository context")
        repo_context = self._build_repository_context(repo_name, rule_ids, file_paths)
        self.logger.info(f"[SWE-BENCH PROMPT] Repository context length: {len(repo_context)} chars")
        variables["repository_context"] = repo_context

        # Add any additional variables from the task
        additional_vars = []
        for key, value in task.items():
            if key not in variables and isinstance(value, (str, int, float, bool)):
                variables[key] = value
                additional_vars.append(key)

        if additional_vars:
            self.logger.info(f"[SWE-BENCH PROMPT] Added additional variables from task: {additional_vars}")

        # Merge with default variables
        self.logger.info(f"[SWE-BENCH PROMPT] Merging with default variables")
        variables = self._merge_variables(variables)

        # Get the template
        template = self.templates["generation"]
        self.logger.info(f"[SWE-BENCH PROMPT] Generation template length: {len(template)} chars")

        # Format the template
        self.logger.info(f"[SWE-BENCH PROMPT] Formatting template with variables")
        formatted_prompt = self._format_template(template, variables)

        # Add system message if available and not already included
        if self.system_message and "{system_message}" not in template:
            self.logger.info(f"[SWE-BENCH PROMPT] Adding system message (length: {len(self.system_message)})")
            formatted_prompt = f"{self.system_message}\n\n{formatted_prompt}"

        self.logger.info(f"[SWE-BENCH PROMPT] Final formatted prompt length: {len(formatted_prompt)} chars")
        self.logger.info(f"[SWE-BENCH PROMPT] Preview of formatted prompt: {formatted_prompt[:200]}...")
        return formatted_prompt

    def format_reflection(
            self,
            original_prompt: str,
            solution: str,
            output: str,
            errors: str,
            context: Dict[str, Any] = None,
            task: Dict[str, Any] = None
    ) -> str:
        """
        Format the reflection prompt.

        Args:
            original_prompt: Original prompt
            solution: Generated solution (patch)
            output: Test execution output
            errors: Test execution errors
            context: Additional context (optional)
            task: Task details (optional)

        Returns:
            Formatted reflection prompt
        """
        self.logger.info(f"\n[SWE-BENCH PROMPT] Formatting reflection prompt")
        self.logger.info(f"[SWE-BENCH PROMPT] Original prompt length: {len(original_prompt)} chars")
        self.logger.info(f"[SWE-BENCH PROMPT] Solution length: {len(solution)} chars")
        self.logger.info(f"[SWE-BENCH PROMPT] Output length: {len(output)} chars")
        self.logger.info(f"[SWE-BENCH PROMPT] Errors length: {len(errors)} chars")

        # Extract task information
        repo_info = task.get("repo_info", {})
        repo_name = repo_info.get("repo", "")
        self.logger.info(f"[SWE-BENCH PROMPT] Repository: {repo_name}")

        # Prepare variables for template
        self.logger.info(f"[SWE-BENCH PROMPT] Generating specific guidance based on test results")
        specific_guidance = self._generate_specific_guidance(solution, output, errors)
        self.logger.info(f"[SWE-BENCH PROMPT] Specific guidance length: {len(specific_guidance)} chars")

        variables = {
            "issue_id": task.get("name", ""),
            "problem_statement": original_prompt,
            "solution": solution,
            "output": output or "No test output",
            "errors": errors or "No errors",
            "repo": repo_info.get("repo", ""),
            "base_commit": repo_info.get("base_commit", ""),
            "specific_guidance": specific_guidance
        }

        # Add any additional variables from the task
        additional_vars = []
        for key, value in task.items():
            if key not in variables and isinstance(value, (str, int, float, bool)):
                variables[key] = value
                additional_vars.append(key)

        if additional_vars:
            self.logger.info(f"[SWE-BENCH PROMPT] Added additional variables from task: {additional_vars}")

        # Merge with default variables
        self.logger.info(f"[SWE-BENCH PROMPT] Merging with default variables")
        variables = self._merge_variables(variables)

        # Get the template
        template = self.templates["reflection"]
        self.logger.info(f"[SWE-BENCH PROMPT] Reflection template length: {len(template)} chars")

        # Format the template
        self.logger.info(f"[SWE-BENCH PROMPT] Formatting template with variables")
        formatted_prompt = self._format_template(template, variables)

        # Add system message if available and not already included
        if self.system_message and "{system_message}" not in template:
            self.logger.info(f"[SWE-BENCH PROMPT] Adding system message")
            formatted_prompt = f"{self.system_message}\n\n{formatted_prompt}"

        self.logger.info(f"[SWE-BENCH PROMPT] Final formatted reflection prompt length: {len(formatted_prompt)} chars")
        self.logger.info(f"[SWE-BENCH PROMPT] Preview of formatted reflection prompt: {formatted_prompt[:200]}...")
        return formatted_prompt

    def _extract_rule_ids(self, prompt: str) -> list:
        """Extract rule IDs from the problem statement."""
        self.logger.info(f"[SWE-BENCH PROMPT] Extracting rule IDs from prompt")
        # Look for common rule ID patterns like L001, L031, etc.
        rule_pattern = re.compile(r'\b([A-Z][0-9]{3})\b')
        matches = rule_pattern.findall(prompt)
        unique_rules = list(set(matches))  # Return unique rule IDs
        self.logger.info(f"[SWE-BENCH PROMPT] Extracted {len(unique_rules)} unique rule IDs: {unique_rules}")
        return unique_rules

    def _extract_file_paths(self, prompt: str) -> list:
        """Extract potential file paths from the problem statement."""
        self.logger.info(f"[SWE-BENCH PROMPT] Extracting file paths from prompt")
        # Look for file paths with extensions
        file_pattern = re.compile(r'\b([a-zA-Z0-9_/.-]+\.(py|sql|js|html|css|java|cpp|h|md|json|yaml|yml))\b')
        matches = file_pattern.findall(prompt)
        file_paths = [m[0] for m in matches]  # Return just the full file paths
        self.logger.info(f"[SWE-BENCH PROMPT] Extracted {len(file_paths)} file paths: {file_paths}")
        return file_paths

    def _build_repository_context(self, repo_name: str, rule_ids: list, file_paths: list) -> str:
        """Build repository context based on extracted information."""
        self.logger.info(f"[SWE-BENCH PROMPT] Building repository context for {repo_name}")

        if not repo_name:
            self.logger.info(f"[SWE-BENCH PROMPT] No repository name provided, returning empty context")
            return ""

        # Build the context section
        context = "# Repository Context\n\n"
        sections = 0

        # Add information about rules
        if rule_ids:
            self.logger.info(f"[SWE-BENCH PROMPT] Adding {len(rule_ids)} rules to context")
            context += "## Identified Rules\n"
            for rule_id in rule_ids:
                context += f"- Rule {rule_id}\n"
            context += "\n"
            sections += 1

        # Add information about potential files
        if file_paths:
            self.logger.info(f"[SWE-BENCH PROMPT] Adding {len(file_paths)} file paths to context")
            context += "## Potentially Relevant Files\n"
            for file_path in file_paths:
                context += f"- {file_path}\n"
            context += "\n"
            sections += 1

        # Add general guidance
        self.logger.info(f"[SWE-BENCH PROMPT] Adding general patch guidance to context")
        context += "## Important Guidelines for Creating Patches\n"
        context += "1. **Use Correct Line Numbers**: Make sure you reference the actual line numbers in the files.\n"
        context += "2. **Use Correct Method Names**: Always use method names exactly as they appear in the code.\n"
        context += "3. **Keep Changes Minimal**: Focus only on the specific issue mentioned.\n"
        context += "4. **Path Correctness**: Use the exact file paths from the repository.\n"
        context += "5. **Validate Existence**: Only modify files that actually exist in the repository.\n\n"
        sections += 1

        self.logger.info(f"[SWE-BENCH PROMPT] Repository context built with {sections} sections, {len(context)} chars")
        return context

    def _generate_specific_guidance(self, solution: str, output: str, errors: str) -> str:
        """Generate specific guidance based on test results."""
        self.logger.info(f"[SWE-BENCH PROMPT] Generating specific guidance based on test results")
        guidance = "# Specific Guidance for Improvement\n\n"
        guidance_points = []

        # Check if the patch failed to apply
        if "Failed to apply patch" in errors:
            self.logger.info(f"[SWE-BENCH PROMPT] Detected 'Failed to apply patch' error")
            guidance += "⚠️ **Your patch failed to apply.** Please check:\n"
            guidance += "1. Are you using correct file paths?\n"
            guidance += "2. Are line numbers matching the actual file?\n"
            guidance += "3. Are you referencing methods/variables that exist in the code?\n\n"
            guidance_points.append("patch failed to apply")

        # Check if tests are failing
        elif "FAIL" in output:
            self.logger.info(f"[SWE-BENCH PROMPT] Detected test failures in output")
            guidance += "⚠️ **Your patch applied but tests are still failing.** Please check:\n"
            guidance += "1. Did you address the specific issue mentioned in the problem?\n"
            guidance += "2. Are your changes compatible with existing code?\n"
            guidance += "3. Have you considered all edge cases mentioned in the tests?\n\n"
            guidance_points.append("tests failing")

        # Analyze the patch format
        if "diff --git" not in solution:
            self.logger.info(f"[SWE-BENCH PROMPT] Patch format issue: missing 'diff --git'")
            guidance += "⚠️ **Your patch has incorrect format.** Make sure to use proper git diff format.\n"
            guidance += "Example:\n```diff\ndiff --git a/path/to/file.py b/path/to/file.py\nindex abcd123..efgh456 100644\n--- a/path/to/file.py\n+++ b/path/to/file.py\n@@ -10,6 +10,6 @@ def some_function():\n     old line\n+    new line\n```\n\n"
            guidance_points.append("incorrect patch format")

        self.logger.info(f"[SWE-BENCH PROMPT] Generated guidance with {len(guidance_points)} issues: {guidance_points}")
        return guidance

    def _format_expanded_details(self, expanded_details: Dict[str, Any]) -> str:
        """Format expanded details from context into a readable format."""
        if not expanded_details:
            return ""

        result = "## Additional Context Information\n\n"

        # Format signatures if available
        if "signatures" in expanded_details:
            result += "### Signatures\n```python\n"
            for name, signature in expanded_details["signatures"].items():
                result += f"{signature}\n"
            result += "```\n\n"

        # Format implementations if available
        if "implementations" in expanded_details:
            result += "### Implementations\n"
            for name, impl in expanded_details["implementations"].items():
                result += f"#### {name}\n```python\n{impl}\n```\n\n"

        # Format explanations if available
        if "explanations" in expanded_details:
            result += "### Explanations\n"
            for concept, explanation in expanded_details["explanations"].items():
                result += f"#### {concept}\n{explanation}\n\n"

        return result

    def format_tot_reasoning(self, original_prompt: str, parent_reasoning: str,
                             depth: int, strategy: str, task: Dict[str, Any]) -> str:
        """
        Format a prompt for Tree of Thought reasoning about GitHub patches.
        
        Args:
            original_prompt: Original problem statement
            parent_reasoning: Reasoning from parent node
            depth: Current reasoning depth
            strategy: Reasoning strategy to focus on
            task: Task details
            
        Returns:
            Formatted ToT reasoning prompt
        """
        self.logger.info(f"[SWE-BENCH PROMPT] Formatting Tree of Thought reasoning prompt (depth: {depth})")
        
        repo_info = task.get("repo_info", {})
        repo_name = repo_info.get("repo", "")
        base_commit = repo_info.get("base_commit", "")
        
        # Prepare variables for template
        variables = {
            "issue_id": task.get("name", ""),
            "original_prompt": original_prompt,
            "parent_reasoning": parent_reasoning,
            "depth": depth,
            "strategy": strategy,
            "repo": repo_name,
            "base_commit": base_commit
        }
        
        # Check if we have a ToT template in config
        if "tot_reasoning" in self.templates:
            self.logger.info(f"[SWE-BENCH PROMPT] Using custom ToT reasoning template from config")
            template = self.templates["tot_reasoning"]
        else:
            # Default template
            self.logger.info(f"[SWE-BENCH PROMPT] Using default ToT reasoning template")
            template = (
                "# GitHub Issue: {issue_id}\n\n"
                "{original_prompt}\n\n"
                "# Repository Information\n"
                "Repository: {repo}\n"
                "Base commit: {base_commit}\n\n"
                "# Tree of Thought Reasoning (Depth {depth})\n"
                "You are exploring different reasoning paths to solve this problem.\n"
                "Focus on: {strategy}\n\n"
                "# Previous Reasoning\n{parent_reasoning}\n\n"
                "# Instructions\n"
                "1. Analyze the problem and previous reasoning carefully\n"
                "2. Think step-by-step about the best approach\n"
                "3. Consider potential solutions and their implications\n"
                "4. Develop a specific patch to solve the issue\n"
                "5. Output your reasoning and a complete git patch in standard diff format\n\n"
                "Please provide your complete reasoning and patch."
            )
        
        # Merge with default variables
        variables = self._merge_variables(variables)
        
        # Format the template
        formatted_prompt = self._format_template(template, variables)
        
        # Add system message if available and not already included
        if self.system_message and "{system_message}" not in template:
            formatted_prompt = f"{self.system_message}\n\n{formatted_prompt}"
            
        self.logger.info(f"[SWE-BENCH PROMPT] ToT reasoning prompt length: {len(formatted_prompt)} chars")
        return formatted_prompt

