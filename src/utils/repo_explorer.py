# src/utils/repo_explorer.py

import os
import re
import subprocess
from typing import List, Optional, Dict, Any
from src.utils.logging import get_logger

logger = get_logger("repo_explorer")


class RepoExplorer:
    """Utility class to explore repository structure for relevant files."""

    def __init__(self, repo_path: str):
        """
        Initialize the repository explorer.

        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = repo_path
        self.file_cache = {}  # Cache file contents to avoid repeated disk reads

    def find_rule_definition(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        Find the file containing a specific rule definition.

        Args:
            rule_id: ID of the rule to locate (e.g. 'L031', 'L060')

        Returns:
            Dictionary with file path and context or None if not found
        """
        try:
            # First approach: Use git grep to find rule class definitions
            class_pattern = f"class.*{rule_id}"
            result = self._run_git_command(['grep', '-l', class_pattern, '--', "*.py"])

            if not result:
                # Second approach: Look for rule mentions
                result = self._run_git_command(['grep', '-l', rule_id, '--', "*.py"])

            if not result:
                logger.warning(f"Could not find any files referencing rule {rule_id}")
                return None

            # Filter likely candidates and prioritize
            candidates = []
            for file_path in result:
                if not os.path.exists(os.path.join(self.repo_path, file_path)):
                    continue

                content = self._get_file_content(file_path)

                # Check for rule class definition pattern
                rule_class_pattern = re.compile(r'class\s+(?:Rule_)?{}(?:\(|\s)'.format(rule_id))
                if rule_class_pattern.search(content):
                    # Higher priority: file contains rule class definition
                    candidates.insert(0, file_path)
                elif re.search(r'["\']{}["\']'.format(rule_id), content):
                    # Lower priority: file just mentions the rule ID
                    candidates.append(file_path)

            if not candidates:
                logger.warning(f"Found files for {rule_id} but none contained rule definition")
                return None

            # Use the most likely candidate
            target_file = candidates[0]
            content = self._get_file_content(target_file)

            # Find the rule implementation section
            context = self._extract_rule_context(content, rule_id)

            return {
                "file_path": target_file,
                "content": content,
                "context": context,
                "rule_id": rule_id
            }

        except Exception as e:
            logger.error(f"Error finding rule definition for {rule_id}: {str(e)}")
            return None

    def _run_git_command(self, args: List[str]) -> List[str]:
        """Run a git command and return output lines."""
        try:
            cmd = ['git'] + args
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )

            if result.returncode != 0:
                logger.debug(f"Git command exited with {result.returncode}: {' '.join(cmd)}")
                return []

            return [line for line in result.stdout.strip().split('\n') if line]

        except Exception as e:
            logger.error(f"Error running git command: {str(e)}")
            return []

    def _get_file_content(self, file_path: str) -> str:
        """Get file content with caching."""
        if file_path in self.file_cache:
            return self.file_cache[file_path]

        try:
            full_path = os.path.join(self.repo_path, file_path)
            with open(full_path, 'r', encoding='utf-8') as file:
                content = file.read()

            self.file_cache[file_path] = content
            return content

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return ""

    def _extract_rule_context(self, content: str, rule_id: str) -> Dict[str, Any]:
        """Extract the relevant context for a rule from file content."""
        context = {
            "line_number": None,
            "description_line": None,
            "class_def_line": None,
            "code_snippet": None
        }

        lines = content.split('\n')
        rule_class_pattern = re.compile(r'class\s+(?:Rule_)?{}(?:\(|\s)'.format(rule_id))
        description_pattern = re.compile(r'description\s*=\s*[\'"]([^\'"]*)[\'"]')

        # Find rule class definition
        for i, line in enumerate(lines):
            if rule_class_pattern.search(line):
                context["class_def_line"] = i + 1  # 1-based line number
                break

        # Find description assignment
        if context["class_def_line"]:
            # Look for description in the class body
            for i in range(context["class_def_line"], len(lines)):
                if description_pattern.search(lines[i]):
                    context["description_line"] = i + 1
                    break

                # If we've reached another class definition, stop searching
                if i > context["class_def_line"] and re.match(r'class\s+', lines[i]):
                    break

        # Extract a code snippet around the description
        if context["description_line"]:
            start_line = max(0, context["description_line"] - 5)
            end_line = min(len(lines), context["description_line"] + 5)
            context["code_snippet"] = '\n'.join(lines[start_line:end_line])
            context["line_number"] = context["description_line"]

        return context
