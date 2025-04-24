
import re
import os
from pathlib import Path
from ..data.data_loader import SWEBenchDataLoader
from ..utils.repository_explorer import RepositoryExplorer

class BugDetector:
    """
    Detects specific problematic code regions based on test failures.
    Replaces RAG in the pipeline by providing precise bug locations.
    """

    def __init__(self, config):
        self.config = config
        self.data_loader = SWEBenchDataLoader(config)
        self.repo_explorer = RepositoryExplorer(config)
        self.repo_base_path = Path(config["data"]["repositories"])

    def detect_bug_location(self, issue_id):
        """
        Analyzes test failures to identify problematic code.

        Args:
            issue_id: The issue identifier

        Returns:
            Dict containing bug location information including:
            - file: The file containing the bug
            - function: The function containing the bug
            - line_numbers: Specific line numbers of problematic code
            - code_snippet: The actual code snippet identified as problematic
            - failing_tests: List of failing tests
            - test_messages: Error messages from the tests
        """
        # Load issue data
        issue = self.data_loader.load_issue(issue_id)
        if not issue:
            return {"error": "Issue not found"}

        # Extract test information
        failing_tests = self.data_loader.get_fail_to_pass_tests(issue)
        passing_tests = self.data_loader.get_pass_to_pass_tests(issue)

        # Get test patch information
        test_patch = self.data_loader.get_test_patch(issue)

        # Extract diff information to see what files were modified
        modified_files = self._extract_modified_files(test_patch)

        # Analyze test failures to locate the bug
        bug_location = self._analyze_test_failures(issue, failing_tests, passing_tests, modified_files)

        # Extract the specific code snippet
        code_snippet = self._extract_code_snippet(bug_location)

        # Add code snippet to bug location
        bug_location["code_snippet"] = code_snippet

        return bug_location

    def _extract_modified_files(self, test_patch):
        """Extract files modified in the test patch."""
        modified_files = []

        if isinstance(test_patch, dict) and "patch" in test_patch:
            patch_content = test_patch["patch"]
        else:
            patch_content = test_patch

        if not patch_content:
            return modified_files

        # Parse diff to find modified files
        import re
        file_pattern = r'diff --git a/(.*?) b/'
        matches = re.findall(file_pattern, patch_content)

        for file in matches:
            modified_files.append(file)

        return modified_files

    def _analyze_test_failures(self, issue, failing_tests, passing_tests, modified_files):
        """
        Analyze test failures to determine the bug location.
        This is where the core bug detection logic would be implemented.
        """
        # Start with a basic location structure
        bug_location = {
            "file": None,
            "function": None,
            "line_numbers": None,
            "failing_tests": failing_tests,
            "issue": issue.get("title", "Unknown issue")
        }

        # If we have modified files from the patch, prioritize those
        if modified_files:
            bug_location["file"] = modified_files[0]  # Start with first file

        # Extract function information from failing tests
        if failing_tests:
            # Parse test names to identify tested functions
            for test in failing_tests:
                # Example: Extract function name from test_function_name
                function_match = re.search(r'test_([a-zA-Z0-9_]+)', test.split("::")[-1])
                if function_match:
                    potential_function = function_match.group(1)
                    # Check if this function exists in the modified file
                    if self._verify_function_in_file(potential_function, bug_location["file"]):
                        bug_location["function"] = potential_function
                        break

        # Extract line numbers by analyzing the modified file
        if bug_location["file"] and bug_location["function"]:
            bug_location["line_numbers"] = self._find_function_lines(bug_location["file"], bug_location["function"])

        return bug_location

    def _verify_function_in_file(self, function_name, file_path):
        """Check if the function exists in the specified file."""
        if not file_path:
            return False

        try:
            repo_path = self.repo_base_path
            full_path = os.path.join(repo_path, file_path)

            with open(full_path, 'r') as f:
                content = f.read()

            # Look for function definition
            pattern = rf'def\s+{function_name}'
            if re.search(pattern, content):
                return True

            # Also check for class methods
            pattern = rf'def\s+{function_name}\s*\('
            return bool(re.search(pattern, content))
        except Exception:
            return False

    def _find_function_lines(self, file_path, function_name):
        """Find the line numbers for a specific function in a file."""
        try:
            repo_path = self.repo_base_path
            full_path = os.path.join(repo_path, file_path)

            with open(full_path, 'r') as f:
                lines = f.readlines()

            start_line = None
            end_line = None
            in_function = False
            indentation = 0

            # Find function definition and its scope
            for i, line in enumerate(lines):
                if re.search(rf'def\s+{function_name}\s*\(', line):
                    start_line = i + 1  # 1-based line numbers
                    indentation = len(line) - len(line.lstrip())
                    in_function = True
                elif in_function:
                    if line.strip() and len(line) - len(line.lstrip()) <= indentation:
                        # We've exited the function scope
                        end_line = i
                        break

            if end_line is None and start_line is not None:
                # Function continues to end of file
                end_line = len(lines)

            if start_line and end_line:
                return f"{start_line}-{end_line}"
            return None

        except Exception:
            return None

    def _extract_code_snippet(self, bug_location):
        """Extract the actual code snippet based on bug location."""
        file_path = bug_location.get("file")
        line_range = bug_location.get("line_numbers")

        if not file_path or not line_range:
            return None

        try:
            repo_path = self.repo_base_path
            full_path = os.path.join(repo_path, file_path)

            with open(full_path, 'r') as f:
                lines = f.readlines()

            # Parse line range
            if '-' in line_range:
                start, end = map(int, line_range.split('-'))
            else:
                start = int(line_range)
                end = start + 10  # Default to 10 lines context

            # Convert to 0-based indices
            start = max(0, start - 1)
            end = min(len(lines), end)

            return ''.join(lines[start:end])

        except Exception:
            return None
