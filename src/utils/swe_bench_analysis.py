# src/utils/swe_bench_analysis.py

"""
SWE-Bench analysis utilities for code understanding and test interpretation.

This module provides functions for parsing, analyzing, and summarizing
SWE-Bench tests, repositories, and patches to support self-reasoning agents.
"""

import os
import re
import difflib
import subprocess
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict

from src.utils.logging import get_logger

logger = get_logger("swe_bench_analysis")


class RepoDiffAnalyzer:
    """
    Analyzer for extracting and interpreting repository differences.

    This class helps understand changes between different versions of files,
    identify the affected components, and provide context to self-reasoning agents.
    """

    def __init__(self, repo_path: str):
        """
        Initialize the repository diff analyzer.

        Args:
            repo_path: Path to the repository
        """
        print(f"[DEBUG] Initializing RepoDiffAnalyzer with repo_path='{repo_path}'")
        self.repo_path = repo_path
        self.file_cache = {}

    def analyze_patch(self, patch_content: str) -> Dict[str, Any]:
        """
        Analyze a patch to extract key information.

        Args:
            patch_content: Content of the patch

        Returns:
            Analysis results
        """
        print("[DEBUG] analyze_patch called.")
        print(f"[DEBUG] Patch content length: {len(patch_content)} characters.")
        result = {
            "files_changed": [],
            "total_additions": 0,
            "total_deletions": 0,
            "file_summaries": {},
            "components_affected": set(),
            "functions_affected": set(),
            "classes_affected": set(),
            "test_files_changed": False,
            "is_valid_patch": False
        }

        # Check if this is a valid patch
        if not patch_content or not self._is_valid_patch(patch_content):
            print("[DEBUG] Patch is not valid or is empty.")
            return result

        result["is_valid_patch"] = True

        # Parse the patch to extract file names and changes
        affected_files = self._extract_affected_files(patch_content)
        print(f"[DEBUG] Affected files: {list(affected_files.keys())}")
        result["files_changed"] = list(affected_files.keys())

        # Analyze each affected file
        for file_path, changes in affected_files.items():
            print(f"[DEBUG] Analyzing file '{file_path}', {len(changes)} lines of changes.")
            file_summary = self._analyze_file_changes(file_path, changes)
            result["file_summaries"][file_path] = file_summary

            # Update global statistics
            result["total_additions"] += file_summary.get("additions", 0)
            result["total_deletions"] += file_summary.get("deletions", 0)

            # Update affected components
            result["components_affected"].update(file_summary.get("components", set()))
            result["functions_affected"].update(file_summary.get("functions", set()))
            result["classes_affected"].update(file_summary.get("classes", set()))

            # Check if this is a test file
            if "test" in file_path.lower() or "spec" in file_path.lower():
                result["test_files_changed"] = True

        # Convert sets to lists for JSON serialization
        result["components_affected"] = list(result["components_affected"])
        result["functions_affected"] = list(result["functions_affected"])
        result["classes_affected"] = list(result["classes_affected"])

        print("[DEBUG] analyze_patch completed successfully.")
        return result

    def get_failing_test_context(self, test_name: str) -> Dict[str, Any]:
        """
        Get context for a failing test.

        Args:
            test_name: Name of the test

        Returns:
            Test context information
        """
        print(f"[DEBUG] get_failing_test_context called with test_name='{test_name}'")
        context = {
            "test_name": test_name,
            "test_file": "",
            "test_function": "",
            "test_class": "",
            "test_content": "",
            "related_code_files": [],
            "found": False
        }

        # Parse the test name to extract file and function/class information
        file_path, test_part = self._parse_test_name(test_name)
        print(f"[DEBUG] Parsed test_name -> file_path='{file_path}', test_part='{test_part}'")

        if not file_path:
            print("[DEBUG] No valid file_path determined; returning empty context.")
            return context

        context["test_file"] = file_path

        # Try to find the test file in the repository
        full_path = os.path.join(self.repo_path, file_path)
        print(f"[DEBUG] Looking for test file at '{full_path}'")
        if not os.path.exists(full_path):
            # Try alternative paths
            print("[DEBUG] Test file not found; searching for possible alternative paths.")
            possible_paths = self._find_test_file(test_name)
            if possible_paths:
                full_path = possible_paths[0]
                context["test_file"] = os.path.relpath(full_path, self.repo_path)
                print(f"[DEBUG] Found alternative path: '{full_path}'")
            else:
                print("[DEBUG] No alternative paths found; returning context without test content.")
                return context

        # Read the test file
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()

            context["found"] = True

            # Extract the test function or class
            if "." in test_part:
                class_name, func_name = test_part.rsplit(".", 1)
                context["test_class"] = class_name
                context["test_function"] = func_name
                test_content = self._extract_test_method(file_content, class_name, func_name)
            else:
                context["test_function"] = test_part
                test_content = self._extract_test_function(file_content, test_part)

            if test_content:
                context["test_content"] = test_content
                # Find related code files based on imports and references
                context["related_code_files"] = self._find_related_code_files(file_content, full_path)

        except Exception as e:
            logger.error(f"Error getting test context for {test_name}: {str(e)}")
            print(f"[DEBUG] Exception when reading test file or extracting content: {e}")

        print("[DEBUG] get_failing_test_context completed.")
        return context

    def get_surrounding_code(self, file_path: str, line_number: int, context_lines: int = 5) -> str:
        """
        Get surrounding code around a line number.

        Args:
            file_path: Path to the file
            line_number: Line number
            context_lines: Number of context lines before and after

        Returns:
            Surrounding code
        """
        print(f"[DEBUG] get_surrounding_code called with file_path='{file_path}', line_number={line_number}, context_lines={context_lines}")
        if not file_path:
            print("[DEBUG] file_path is empty, returning empty string.")
            return ""

        full_path = os.path.join(self.repo_path, file_path)
        if not os.path.exists(full_path):
            print(f"[DEBUG] File not found at '{full_path}', returning empty string.")
            return ""

        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            start_line = max(0, line_number - context_lines - 1)
            end_line = min(len(lines), line_number + context_lines)
            context_code = "".join(lines[start_line:end_line])
            print(f"[DEBUG] Extracted surrounding code from lines {start_line+1} to {end_line}.")
            return context_code

        except Exception as e:
            logger.error(f"Error getting surrounding code for {file_path}:{line_number}: {str(e)}")
            print(f"[DEBUG] Exception while reading file or slicing lines: {e}")
            return ""

    def get_function_definition(self, file_path: str, function_name: str) -> str:
        """
        Get the definition of a function.

        Args:
            file_path: Path to the file
            function_name: Name of the function

        Returns:
            Function definition
        """
        print(f"[DEBUG] get_function_definition called with file_path='{file_path}', function_name='{function_name}'")
        if not file_path or not function_name:
            print("[DEBUG] Invalid arguments, returning empty string.")
            return ""

        full_path = os.path.join(self.repo_path, file_path)
        if not os.path.exists(full_path):
            print(f"[DEBUG] File not found: '{full_path}'")
            return ""

        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()

            definition = self._extract_function(file_content, function_name)
            print(f"[DEBUG] Extracted function definition length: {len(definition)} chars.")
            return definition

        except Exception as e:
            logger.error(f"Error getting function definition for {function_name} in {file_path}: {str(e)}")
            print(f"[DEBUG] Exception reading file or extracting function: {e}")
            return ""

    def find_function_references(self, function_name: str, max_files: int = 5) -> List[Dict[str, Any]]:
        """
        Find references to a function across the repository.

        Args:
            function_name: Name of the function
            max_files: Maximum number of files to search

        Returns:
            List of references
        """
        print(f"[DEBUG] find_function_references called with function_name='{function_name}', max_files={max_files}")
        references = []

        try:
            # Use git grep to find references
            cmd = ["git", "-C", self.repo_path, "grep", "-l", function_name, "--", "*.py"]
            print(f"[DEBUG] Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode not in [0, 1]:
                logger.error(f"Error searching for function references: {result.stderr}")
                print(f"[DEBUG] Non-zero/1 return code, grep error: {result.stderr}")
                return references

            files = result.stdout.strip().split("\n")
            files = [f for f in files if f]  # Remove empty strings
            print(f"[DEBUG] Found {len(files)} files containing references to '{function_name}'")

            # Limit the number of files
            if max_files > 0 and len(files) > max_files:
                files = files[:max_files]
                print(f"[DEBUG] Truncated files to first {max_files} for performance.")

            for file_path in files:
                full_path = os.path.join(self.repo_path, file_path)
                if not os.path.exists(full_path):
                    print(f"[DEBUG] Skipping because file no longer exists: '{full_path}'")
                    continue

                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

                # Find line numbers with references
                line_numbers = []
                for i, line in enumerate(content.split("\n")):
                    if function_name in line:
                        line_numbers.append(i + 1)

                if line_numbers:
                    snippet = content if len(content) < 10000 else "(content too large)"
                    references.append({
                        "file": file_path,
                        "line_numbers": line_numbers,
                        "content": snippet
                    })

        except Exception as e:
            logger.error(f"Error finding function references for {function_name}: {str(e)}")
            print(f"[DEBUG] Exception in find_function_references: {e}")

        print(f"[DEBUG] find_function_references returning {len(references)} references.")
        return references

    def _is_valid_patch(self, patch_content: str) -> bool:
        """
        Check if a patch is valid.

        Args:
            patch_content: Content of the patch

        Returns:
            True if the patch is valid, False otherwise
        """
        print("[DEBUG] _is_valid_patch called.")
        if not patch_content:
            print("[DEBUG] patch_content is empty.")
            return False

        if "diff --git" not in patch_content and "--- a/" not in patch_content:
            print("[DEBUG] Missing diff headers in patch_content.")
            return False

        if "@@ " not in patch_content:
            print("[DEBUG] Missing chunk headers '@@ ' in patch_content.")
            return False

        print("[DEBUG] Patch is considered valid.")
        return True

    def _extract_affected_files(self, patch_content: str) -> Dict[str, List[str]]:
        """
        Extract affected files from a patch.

        Args:
            patch_content: Content of the patch

        Returns:
            Dictionary mapping file paths to changes
        """
        print("[DEBUG] _extract_affected_files called.")
        affected_files = {}
        current_file = None
        current_changes = []

        for line in patch_content.split("\n"):
            # Check for file headers
            if line.startswith("diff --git"):
                # Save previous file changes
                if current_file and current_changes:
                    affected_files[current_file] = current_changes

                # Extract new file name
                parts = line.split(" ")
                if len(parts) >= 3:
                    # Format: diff --git a/file b/file
                    current_file = parts[2][2:]  # Remove "b/" prefix
                    current_changes = []

            elif line.startswith("--- a/") or line.startswith("+++ b/"):
                # Alternative format: --- a/file or +++ b/file
                if line.startswith("+++ b/"):
                    current_file = line[6:]
                    current_changes = []

            # Collect changes
            elif current_file and line:
                current_changes.append(line)

        # Add the last file
        if current_file and current_changes:
            affected_files[current_file] = current_changes

        print(f"[DEBUG] Found {len(affected_files)} affected files in the patch.")
        return affected_files

    def _analyze_file_changes(self, file_path: str, changes: List[str]) -> Dict[str, Any]:
        """
        Analyze changes to a file.

        Args:
            file_path: Path to the file
            changes: List of changes

        Returns:
            Analysis results
        """
        print(f"[DEBUG] _analyze_file_changes called for '{file_path}', changes size={len(changes)} lines.")
        result = {
            "file_path": file_path,
            "additions": 0,
            "deletions": 0,
            "chunks": 0,
            "components": set(),
            "functions": set(),
            "classes": set(),
            "is_test_file": "test" in file_path.lower() or "spec" in file_path.lower()
        }

        # Count additions and deletions
        for line in changes:
            if line.startswith("+") and not line.startswith("+++"):
                result["additions"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                result["deletions"] += 1
            elif line.startswith("@@"):
                result["chunks"] += 1

        print(f"[DEBUG] additions={result['additions']}, deletions={result['deletions']}, chunks={result['chunks']}")

        # Analyze affected components
        try:
            # Try to read the file
            full_path = os.path.join(self.repo_path, file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    file_content = f.read()

                # Extract modified line numbers from the patch
                modified_lines = self._extract_modified_lines(changes)
                print(f"[DEBUG] Found {len(modified_lines)} modified lines in '{file_path}'")

                # Find affected functions and classes
                if file_path.endswith(".py"):
                    functions, classes = self._find_affected_python_components(file_content, modified_lines)
                    result["functions"].update(functions)
                    result["classes"].update(classes)

                # Extract module/component name from file path
                parts = file_path.split("/")
                if len(parts) > 1:
                    component = parts[0]
                    result["components"].add(component)
                    print(f"[DEBUG] Identified component '{component}' from file path.")
        except Exception as e:
            logger.error(f"Error analyzing file changes for {file_path}: {str(e)}")
            print(f"[DEBUG] Exception analyzing file changes for '{file_path}': {e}")

        return result

    def _extract_modified_lines(self, changes: List[str]) -> Set[int]:
        """
        Extract modified line numbers from patch changes.

        Args:
            changes: List of changes

        Returns:
            Set of modified line numbers
        """
        print("[DEBUG] _extract_modified_lines called.")
        modified_lines = set()
        current_line = 0

        for line in changes:
            if line.startswith("@@"):
                # Parse the chunk header, format: @@ -old_start,old_count +new_start,new_count @@
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1))
                    print(f"[DEBUG] Updated current_line to {current_line} from hunk header.")
            elif line.startswith("+") and not line.startswith("+++"):
                modified_lines.add(current_line)
                current_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                # Skip deleted lines (they don't exist in the current file)
                pass
            else:
                current_line += 1

        print(f"[DEBUG] Found total {len(modified_lines)} modified lines.")
        return modified_lines

    def _find_affected_python_components(self, file_content: str, modified_lines: Set[int]) -> Tuple[Set[str], Set[str]]:
        """
        Find affected Python functions and classes.

        Args:
            file_content: Content of the file
            modified_lines: Set of modified line numbers

        Returns:
            Tuple of (affected_functions, affected_classes)
        """
        print("[DEBUG] _find_affected_python_components called.")
        affected_functions = set()
        affected_classes = set()

        # Split the file into lines
        lines = file_content.split("\n")

        # Find function and class definitions
        function_pattern = re.compile(r"^\s*def\s+(\w+)\s*\(")
        class_pattern = re.compile(r"^\s*class\s+(\w+)\s*[\(:]")

        # Map line numbers to component definitions
        component_lines = {}
        current_class = None
        current_function = None
        current_component_start = 0

        for i, line in enumerate(lines):
            line_num = i + 1

            # Check for class definition
            class_match = class_pattern.match(line)
            if class_match:
                current_class = class_match.group(1)
                current_function = None
                current_component_start = line_num
                continue

            # Check for function definition
            function_match = function_pattern.match(line)
            if function_match:
                current_function = function_match.group(1)
                current_component_start = line_num
                continue

            # Check for end of indentation block (potential end of component)
            if (current_class or current_function) and line and not line.startswith(" ") and not line.startswith("\t"):
                # Record component range
                component_name = current_function or current_class
                if component_name:
                    component_lines[(current_component_start, line_num - 1)] = {
                        "name": component_name,
                        "type": "function" if current_function else "class"
                    }

                current_function = None
                if not function_match:
                    current_class = None

        # Check if any modified lines are within component ranges
        for line_range, component in component_lines.items():
            start, end = line_range
            # For each line in modified_lines, check if it's in this component range
            for line_num in modified_lines:
                if start <= line_num <= end:
                    if component["type"] == "function":
                        affected_functions.add(component["name"])
                    else:
                        affected_classes.add(component["name"])
                    break

        print(f"[DEBUG] Affected functions: {affected_functions}, affected classes: {affected_classes}")
        return affected_functions, affected_classes

    def _parse_test_name(self, test_name: str) -> Tuple[str, str]:
        """
        Parse a test name to extract file path and test part.

        Args:
            test_name: Name of the test

        Returns:
            Tuple of (file_path, test_part)
        """
        print(f"[DEBUG] _parse_test_name called with test_name='{test_name}'")
        if not test_name:
            print("[DEBUG] test_name is empty, returning empty tuple.")
            return "", ""

        # Common formats:
        # 1. path/to/test_file.py::TestClass::test_function
        # 2. path.to.test_file.TestClass.test_function
        # 3. path/to/test_file.py::test_function

        # Handle format 1
        if "::" in test_name:
            file_path, test_part = test_name.split("::", 1)
            return file_path, test_part

        # Handle format 2
        if "." in test_name:
            # Convert dots to slashes for file path
            parts = test_name.split(".")

            # Find where the file path ends and the test part begins
            for i in range(len(parts) - 1, 0, -1):
                if parts[i][0].isupper() or parts[i].startswith("test_"):
                    # This is probably the start of the test part
                    file_path = "/".join(parts[:i])
                    test_part = ".".join(parts[i:])
                    return file_path + ".py", test_part

            # If we couldn't determine, use the last component as the test part
            file_path = "/".join(parts[:-1])
            test_part = parts[-1]
            return file_path + ".py", test_part

        # Handle simple file path
        return test_name, ""

    def _find_test_file(self, test_name: str) -> List[str]:
        """
        Find a test file in the repository.

        Args:
            test_name: Name of the test

        Returns:
            List of possible file paths
        """
        print(f"[DEBUG] _find_test_file called with test_name='{test_name}'")
        possible_paths = []

        # Extract the base name (without path)
        if "/" in test_name:
            base_name = test_name.split("/")[-1]
        elif "." in test_name:
            parts = test_name.split(".")
            for i, part in enumerate(parts):
                if part.startswith("test_") or "test" in part.lower():
                    base_name = part
                    break
            else:
                base_name = parts[-1]
        else:
            base_name = test_name

        print(f"[DEBUG] Searching for base_name='{base_name}' in repo.")
        # Search for files matching the name
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if base_name in file and (file.startswith("test_") or "test" in file.lower()) and file.endswith(".py"):
                    found_path = os.path.join(root, file)
                    possible_paths.append(found_path)
                    print(f"[DEBUG] Found test file candidate: '{found_path}'")

        print(f"[DEBUG] Returning {len(possible_paths)} possible test files.")
        return possible_paths

    def _extract_test_function(self, file_content: str, function_name: str) -> str:
        """
        Extract a test function from file content.

        Args:
            file_content: Content of the file
            function_name: Name of the function

        Returns:
            Test function code
        """
        print(f"[DEBUG] _extract_test_function called with function_name='{function_name}'")
        if not file_content or not function_name:
            print("[DEBUG] Invalid arguments or empty content, returning empty string.")
            return ""

        # Define the pattern to match the function
        pattern = re.compile(f"(def\\s+{re.escape(function_name)}\\s*\\(.*?)(def\\s+|$)", re.DOTALL)

        # Find the function
        match = pattern.search(file_content)
        if match:
            print(f"[DEBUG] Found test function definition for '{function_name}'.")
            return match.group(1)

        print("[DEBUG] Test function definition not found.")
        return ""

    def _extract_test_method(self, file_content: str, class_name: str, method_name: str) -> str:
        """
        Extract a test method from file content.

        Args:
            file_content: Content of the file
            class_name: Name of the class
            method_name: Name of the method

        Returns:
            Test method code
        """
        print(f"[DEBUG] _extract_test_method called with class_name='{class_name}', method_name='{method_name}'")
        if not file_content or not class_name or not method_name:
            print("[DEBUG] Invalid arguments or empty content, returning empty string.")
            return ""

        # First, find the class
        class_pattern = re.compile(f"(class\\s+{re.escape(class_name)}\\s*\\(.*?)(class\\s+|$)", re.DOTALL)
        class_match = class_pattern.search(file_content)

        if not class_match:
            print("[DEBUG] Class definition not found in file_content.")
            return ""

        class_content = class_match.group(1)

        # Now find the method within the class
        method_pattern = re.compile(f"(\\s+def\\s+{re.escape(method_name)}\\s*\\(.*?)(\\s+def\\s+|$)", re.DOTALL)
        method_match = method_pattern.search(class_content)

        if method_match:
            print(f"[DEBUG] Found method definition for '{method_name}' in class '{class_name}'.")
            return method_match.group(1)

        print("[DEBUG] Method definition not found.")
        return ""

    def _extract_function(self, file_content: str, function_name: str) -> str:
        """
        Extract a function from file content.

        Args:
            file_content: Content of the file
            function_name: Name of the function

        Returns:
            Function code
        """
        print(f"[DEBUG] _extract_function called with function_name='{function_name}'")
        if not file_content or not function_name:
            print("[DEBUG] Invalid arguments or empty content, returning empty string.")
            return ""

        # Define the pattern to match the function
        pattern = re.compile(f"(def\\s+{re.escape(function_name)}\\s*\\(.*?)(def\\s+|$)", re.DOTALL)

        # Find the function
        match = pattern.search(file_content)
        if match:
            print(f"[DEBUG] Found function definition for '{function_name}'.")
            return match.group(1)

        print("[DEBUG] Function definition not found.")
        return ""

    def _find_related_code_files(self, test_content: str, test_file_path: str) -> List[str]:
        """
        Find code files related to a test.

        Args:
            test_content: Content of the test file
            test_file_path: Path to the test file

        Returns:
            List of related file paths
        """
        print("[DEBUG] _find_related_code_files called.")
        related_files = []

        # Extract imports
        import_pattern = re.compile(r"(?:from|import)\s+([\w\.]+)")
        imports = import_pattern.findall(test_content)
        print(f"[DEBUG] Found {len(imports)} import statements to analyze.")

        # Look for paths and modules being tested
        for import_path in imports:
            path_guess = import_path.replace(".", "/")

            # Try common file extensions
            for ext in [".py", "/__init__.py", ".pyx", ".pxd"]:
                file_guess = path_guess + ext
                full_path = os.path.join(self.repo_path, file_guess)

                if os.path.exists(full_path):
                    related_files.append(file_guess)
                    print(f"[DEBUG] Found related code file: '{file_guess}'")

        # Look for mentions of code files in test content
        file_pattern = re.compile(r"['\"]([\/\w\-\.]+\.py)['\"]")
        file_mentions = file_pattern.findall(test_content)

        for file_path in file_mentions:
            full_path = os.path.join(self.repo_path, file_path)
            if os.path.exists(full_path):
                related_files.append(file_path)
                print(f"[DEBUG] Found code file mention: '{file_path}'")

        # If test_file_path follows naming convention like test_xyz.py,
        # look for xyz.py in the same directory
        test_file_name = os.path.basename(test_file_path)
        if test_file_name.startswith("test_"):
            base_name = test_file_name[5:]
            parent_dir = os.path.dirname(test_file_path)
            possible_source = os.path.join(parent_dir, base_name)

            if os.path.exists(possible_source):
                rel_path = os.path.relpath(possible_source, self.repo_path)
                related_files.append(rel_path)
                print(f"[DEBUG] Found matching non-test file for '{test_file_name}': '{rel_path}'")

        # Remove duplicates while preserving order
        unique_files = []
        for file_path in related_files:
            if file_path not in unique_files:
                unique_files.append(file_path)

        print(f"[DEBUG] _find_related_code_files returning {len(unique_files)} related files.")
        return unique_files




