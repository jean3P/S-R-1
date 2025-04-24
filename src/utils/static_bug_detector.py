"""
Static Analysis Bug Detector for SWE-bench.

This module implements a bug detection approach using static code analysis
to precisely locate bugs in code.
"""

import logging
import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..data.data_loader import SWEBenchDataLoader

logger = logging.getLogger(__name__)


class StaticAnalysisBugDetector:
    """
    Detects bugs in code using static analysis.

    This implements an approach that:
    1. Extracts context from code using AST parsing
    2. Uses test files to infer implementation files
    3. Ranks code locations by likelihood of containing bugs
    """

    def __init__(self, config):
        """
        Initialize the bug detector.

        Args:
            config: Configuration object containing paths and settings.
        """
        self.config = config
        self.data_loader = SWEBenchDataLoader(config)
        self.repo_base_path = Path(config["data"]["repositories"])

    def detect_bug_location(self, issue_id: str) -> Dict[str, Any]:
        """
        Analyzes code to identify the location of a bug.

        Args:
            issue_id: The issue identifier

        Returns:
            Dict containing bug location information
        """
        logger.info(f"Detecting bug location for issue {issue_id}")

        # Load issue data
        issue = self.data_loader.load_issue(issue_id)
        if not issue:
            logger.error(f"Issue {issue_id} not found")
            return {"error": "Issue not found"}

        logger.info(f"Successfully loaded issue {issue_id}")

        # Prepare repository for analysis
        if not self.data_loader.prepare_repository_for_analysis(issue):
            logger.error(f"Failed to prepare repository for analysis for issue {issue_id}")
            return {"error": "Failed to prepare repository for analysis"}

        logger.info(f"Repository prepared for analysis for issue {issue_id}")

        # Get repository information
        repo = issue.get("repo", "")
        repo_path = self.repo_base_path / repo

        logger.info(f"Repository path: {repo_path}, exists: {repo_path.exists()}")

        # Index the repository structure
        logger.info(f"Indexing repository structure for {repo}")
        python_files = self._index_repository(repo_path)
        logger.info(f"Found {len(python_files)} Python files in repository")

        # Perform static analysis
        logger.info("Using static analysis to locate bug")
        static_result = self._perform_static_analysis(issue, python_files)

        if static_result and "error" not in static_result:
            logger.info("Found bug location through static analysis")
            static_result["issue_id"] = issue_id
            return static_result

        # Fallback method
        logger.info("Using fallback method: inferring from description")
        result = self._infer_from_description(issue, python_files)
        result["issue_id"] = issue_id
        return result

    def _index_repository(self, repo_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Create an index of all Python files in the repository.

        Args:
            repo_path: Path to the repository

        Returns:
            Dictionary mapping file paths to file information
        """
        python_files = {}

        try:
            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and
                           d not in ('venv', 'env', '.env', '__pycache__')]

                for file in files:
                    if file.endswith('.py'):
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, repo_path)
                        rel_path = rel_path.replace('\\', '/')

                        python_files[rel_path] = {
                            "path": rel_path,
                            "full_path": abs_path,
                            "is_test": self._is_test_file(rel_path)
                        }

            return python_files

        except Exception as e:
            logger.error(f"Error indexing repository: {e}")
            return {}

    def _is_test_file(self, file_path: str) -> bool:
        """Determine if a file is a test file based on naming patterns."""
        if not file_path:
            return False

        file_path = file_path.lower()
        file_name = os.path.basename(file_path)

        return (
                file_name.startswith('test_') or
                file_name.endswith('_test.py') or
                '/test/' in file_path or
                '/tests/' in file_path or
                'testing' in file_path or
                '/conftest.py' in file_path
        )

    def _get_failing_tests(self, issue: Dict[str, Any]) -> List[str]:
        """
        Extract failing tests from the issue.

        Args:
            issue: Issue dictionary

        Returns:
            List of failing test identifiers
        """
        fail_to_pass = self.data_loader.get_fail_to_pass_tests(issue)

        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse FAIL_TO_PASS as JSON: {fail_to_pass}")
                fail_to_pass = []

        return fail_to_pass

    def _get_implementation_file(self, file_path: str) -> Optional[str]:
        """
        Convert a test file path to its implementation file path.

        Args:
            file_path: Path to the file (possibly a test file)

        Returns:
            Implementation file path or None if not determinable
        """
        # If not a test file, return as is
        if not '/tests/' in file_path and not '/test/' in file_path:
            return file_path

        # Handle file in tests directory without test_ prefix
        if '/tests/' in file_path:
            base_name = os.path.basename(file_path)
            if not base_name.startswith('test_'):
                # Convert to implementation path
                impl_path = file_path.replace('/tests/', '/')
                return impl_path

        # Handle test_ prefix
        if '/tests/' in file_path:
            impl_path = file_path.replace('/tests/', '/')
            base_name = os.path.basename(impl_path)
            if base_name.startswith('test_'):
                impl_path = os.path.join(os.path.dirname(impl_path), base_name[5:])
            return impl_path

        if '/test/' in file_path:
            impl_path = file_path.replace('/test/', '/')
            base_name = os.path.basename(impl_path)
            if base_name.startswith('test_'):
                impl_path = os.path.join(os.path.dirname(impl_path), base_name[5:])
            return impl_path

        return file_path

    def _extract_keywords_from_description(self, description: str) -> List[str]:
        """
        Extract important keywords from the issue description.

        Args:
            description: Issue description

        Returns:
            List of keywords
        """
        # Remove code blocks
        description_clean = re.sub(r'```.*?```', '', description, flags=re.DOTALL)

        # Extract code elements (camelCase and snake_case)
        code_pattern = r'\b([A-Z][a-z]+[A-Z][a-zA-Z]*|[a-z]+_[a-z_]+)\b'
        code_elements = re.findall(code_pattern, description_clean)

        # Extract quoted terms
        quoted_pattern = r'[\'"]([^\'"]+)[\'"]'
        quoted_elements = re.findall(quoted_pattern, description_clean)

        # Extract filenames
        file_pattern = r'\b(\w+\.\w+)\b'
        file_elements = re.findall(file_pattern, description_clean)

        # Important indicator words
        important_words = ["error", "bug", "issue", "fail", "incorrect", "wrong", "fix"]

        # Combine all keywords
        keywords = set(code_elements + quoted_elements + file_elements + important_words)

        # Filter short words
        return [k for k in keywords if len(k) >= 3]

    def _get_file_content(self, file_path: str, issue: Dict[str, Any]) -> Optional[str]:
        """
        Get the content of a file.

        Args:
            file_path: Path to the file
            issue: Issue dictionary for repository context

        Returns:
            File content or None if not found
        """
        try:
            repo_path = self.repo_base_path / issue.get("repo", "")
            file_path = file_path.replace('\\', '/')

            # Fix invalid test paths
            if '/tests/' in file_path and not os.path.basename(file_path).startswith('test_'):
                impl_file = self._get_implementation_file(file_path)
                if impl_file:
                    file_path = impl_file

            # Try direct path
            full_path = repo_path / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()

            # Try to find the file
            base_name = os.path.basename(file_path)
            candidates = []

            for root, _, files in os.walk(repo_path):
                if base_name in files:
                    candidates.append(Path(root) / base_name)

            if candidates:
                with open(candidates[0], 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()

            return None

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    def _find_potential_functions(self, file_content: str, description: str) -> List[Dict[str, Any]]:
        """
        Find potential functions in a file that might be related to a bug.
        Args:
            file_content: Content of the file
            description: Issue description
        Returns:
            List of function dictionaries
        """
        potential_functions = []
        # Extract keywords from description
        keywords = self._extract_keywords_from_description(description)
        # Find all functions - fixed regex pattern
        func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):'
        for match in re.finditer(func_pattern, file_content):
            func_name = match.group(1)
            start_pos = match.start()
            start_line = file_content[:start_pos].count('\n') + 1
            # Parse function body to find the end
            lines = file_content[start_pos:].split('\n')
            end_line = start_line
            in_function = False
            base_indent = None

            # Find the indentation of the function definition
            func_def_indent = 0
            if lines and lines[0].strip():
                func_def_indent = len(lines[0]) - len(lines[0].lstrip())

            for i, line in enumerate(lines):
                if i == 0:  # Function definition line
                    in_function = True
                    # Find indentation in the next non-empty line
                    for next_line in lines[1:]:
                        if next_line.strip():
                            base_indent = len(next_line) - len(next_line.lstrip())
                            break
                    continue

                if not line.strip():  # Skip empty lines
                    end_line += 1
                    continue

                curr_indent = len(line) - len(line.lstrip())

                # End of function when indentation returns to same or less than function definition
                if in_function and curr_indent <= func_def_indent:
                    break

                end_line += 1

            # Limit function size to 30 lines
            if end_line - start_line > 30:
                end_line = start_line + 30

            # Score function relevance
            score = 0
            # Function name matches keywords
            for keyword in keywords:
                if keyword.lower() in func_name.lower():
                    score += 2
            # Function body contains keywords
            func_body = '\n'.join(lines[:end_line - start_line + 1])
            for keyword in keywords:
                if keyword.lower() in func_body.lower():
                    score += 1
            # Error-prone patterns
            error_keywords = ["error", "exception", "raise", "assert", "fail"]
            for keyword in error_keywords:
                if keyword in func_body.lower():
                    score += 1
            # Complex control flow
            if "if" in func_body and ("else" in func_body or "elif" in func_body):
                score += 0.5
            if "try" in func_body and "except" in func_body:
                score += 0.5
            potential_functions.append({
                "name": func_name,
                "start_line": start_line,
                "end_line": end_line,
                "score": score
            })
        # Sort by score
        return sorted(potential_functions, key=lambda x: x.get("score", 0), reverse=True)

    def _extract_code_snippet(self, file_path: str, line_range: str, repo_path: Path) -> str:
        """
        Extract code snippet for the given file and line range.

        Args:
            file_path: Path to the file
            line_range: Line range (e.g., "10-20")
            repo_path: Path to the repository

        Returns:
            Code snippet as string
        """
        try:
            # Fix invalid test file paths
            if '/tests/' in file_path and not os.path.basename(file_path).startswith('test_'):
                impl_file = self._get_implementation_file(file_path)
                if impl_file:
                    file_path = impl_file

            # Check if file exists
            full_path = repo_path / file_path
            if not full_path.exists():
                # If it's a test file, try to find the implementation file
                if self._is_test_file(file_path):
                    impl_file = self._infer_implementation_file_from_test(file_path, repo_path)
                    if impl_file:
                        logger.info(f"Using implementation file {impl_file} instead of test file {file_path}")
                        return self._extract_code_snippet(impl_file, line_range, repo_path)

                # Try to find a similar file
                file_name = os.path.basename(file_path)
                similar_files = []

                for root, _, files in os.walk(repo_path):
                    if file_name in files:
                        rel_path = os.path.relpath(os.path.join(root, file_name), repo_path)
                        rel_path = rel_path.replace('\\', '/')
                        if not self._is_test_file(rel_path):  # Prefer non-test files
                            similar_files.append(rel_path)

                if similar_files:
                    logger.info(f"File {file_path} not found, using {similar_files[0]}")
                    full_path = repo_path / similar_files[0]
                else:
                    return f"File not found: {file_path}"

            # Read file content
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()

            # Parse line range
            if '-' in line_range:
                parts = line_range.split('-')
                start_line = int(parts[0])
                end_line = int(parts[1])
            else:
                start_line = int(line_range)
                end_line = start_line + 10  # Default to 10 lines for context

            # Ensure lines are within bounds
            start_line = max(1, start_line)
            end_line = min(len(all_lines) + 1, end_line)

            # Sanity check: reasonable line range (max 30 lines)
            if end_line - start_line > 30:
                end_line = start_line + 30

            # Extract snippet
            snippet_lines = all_lines[start_line - 1:end_line - 1]
            return ''.join(snippet_lines)

        except Exception as e:
            logger.error(f"Error extracting code snippet: {e}")
            return f"Error extracting code: {str(e)}"

    def _infer_implementation_file_from_test(self, test_file_path: str, repo_path: Path) -> Optional[str]:
        """
        Infer the implementation file path from a test file path.

        Args:
            test_file_path: Path to the test file
            repo_path: Path to the repository

        Returns:
            Path to the implementation file or None if not found
        """
        if not test_file_path or not self._is_test_file(test_file_path):
            return None

        # Normalize path
        test_file_path = test_file_path.replace('\\', '/')
        base_name = os.path.basename(test_file_path)
        parent_dir = os.path.dirname(test_file_path)

        potential_paths = []

        # Handle test_ prefix
        if base_name.startswith('test_'):
            impl_name = base_name[5:]  # Remove 'test_' prefix

            # Same directory, just remove prefix
            potential_paths.append(os.path.join(parent_dir, impl_name))

            # Move from tests directory to parent
            if '/tests/' in parent_dir:
                impl_dir = parent_dir.replace('/tests/', '/')
                potential_paths.append(os.path.join(impl_dir, impl_name))

                # Module directory
                module_dir = parent_dir.split('/tests/')[0]
                potential_paths.append(os.path.join(module_dir, impl_name))

            # Special case for astropy
            if 'astropy/modeling/tests/' in test_file_path:
                impl_path = test_file_path.replace('astropy/modeling/tests/', 'astropy/modeling/')
                impl_path = impl_path.replace('test_', '')
                potential_paths.append(impl_path)

        # Check if any potential path exists
        for path in potential_paths:
            if (repo_path / path).exists():
                return path

        return None

    def _perform_static_analysis(self, issue: Dict[str, Any], python_files: Dict[str, Dict[str, Any]]) -> Dict[
        str, Any]:
        """
        Perform static analysis to identify potential bug locations.

        Args:
            issue: Issue dictionary
            python_files: Dictionary of Python files

        Returns:
            Dictionary with bug location information
        """
        logger.info("Starting static analysis")
        # Extract description and implementation files
        description = self.data_loader.get_issue_description(issue)
        keywords = self._extract_keywords_from_description(description)
        logger.info(f"Extracted keywords: {keywords}")

        # Find implementation files
        implementation_files = []

        # From failing tests
        failing_tests = self._get_failing_tests(issue)
        logger.info(f"Using failing tests to infer implementation files: {failing_tests}")
        for test in failing_tests:
            if "::" in test:
                test_file = test.split("::")[0]
                impl_file = self._infer_implementation_file_from_test(
                    test_file,
                    self.repo_base_path / issue.get("repo", "")
                )
                if impl_file and impl_file not in implementation_files:
                    logger.info(f"Inferred implementation file {impl_file} from test {test_file}")
                    implementation_files.append(impl_file)

        # If still no implementation files, use non-test files
        if not implementation_files:
            logger.info("No implementation files found, searching based on keywords")
            for file_path in python_files:
                if not python_files[file_path]["is_test"]:
                    # Check for keyword matches in path
                    matches = sum(1 for kw in keywords if kw.lower() in file_path.lower())
                    if matches > 0:
                        implementation_files.append(file_path)

            # Sort by likelihood
            implementation_files.sort(key=lambda f: sum(1 for kw in keywords if kw.lower() in f.lower()), reverse=True)
            implementation_files = implementation_files[:5]  # Limit to top 5
            logger.info(f"Found implementation files based on keywords: {implementation_files}")

        # Fix any invalid test paths
        fixed_impl_files = []
        for file_path in implementation_files:
            if '/tests/' in file_path and not os.path.basename(file_path).startswith('test_'):
                impl_file = self._get_implementation_file(file_path)
                if impl_file:
                    fixed_impl_files.append(impl_file)
                else:
                    fixed_impl_files.append(file_path)
            else:
                fixed_impl_files.append(file_path)

        implementation_files = fixed_impl_files
        logger.info(f"Final implementation files for analysis: {implementation_files}")

        # Analyze implementation files
        potential_locations = []

        for file_path in implementation_files:
            logger.info(f"Analyzing file: {file_path}")
            file_content = self._get_file_content(file_path, issue)
            if not file_content:
                logger.warning(f"Could not read content of file {file_path}")
                continue

            # Find potential functions
            functions = self._find_potential_functions(file_content, description)
            logger.info(f"Found {len(functions)} potential functions in {file_path}")

            if functions:
                for function in functions[:2]:  # Top 2 functions only
                    confidence = 0.6  # Higher confidence for implementation files
                    confidence += min(0.2, function.get("score", 0) * 0.1)

                    potential_locations.append({
                        "file": file_path,
                        "function": function["name"],
                        "line_numbers": f"{function['start_line']}-{function['end_line']}",
                        "confidence": min(1.0, confidence)
                    })

        # Sort by confidence
        potential_locations.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        logger.info(f"Found {len(potential_locations)} potential bug locations through static analysis")

        # Return top location if found
        if potential_locations:
            top_location = potential_locations[0]
            repo_path = self.repo_base_path / issue.get("repo", "")

            # Add code_content to primary location
            top_location["code_content"] = self._extract_code_snippet(
                top_location["file"],
                top_location.get("line_numbers", ""),
                repo_path
            )

            # Add alternatives with deduplication
            alternatives = []
            seen_keys = set()

            # Create a unique key for the primary location to avoid duplicates
            primary_key = (
                top_location["file"],
                top_location.get("function", ""),
                top_location.get("line_numbers", "")
            )
            seen_keys.add(primary_key)

            # Process other locations for alternatives
            for loc in potential_locations[1:]:
                current_key = (
                    loc["file"],
                    loc.get("function", ""),
                    loc.get("line_numbers", "")
                )

                # Skip if already seen
                if current_key in seen_keys:
                    continue

                seen_keys.add(current_key)

                # Add code_content to alternative
                alternative = {
                    "file": loc["file"],
                    "function": loc.get("function"),
                    "line_numbers": loc.get("line_numbers"),
                    "confidence": loc.get("confidence", 0),
                    "code_content": self._extract_code_snippet(
                        loc["file"],
                        loc.get("line_numbers", ""),
                        repo_path
                    )
                }
                alternatives.append(alternative)

                # Limit to 5 alternatives
                if len(alternatives) >= 5:
                    break

            top_location["alternative_locations"] = alternatives
            logger.info(f"Returning top location with {len(alternatives)} unique alternatives")
            return top_location

        logger.warning("No potential bug locations found through static analysis")
        return None

    def _infer_from_description(self, issue: Dict[str, Any], python_files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Infer bug location from issue description.

        Args:
            issue: Issue dictionary
            python_files: Dictionary of Python files

        Returns:
            Dictionary with bug location information
        """
        repo_path = self.repo_base_path / issue.get("repo", "")
        description = self.data_loader.get_issue_description(issue)

        # Try to extract file and function information
        extracted_info = self._extract_file_info_from_description(description)

        # If no file found, try more aggressive methods
        if not extracted_info.get("file"):
            keywords = self._extract_keywords_from_description(description)

            # Score files based on keywords
            scored_files = []
            for file_path in python_files:
                # Skip test files
                if python_files[file_path]["is_test"]:
                    continue

                # Count keyword matches
                matches = sum(1 for kw in keywords if kw.lower() in file_path.lower())
                if matches > 0:
                    scored_files.append((file_path, matches))

            # Sort and select top match
            if scored_files:
                scored_files.sort(key=lambda x: x[1], reverse=True)
                extracted_info["file"] = scored_files[0][0]

        # If we still don't have a file, return error
        if not extracted_info.get("file"):
            return {"error": "Could not identify file location from issue description"}

        # Get file content
        file_path = extracted_info["file"]

        # Fix invalid test paths
        if '/tests/' in file_path and not os.path.basename(file_path).startswith('test_'):
            impl_file = self._get_implementation_file(file_path)
            if impl_file:
                file_path = impl_file
                extracted_info["file"] = impl_file

        file_content = self._get_file_content(file_path, issue)

        if not file_content:
            return {"error": f"Could not read file {file_path}"}

        # Find functions if not specified
        function_name = extracted_info.get("function")
        if not function_name:
            functions = self._find_potential_functions(file_content, description)
            if functions:
                function_name = functions[0]["name"]

        # Get function info and line numbers
        if function_name:
            # Try to use AST parsing to get function boundaries
            try:
                tree = ast.parse(file_content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == function_name:
                        start_line = node.lineno
                        end_line = start_line
                        for child in ast.walk(node):
                            if hasattr(child, 'lineno'):
                                end_line = max(end_line, child.lineno)

                        # Add a small margin and limit to reasonable size
                        end_line = min(end_line + 2, file_content.count('\n') + 1)
                        if end_line - start_line > 30:
                            end_line = start_line + 30

                        line_numbers = f"{start_line}-{end_line}"
                        lines = file_content.split('\n')
                        code_content = '\n'.join(lines[start_line - 1:end_line - 1])
                        break
                else:
                    # Function not found with AST, use regex fallback
                    func_pattern = rf'def\s+{re.escape(function_name)}\s*\([^)]*\):'
                    match = re.search(func_pattern, file_content)
                    if match:
                        start_pos = match.start()
                        start_line = file_content[:start_pos].count('\n') + 1
                        end_line = start_line + 20  # Approximate function length
                        line_numbers = f"{start_line}-{end_line}"
                        lines = file_content.split('\n')
                        code_content = '\n'.join(lines[start_line - 1:end_line - 1])
                    else:
                        # No function found, use first 30 lines
                        line_numbers = "1-30"
                        lines = file_content.split('\n')
                        code_content = '\n'.join(lines[:min(30, len(lines))])
            except SyntaxError:
                # AST parsing failed, use first 30 lines
                line_numbers = "1-30"
                lines = file_content.split('\n')
                code_content = '\n'.join(lines[:min(30, len(lines))])
        else:
            # No function specified, use first 30 lines
            line_numbers = "1-30"
            lines = file_content.split('\n')
            code_content = '\n'.join(lines[:min(30, len(lines))])

        # Return result
        return {
            "file": file_path,
            "function": function_name,
            "line_numbers": line_numbers,
            "code_content": code_content,
            "confidence": 0.5,  # Moderate confidence
            "alternative_locations": []
        }

    def _extract_file_info_from_description(self, description: str) -> Dict[str, Any]:
        """
        Extract file and function information from issue description.

        Args:
            description: Issue description text

        Returns:
            Dictionary with file and function information
        """
        info = {
            "file": None,
            "function": None
        }

        # Look for file references
        file_patterns = [
            r'file[:\s]+([\/\w\-\.]+\.py)',
            r'in\s+([\/\w\-\.]+\.py)',
            r'([\/\w\-\.]+\.py):\d+',
            r'([\/\w\-\.]+\.py)\s+line',
            r'bug\s+in\s+([\/\w\-\.]+\.py)',
            r'"([\/\w\-\.]+\.py)"',
            r"'([\/\w\-\.]+\.py)'",
            r"`([\/\w\-\.]+\.py)`"
        ]

        for pattern in file_patterns:
            matches = re.findall(pattern, description)
            if matches:
                info["file"] = matches[0]
                break

        # If no Python file found, try any file
        if not info["file"]:
            any_file_pattern = r'([\w\-\.\/]+\.\w+)'
            matches = re.findall(any_file_pattern, description)
            if matches:
                python_files = [f for f in matches if f.endswith('.py')]
                if python_files:
                    info["file"] = python_files[0]
                else:
                    info["file"] = matches[0]

        # Look for function references
        function_patterns = [
            r'function[:\s]+(\w+)',
            r'method[:\s]+(\w+)',
            r'in\s+(\w+)\s*\(',
            r'(\w+)\s*\(\).*fails',
            r'bug\s+in\s+(\w+)\s*\(',
            r'"(\w+)".*function',
            r'\'(\w+)\'.*function',
            r'`(\w+)`.*function'
        ]

        for pattern in function_patterns:
            matches = re.findall(pattern, description)
            if matches:
                info["function"] = matches[0]
                break

        return info
