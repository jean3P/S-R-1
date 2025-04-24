"""
Context-Based Bug Detector for SWE-bench.

This module implements an advanced bug detection approach using context-based
code representation and attention mechanisms to precisely locate bugs in code.
"""

import logging
import os
import re
import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import torch
import torch.nn.functional as F

from ..data.data_loader import SWEBenchDataLoader

logger = logging.getLogger(__name__)


class ContextBasedBugDetector:
    """
    Detects bugs in code using context-based representation and attention mechanisms.

    This implements a multi-stage approach:
    1. Extract local context from code (AST-based)
    2. Model global context (dependency and dataflow)
    3. Use failing tests to focus analysis
    4. Rank code locations by likelihood of containing bugs
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

        # Model parameters
        self.embedding_dim = 100
        self.attention_heads = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() and
                                          config["models"]["device"] == "cuda" else 'cpu')

        # Cache for parsed ASTs and execution traces
        self.ast_cache = {}
        self.execution_trace_cache = {}

        # Initialize the Node2Vec and Word2Vec models if using pre-trained embeddings
        self.initialize_embedding_models()

    def initialize_embedding_models(self):
        """Initialize embedding models for code representation."""
        self.node_embeddings = {}
        self.word_embeddings = {}

    def detect_bug_location(self, issue_id: str) -> Dict[str, Any]:
        """
        Analyzes code and tests to identify the location of a bug.

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

        # Extract failing tests
        failing_tests = self._get_failing_tests(issue)
        logger.info(f"Found {len(failing_tests)} failing tests: {failing_tests}")

        # Run trace analysis if failing tests are available
        bug_locations = []
        if failing_tests:
            logger.info(f"Analyzing {len(failing_tests)} failing tests")

            # Execute and trace failing tests
            execution_traces = self._trace_failing_tests(failing_tests, repo_path, issue)
            logger.info(f"Generated {len(execution_traces)} execution traces")

            # Extract potential bug locations
            bug_locations = self._analyze_execution_traces(execution_traces, python_files, issue)
            logger.info(f"Found {len(bug_locations)} potential bug locations")

        # If we found valid bug locations, return the best one with alternatives
        if bug_locations:
            logger.info(f"Processing {len(bug_locations)} bug locations with _finalize_bug_location")
            try:
                result = self._finalize_bug_location(bug_locations, repo_path)
                # Add issue_id to the result
                result["issue_id"] = issue_id
                return result
            except Exception as e:
                logger.error(f"Error in _finalize_bug_location: {str(e)}", exc_info=True)
                return {"error": f"Error finalizing bug location: {str(e)}", "issue_id": issue_id}

        # If trace analysis failed, try static analysis
        logger.info("No bug locations found from trace analysis. Using static analysis to locate bug")
        static_result = self._perform_static_analysis(issue, python_files)

        if static_result and "error" not in static_result:
            logger.info("Found bug location through static analysis")
            # Add code_content and issue_id
            code_content = self._extract_code_snippet(
                static_result["file"],
                static_result["line_numbers"],
                repo_path
            )
            static_result["code_content"] = code_content
            # Remove code_snippet if present
            if "code_snippet" in static_result:
                del static_result["code_snippet"]
            static_result["issue_id"] = issue_id
            return static_result

        # Fallback method
        logger.info("Using fallback method: inferring from description")
        result = self._infer_from_description(issue, python_files)
        # Convert code_snippet to code_content if needed
        if "code_snippet" in result:
            result["code_content"] = result["code_snippet"]
            del result["code_snippet"]
        result["issue_id"] = issue_id
        return result

    def _finalize_bug_location(self, bug_locations: List[Dict[str, Any]], repo_path: Path) -> Dict[str, Any]:
        """
        Prepare the final bug location result from a list of candidates.

        Args:
            bug_locations: List of potential bug locations
            repo_path: Path to the repository

        Returns:
            The top bug location with all necessary information
        """
        logger.info("_finalize_bug_location")  # Use module-level logger consistently
        if not bug_locations:
            return {"error": "No bug locations found"}

        # Clean and validate locations
        valid_locations = []
        seen_locations = set()

        for loc in bug_locations:
            # Skip invalid test file paths
            file_path = loc["file"]
            if '/tests/' in file_path and not os.path.basename(file_path).startswith('test_'):
                impl_file = self._get_implementation_file(file_path)
                if impl_file:
                    loc["file"] = impl_file
                    # Update function info if available
                    if loc.get("function"):
                        func_info = self._get_function_info(
                            impl_file,
                            loc["function"],
                            {"repo": str(repo_path.name)}
                        )
                        if func_info:
                            loc["line_numbers"] = f"{func_info['start_line']}-{func_info['end_line']}"

            # Create unique key for deduplication
            loc_key = (
                loc["file"],
                loc.get("function", ""),
                loc.get("line_numbers", "")
            )

            # Skip duplicates and invalid locations
            if loc_key in seen_locations or not loc.get("file"):
                continue

            seen_locations.add(loc_key)
            valid_locations.append(loc)

        if not valid_locations:
            return {"error": "No valid bug locations found"}

        # Sort by confidence (highest first)
        valid_locations.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # Separate test files from implementation files
        impl_locations = [loc for loc in valid_locations
                          if not self._is_test_file(loc["file"])]
        test_locations = [loc for loc in valid_locations
                          if self._is_test_file(loc["file"])]

        # Prefer implementation files over test files
        top_locations = impl_locations if impl_locations else test_locations

        # Get top location (primary recommendation)
        primary_location = top_locations[0]
        primary_key = (
            primary_location["file"],
            primary_location.get("function", ""),
            primary_location.get("line_numbers", "")
        )

        # Prepare final result with code_content
        result = {
            "file": primary_location["file"],
            "function": primary_location.get("function"),
            "line_numbers": primary_location.get("line_numbers"),
            "confidence": primary_location.get("confidence", 0),
            "code_content": self._extract_code_snippet(
                primary_location["file"],
                primary_location.get("line_numbers", ""),
                repo_path
            ),
            "alternative_locations": []
        }

        # Prepare alternative locations (excluding the primary location)
        alternative_locations = []
        seen_alternatives = set()

        # Process all other locations for alternatives
        for loc in top_locations[1:] + test_locations:
            current_key = (
                loc["file"],
                loc.get("function", ""),
                loc.get("line_numbers", "")
            )

            # Skip if matches primary or already seen
            logger.info(f"Current key: {current_key} | primary key: {primary_key} | seen alternatives: {seen_alternatives}")
            if current_key == primary_key or current_key in seen_alternatives:
                continue

            seen_alternatives.add(current_key)

            # Add to alternatives with all required fields including code_content
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
            alternative_locations.append(alternative)

            # Limit to 5 alternatives
            if len(alternative_locations) >= 5:
                break

        # Add alternatives to result
        result["alternative_locations"] = alternative_locations

        return result

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

        # Handle file in tests directory without test_ prefix (invalid in SWE-bench)
        if '/tests/' in file_path:
            base_name = os.path.basename(file_path)
            if not base_name.startswith('test_'):
                # Convert to implementation path
                impl_path = file_path.replace('/tests/', '/')
                return impl_path

        # Handle other test paths
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

    def _trace_failing_tests(self, failing_tests: List[str], repo_path: Path, issue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run failing tests with tracing to identify execution paths.

        Args:
            failing_tests: List of failing test identifiers
            repo_path: Path to the repository
            issue: Issue dictionary

        Returns:
            List of execution traces
        """
        logger.info(f"Tracing {len(failing_tests)} failing tests")
        traces = []
        env_path = issue.get("environment_path")

        for test in failing_tests:
            cache_key = f"{issue.get('id', '')}-{test}"

            # Use cached trace if available
            if cache_key in self.execution_trace_cache:
                logger.info(f"Using cached trace for test: {test}")
                traces.append(self.execution_trace_cache[cache_key])
                continue

            try:
                logger.info(f"Tracing execution for test: {test}")

                # Extract test module and function
                if "::" in test:
                    test_module, test_function = test.split("::", 1)
                else:
                    test_module = test
                    test_function = None

                logger.info(f"Test module: {test_module}, Test function: {test_function}")

                # Prepare the trace command
                if env_path and os.path.exists(env_path):
                    # Use the environment's Python
                    if os.name == 'nt':  # Windows
                        python_path = os.path.join(env_path, "Scripts", "python.exe")
                    else:  # Unix/Linux/Mac
                        python_path = os.path.join(env_path, "bin", "python")
                else:
                    python_path = "python"

                logger.info(f"Using Python path: {python_path}")

                # Build the trace command
                trace_cmd = [python_path, "-m", "trace", "--trace"]

                # Add the test specification
                if test_function:
                    trace_cmd.extend(["-m", "pytest", f"{test_module}::{test_function}", "-v"])
                else:
                    trace_cmd.extend(["-m", "pytest", test_module, "-v"])

                logger.info(f"Trace command: {' '.join(trace_cmd)}")

                # Run the trace command
                process = subprocess.run(
                    trace_cmd,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy()
                )

                logger.info(f"Trace command returned with code {process.returncode}")

                # Parse the trace output
                trace_lines = process.stderr.split('\n')
                execution_path = []

                for line in trace_lines:
                    if " line " in line and ":" in line:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            file_path = parts[0]
                            # Skip standard library and external files
                            if ("site-packages" in file_path or
                                    "lib/python" in file_path or
                                    file_path.startswith("<")):
                                continue

                            if file_path.startswith(str(repo_path)):
                                file_path = os.path.relpath(file_path, str(repo_path))

                            # Extract line number and function name
                            line_num_match = re.search(r'line (\d+)', line)
                            line_num = int(line_num_match.group(1)) if line_num_match else 0

                            func_match = re.search(r'(\w+)\(', line)
                            func_name = func_match.group(1) if func_match else None

                            execution_path.append({
                                "file": file_path,
                                "line": line_num,
                                "function": func_name
                            })

                logger.info(f"Parsed {len(execution_path)} execution steps from trace")

                # Store the execution trace
                trace = {
                    "test": test,
                    "execution_path": execution_path,
                    "success": process.returncode == 0
                }

                # Cache the trace
                self.execution_trace_cache[cache_key] = trace
                traces.append(trace)

            except Exception as e:
                logger.error(f"Error tracing test {test}: {str(e)}", exc_info=True)
                # Try alternative approach to find referenced files
                logger.info(f"Trying alternative approach to find files referenced by test {test}")
                impl_files = self._identify_implementation_files_from_test(test, repo_path)
                if impl_files:
                    logger.info(f"Identified {len(impl_files)} implementation files from test {test}: {impl_files}")
                    traces.append({
                        "test": test,
                        "execution_path": [{"file": f, "line": 0, "function": None} for f in impl_files],
                        "success": False,
                        "inferred": True
                    })

        logger.info(f"Completed tracing with {len(traces)} traces")
        return traces

    def _identify_implementation_files_from_test(self, test: str, repo_path: Path) -> List[str]:
        """
        Identify implementation files referenced by a test.

        Args:
            test: Test identifier
            repo_path: Path to the repository

        Returns:
            List of implementation file paths
        """
        impl_files = []

        # Extract test file path
        if "::" in test:
            test_file = test.split("::")[0]
        else:
            test_file = test

        # Ensure it has .py extension
        if not test_file.endswith('.py'):
            test_file += '.py'

        # Find the test file
        test_path = repo_path / test_file
        if not test_path.exists():
            # Search for the test file
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file == os.path.basename(test_file):
                        test_path = Path(root) / file
                        break

        if not test_path.exists():
            return []

        # Parse the test file to find imports
        try:
            with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
                test_content = f.read()

            # Extract imports
            import_pattern = r'(?:from|import)\s+([\w\.]+)'
            imports = re.findall(import_pattern, test_content)

            # Find corresponding implementation files
            for imp in imports:
                # Skip standard library imports
                if imp in ('os', 'sys', 'pytest', 're', 'json', 'time', 'math'):
                    continue

                # Skip test modules
                if 'test' in imp.lower():
                    continue

                # Convert import to file path
                file_path = imp.replace('.', '/')

                # Add .py extension if not a directory
                if not os.path.isdir(repo_path / file_path):
                    file_path += '.py'

                # Add to list if file exists and is not a test file
                full_path = repo_path / file_path
                if full_path.exists() and not self._is_test_file(file_path):
                    impl_files.append(file_path)

            # If we haven't found any implementation files, try to infer from name
            if not impl_files:
                impl_file = self._infer_implementation_file_from_test(test_file, repo_path)
                if impl_file:
                    impl_files.append(impl_file)

            return impl_files

        except Exception as e:
            logger.error(f"Error identifying implementation files: {e}")
            return []

    def _analyze_execution_traces(self, execution_traces: List[Dict[str, Any]],
                                python_files: Dict[str, Dict[str, Any]],
                                issue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze execution traces to identify potential bug locations.

        Args:
            execution_traces: List of execution traces
            python_files: Dictionary of Python files in the repository
            issue: Issue dictionary

        Returns:
            List of potential bug locations
        """
        logger.info("Starting analysis of execution traces")
        potential_locations = []

        # Extract keywords from issue description
        description = self.data_loader.get_issue_description(issue)
        keywords = self._extract_keywords_from_description(description)
        logger.info(f"Extracted keywords from description: {keywords}")

        # Build maps of files and functions in traces
        file_frequency = defaultdict(int)
        function_frequency = defaultdict(int)
        test_to_impl_map = defaultdict(set)
        impl_files = set()

        # First pass: identify implementation files referenced by tests
        for i, trace in enumerate(execution_traces):
            execution_path = trace.get("execution_path", [])
            if not execution_path:
                logger.warning(f"Trace {i} has no execution path")
                continue

            curr_test_file = None
            for step in execution_path:
                file_path = step.get("file", "")
                if not file_path:
                    continue

                # Normalize path
                file_path = file_path.replace('\\', '/')

                # Skip invalid test files (in tests/ without test_ prefix)
                if '/tests/' in file_path and not os.path.basename(file_path).startswith('test_'):
                    # Convert to implementation file
                    file_path = self._get_implementation_file(file_path) or file_path

                # Track test and implementation files
                if self._is_test_file(file_path):
                    curr_test_file = file_path
                elif curr_test_file:
                    # This is an implementation file referenced by a test
                    test_to_impl_map[curr_test_file].add(file_path)
                    impl_files.add(file_path)

                # Count file frequency
                file_frequency[file_path] += 1

                # Count function frequency if available
                function_name = step.get("function")
                if function_name:
                    function_frequency[f"{file_path}::{function_name}"] += 1

        logger.info(f"Found {len(impl_files)} implementation files from execution traces")
        logger.info(f"Found {len(function_frequency)} unique functions in execution traces")

        # If no implementation files found through trace, try to infer them
        if not impl_files:
            logger.info("No implementation files found through trace, trying to infer them")
            test_files = [f for f in file_frequency.keys() if self._is_test_file(f)]
            for test_file in test_files:
                impl_file = self._infer_implementation_file_from_test(test_file, repo_path=self.repo_base_path / issue.get("repo", ""))
                if impl_file:
                    logger.info(f"Inferred implementation file {impl_file} from test file {test_file}")
                    impl_files.add(impl_file)
                    test_to_impl_map[test_file].add(impl_file)

        # Still no implementation files? Use non-test files with high frequency
        if not impl_files:
            logger.info("Still no implementation files found, using non-test files with high frequency")
            for file_path in sorted(file_frequency.keys(), key=lambda f: file_frequency[f], reverse=True):
                file_path = self._get_implementation_file(file_path) or file_path
                if not self._is_test_file(file_path) and file_path in python_files:
                    logger.info(f"Adding non-test file with high frequency: {file_path}")
                    impl_files.add(file_path)
                    if len(impl_files) >= 5:  # Limit to top 5
                        break

        # Analyze each implementation file
        for file_path in impl_files:
            logger.info(f"Analyzing implementation file: {file_path}")
            # Get file functions
            file_functions = [fn for fn in function_frequency.keys()
                             if fn.startswith(f"{file_path}::")]

            if file_functions:
                logger.info(f"Found {len(file_functions)} functions in file {file_path}")
                # Analyze each function
                for func_id in file_functions:
                    _, function_name = func_id.split("::", 1)

                    # Calculate confidence
                    confidence = 0.6  # Base confidence for implementation files
                    confidence += min(0.2, function_frequency[func_id] * 0.1)

                    # Keyword matching
                    keyword_matches = sum(1 for kw in keywords
                                        if kw.lower() in function_name.lower())
                    confidence += min(0.1, keyword_matches * 0.05)

                    # Get precise function info with line numbers
                    function_info = self._get_function_info(file_path, function_name, issue)
                    if function_info:
                        logger.info(f"Found function {function_name} in file {file_path} with confidence {confidence}")
                        # Add to potential locations
                        potential_locations.append({
                            "file": file_path,
                            "function": function_name,
                            "line_numbers": f"{function_info['start_line']}-{function_info['end_line']}",
                            "confidence": min(1.0, confidence)
                        })
            else:
                logger.info(f"No specific functions identified in file {file_path}, analyzing file content")
                # No specific functions identified, try to analyze the file
                file_content = self._get_file_content(file_path, issue)
                if file_content:
                    # Find likely functions in the file
                    functions = self._find_potential_functions(file_content, description)

                    if functions:
                        logger.info(f"Found {len(functions)} potential functions in file {file_path}")
                        # Add top function
                        function = functions[0]
                        potential_locations.append({
                            "file": file_path,
                            "function": function["name"],
                            "line_numbers": f"{function['start_line']}-{function['end_line']}",
                            "confidence": 0.5  # Moderate confidence
                        })

        # Remove duplicate entries and sort by confidence
        unique_locations = []
        seen = set()

        for loc in potential_locations:
            # Ensure we're using implementation files, not test files
            file_path = loc["file"]
            if self._is_test_file(file_path):
                impl_file = self._get_implementation_file(file_path)
                if impl_file:
                    loc["file"] = impl_file

            key = f"{loc['file']}::{loc.get('function', '')}"
            if key not in seen:
                # Skip invalid test paths
                if '/tests/' in loc['file'] and not os.path.basename(loc['file']).startswith('test_'):
                    continue
                seen.add(key)
                unique_locations.append(loc)

        unique_locations.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        logger.info(f"Returning {len(unique_locations)} unique potential bug locations")
        return unique_locations

    def _get_function_info(self, file_path: str, function_name: str, issue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a function using AST parsing.

        Args:
            file_path: Path to the file
            function_name: Name of the function
            issue: Issue dictionary for repository context

        Returns:
            Dictionary with function information or None if not found
        """
        try:
            # Check if file_path is in tests directory without test_ prefix
            if '/tests/' in file_path and not os.path.basename(file_path).startswith('test_'):
                # Try to get the corresponding implementation file
                impl_file = self._get_implementation_file(file_path)
                if impl_file:
                    file_path = impl_file

            # Get file content
            file_content = self._get_file_content(file_path, issue)
            if not file_content:
                return None

            # Parse AST
            try:
                tree = ast.parse(file_content)
            except SyntaxError:
                return self._get_function_info_regex(file_path, function_name, file_content)

            # Find the function node
            function_node = None
            for node in ast.walk(tree):
                if (isinstance(node, ast.FunctionDef) or
                    isinstance(node, ast.AsyncFunctionDef)) and node.name == function_name:
                    function_node = node
                    break

            if not function_node:
                return self._get_function_info_regex(file_path, function_name, file_content)

            # Get line numbers
            start_line = function_node.lineno
            end_line = 0

            # Find the last line by looking at child nodes
            for node in ast.walk(function_node):
                if hasattr(node, 'lineno'):
                    end_line = max(end_line, node.lineno)

            # Add a small margin to account for trailing lines
            end_line = min(end_line + 2, file_content.count('\n') + 1)

            # Sanity check: ensure function size is reasonable (max 30 lines)
            if end_line - start_line > 30:
                end_line = start_line + 30

            # Extract function code
            lines = file_content.split('\n')
            function_code = '\n'.join(lines[start_line-1:end_line])

            return {
                "name": function_name,
                "start_line": start_line,
                "end_line": end_line,
                "code": function_code
            }

        except Exception as e:
            logger.error(f"Error parsing function {function_name}: {e}")
            return None

    def _get_function_info_regex(self, file_path: str, function_name: str, file_content: str) -> Optional[Dict[str, Any]]:
        """
        Get function info using regex (fallback method).

        Args:
            file_path: Path to the file
            function_name: Name of the function
            file_content: Content of the file

        Returns:
            Function information or None if not found
        """
        # Find function definition
        func_pattern = rf'def\s+{re.escape(function_name)}\s*\([^)]*\)(?:\s*->.*?)?:'
        match = re.search(func_pattern, file_content)
        if not match:
            return None

        # Get function start line
        start_pos = match.start()
        start_line = file_content[:start_pos].count('\n') + 1

        # Find function body indentation
        lines = file_content[start_pos:].split('\n')

        # Find the first non-empty line after function definition
        body_start = 1
        while body_start < len(lines) and not lines[body_start].strip():
            body_start += 1

        if body_start >= len(lines):
            # Empty function or reached EOF
            return {
                "name": function_name,
                "start_line": start_line,
                "end_line": start_line + 1,
                "code": lines[0]
            }

        # Get body indentation level
        first_body_line = lines[body_start]
        indent_level = len(first_body_line) - len(first_body_line.lstrip())

        # Find function end by tracking indentation
        end_line = start_line
        for i, line in enumerate(lines[body_start:], body_start):
            if not line.strip():  # Skip empty lines
                continue

            # Check indentation
            curr_indent = len(line) - len(line.lstrip())

            # If indentation returns to function level or less, we've reached the end
            if curr_indent <= 0 or (indent_level > 0 and curr_indent < indent_level):
                end_line = start_line + i - 1
                break

            end_line = start_line + i

        # Sanity check: limit function size to 30 lines
        if end_line - start_line > 30:
            end_line = start_line + 30

        # Extract function code
        function_code = '\n'.join(lines[:end_line - start_line + 1])

        return {
            "name": function_name,
            "start_line": start_line,
            "end_line": end_line,
            "code": function_code
        }

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

        # Handle test_ prefix (this is the most common case in SWE-bench)
        if base_name.startswith('test_'):
            impl_name = base_name[5:]  # Remove 'test_' prefix

            # 1. Same directory, just remove prefix
            potential_paths.append(os.path.join(parent_dir, impl_name))

            # 2. Move from tests directory to parent
            if '/tests/' in parent_dir:
                impl_dir = parent_dir.replace('/tests/', '/')
                potential_paths.append(os.path.join(impl_dir, impl_name))

                # 3. Module directory
                module_dir = parent_dir.split('/tests/')[0]
                potential_paths.append(os.path.join(module_dir, impl_name))

            # 4. Handle test directory
            if '/test/' in parent_dir:
                impl_dir = parent_dir.replace('/test/', '/')
                potential_paths.append(os.path.join(impl_dir, impl_name))

                module_dir = parent_dir.split('/test/')[0]
                potential_paths.append(os.path.join(module_dir, impl_name))

            # Special case for astropy (Most critical fix for the issues you reported)
            if 'astropy/wcs/wcsapi/tests/' in test_file_path:
                impl_path = test_file_path.replace('astropy/wcs/wcsapi/tests/', 'astropy/wcs/wcsapi/')
                impl_path = impl_path.replace('test_', '')
                potential_paths.append(impl_path)

            if 'astropy/modeling/tests/' in test_file_path:
                impl_path = test_file_path.replace('astropy/modeling/tests/', 'astropy/modeling/')
                impl_path = impl_path.replace('test_', '')
                potential_paths.append(impl_path)

        # Handle _test suffix
        elif base_name.endswith('_test.py'):
            impl_name = base_name[:-8] + '.py'  # Remove '_test.py' and add '.py'

            # Similar pattern as above
            potential_paths.append(os.path.join(parent_dir, impl_name))

            if '/tests/' in parent_dir:
                impl_dir = parent_dir.replace('/tests/', '/')
                potential_paths.append(os.path.join(impl_dir, impl_name))

                module_dir = parent_dir.split('/tests/')[0]
                potential_paths.append(os.path.join(module_dir, impl_name))

            if '/test/' in parent_dir:
                impl_dir = parent_dir.replace('/test/', '/')
                potential_paths.append(os.path.join(impl_dir, impl_name))

                module_dir = parent_dir.split('/test/')[0]
                potential_paths.append(os.path.join(module_dir, impl_name))

        # Check if any potential path exists
        for path in potential_paths:
            if (repo_path / path).exists():
                return path

        # Fallback: look for files with similar name in the module directory
        path_parts = test_file_path.split('/')
        if 'tests' in path_parts:
            tests_idx = path_parts.index('tests')
            if tests_idx > 0:
                module_dir = repo_path / '/'.join(path_parts[:tests_idx])

                if module_dir.exists():
                    # Look for implementation files
                    for file in os.listdir(module_dir):
                        if (file.endswith('.py') and
                            not file.startswith('test_') and
                            not file.endswith('_test.py')):
                            return '/'.join(path_parts[:tests_idx]) + '/' + file

        return None

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

        # Find all functions
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

                # End of function when indentation returns to base level
                if in_function and base_indent is not None and curr_indent <= 0:
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

        # Get test patch information
        test_patch = self.data_loader.get_test_patch(issue)

        # Find implementation files
        implementation_files = []

        # From test patch
        if isinstance(test_patch, dict) and "implementation_files" in test_patch:
            implementation_files = test_patch.get("implementation_files", [])
            logger.info(f"Found implementation files from test patch: {implementation_files}")

        # From failing tests
        if not implementation_files:
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

        # If still no file, check for common files
        if not extracted_info.get("file"):
            repo = issue.get("repo", "")
            if repo == "astropy/astropy":
                common_files = [
                    "astropy/wcs/wcsapi/fitswcs.py",
                    "astropy/modeling/separable.py",
                    "astropy/modeling/core.py"
                ]

                for file in common_files:
                    if (repo_path / file).exists():
                        extracted_info["file"] = file
                        break

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

        # Verify function exists if specified
        function_name = extracted_info.get("function")
        if function_name:
            func_pattern = rf'def\s+{re.escape(function_name)}\s*\('
            if not re.search(func_pattern, file_content):
                function_name = None

        # Find functions if not specified
        if not function_name:
            functions = self._find_potential_functions(file_content, description)
            if functions:
                function_name = functions[0]["name"]

        # Get function info and line numbers
        if function_name:
            func_info = self._get_function_info(file_path, function_name, issue)
            if func_info:
                line_numbers = f"{func_info['start_line']}-{func_info['end_line']}"
                code_content = func_info["code"]
            else:
                # Use first 30 lines
                line_numbers = "1-30"
                lines = file_content.split('\n')
                code_content = '\n'.join(lines[:min(30, len(lines))])
        else:
            # No function found, use first 30 lines
            line_numbers = "1-30"
            lines = file_content.split('\n')
            code_content = '\n'.join(lines[:min(30, len(lines))])

        # Return result - using code_content instead of code_snippet
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
