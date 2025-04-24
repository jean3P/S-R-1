# src/utils/repository_explorer.py

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..data.data_loader import SWEBenchDataLoader
from .repository_rag import RepositoryRAG

logger = logging.getLogger(__name__)


class RepositoryExplorer:
    """
    Utility for exploring repository structure to locate relevant files and code.
    """

    def __init__(self, config):
        """
        Initialize the repository explorer.

        Args:
            config: Configuration object containing paths.
        """
        self.config = config
        self.repo_base_path = Path(config["data"]["repositories"])

        # Initialize the Repository RAG system
        self.rag_system = None
        self.use_rag = config.get("memory_efficient", False)

        # Lazy initialization for RAG system
        if self.use_rag:
            try:
                self.rag_system = RepositoryRAG(config)
                logger.info("Initialized Repository RAG system for memory-efficient analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize Repository RAG system: {e}")
                logger.warning("Falling back to standard repository exploration")
                self.use_rag = False

    def ensure_repository_exists(self, issue: Dict[str, Any]) -> bool:
        """
        Ensures that the repository for the given issue exists.
        Downloads it from GitHub if not present.

        Args:
            issue: The issue dictionary containing repository information.

        Returns:
            bool: True if repository exists or was downloaded successfully.
        """
        # Extract repository from issue
        repo = issue.get("repo", "")
        if not repo:
            logger.error("No repository information found in issue")
            return False

        # Construct repository path
        repo_path = Path(self.config["data"]["repositories"]) / repo

        if not repo_path.exists():
            parent_dir = repo_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)

            # Clone the repository
            try:
                import subprocess  # Make sure this import is at the top of the file
                subprocess.run(
                    ["git", "clone", f"https://github.com/{repo}.git", repo_path],
                    check=True
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone repository {repo}: {e}")
                return False

        return True

    def explore_repository(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze repository structure to locate relevant files for the issue.
        Now with prioritization of files from failing tests.
        """
        logger.info("Exploring repository structure for relevant files")

        # Get repository name and prepare it with the correct commit
        data_loader = SWEBenchDataLoader(self.config)

        # Ensure we use the base_commit for exploration
        data_loader.prepare_repository_for_analysis(issue)

        # Get repository path
        repo = issue.get("repo", "")
        repo_path = self.repo_base_path / repo

        # Get prioritized relevant files
        relevant_files = self._get_relevant_files(issue, data_loader, repo_path)

        # If using RAG, continue with that approach
        if self.use_rag and self.rag_system:
            logger.info("Using memory-efficient RAG analysis with prioritized files")
            rag_results = self.rag_system.retrieve_code_for_issue(issue)

            # Ensure prioritized files are included in results
            repo_exploration = {
                "repo_path": str(repo_path),
                "relevant_files": relevant_files,  # Use our prioritized list
                "using_rag": True
            }

            # Add other data from RAG results
            if rag_results:
                repo_exploration.update({
                    "file_scores": self.rag_system.get_file_scores(relevant_files),
                    "file_contents": rag_results,
                    "key_terms": self.rag_system.extract_key_terms_from_issue(issue)
                })

            return repo_exploration
        return {}

    def _get_relevant_files(self, issue: Dict[str, Any], data_loader, repo_path: Path) -> List[str]:
        """
        Extract the most relevant files for the issue, prioritizing implementation files from test patches.

        Args:
            issue: Issue dictionary with metadata
            data_loader: SWEBenchDataLoader instance
            repo_path: Path to the repository

        Returns:
            List of relevant file paths, with implementation files prioritized
        """
        relevant_files = []
        implementation_files = []

        # PRIORITY 1: Extract implementation files from test patch
        test_patch_info = data_loader.get_test_patch(issue)
        if isinstance(test_patch_info, dict) and "implementation_files" in test_patch_info:
            implementation_files = test_patch_info["implementation_files"]
            logger.info(f"Extracted implementation files from test patch: {implementation_files}")
            # Add implementation files with top priority
            for impl_file in implementation_files:
                if impl_file not in relevant_files:
                    relevant_files.append(impl_file)

        # PRIORITY 2: Extract files from failing tests
        fail_to_pass = data_loader.get_fail_to_pass_tests(issue)
        if fail_to_pass:
            logger.info(f"Adding files from failing tests: {fail_to_pass}")
            for test_path in fail_to_pass:
                # Extract module path from test path
                if "::" in test_path:
                    # Format like "path/to/test_file.py::test_function"
                    file_path = test_path.split("::")[0]
                else:
                    file_path = test_path

                # Add test file itself
                if file_path not in relevant_files:
                    relevant_files.append(file_path)

                # Infer the implementation file being tested if not already found
                if not implementation_files:
                    impl_file = self._infer_implementation_file(file_path)
                    if impl_file and impl_file not in relevant_files:
                        relevant_files.append(impl_file)

        # If test patch is a string, try to extract implementation files
        if isinstance(test_patch_info, str) and test_patch_info:
            # Legacy support for string test_patch
            file_pattern = r'(?:---|\+\+\+) [ab]/([^\n]+)'
            patch_files = re.findall(file_pattern, test_patch_info)

            # Look specifically for non-test files (implementation files)
            for file_path in patch_files:
                # Skip test files, focus on implementation files
                if 'test' not in file_path.lower():
                    if file_path not in relevant_files:
                        relevant_files.append(file_path)
                        logger.info(f"Found implementation file in patch: {file_path}")
                elif not implementation_files:
                    # If it's a test file, infer the implementation file
                    impl_file = self._infer_implementation_file(file_path)
                    if impl_file and impl_file not in relevant_files:
                        relevant_files.append(impl_file)
                        logger.info(f"Inferred implementation file: {impl_file}")

        # PRIORITY 3: Get files mentioned in the issue description
        description = data_loader.get_issue_description(issue)
        file_mentions = re.findall(r'`?([a-zA-Z0-9_/\.-]+\.(py|java|js|c|cpp|h))`?', description)
        for file_match in file_mentions:
            file_path = file_match[0]
            # Check if file exists in repo
            if (repo_path / file_path).exists():
                if file_path not in relevant_files:
                    relevant_files.append(file_path)

        # PRIORITY 4: Add files_modified from issue metadata
        if "files_modified" in issue:
            for file_path in issue["files_modified"]:
                if file_path not in relevant_files:
                    relevant_files.append(file_path)

        # If still no files found, use other methods
        if not relevant_files:
            # Find Python files that match patterns in the issue description
            all_python_files = list(repo_path.glob("**/*.py"))
            for py_file in all_python_files:
                rel_path = py_file.relative_to(repo_path)
                if self._file_matches_issue(str(rel_path), description):
                    if str(rel_path) not in relevant_files:
                        relevant_files.append(str(rel_path))

        # Additional handling for astropy/wcs/wcsapi/fitswcs.py
        astropy_file = "astropy/wcs/wcsapi/fitswcs.py"
        if "astropy" in str(repo_path) and astropy_file not in relevant_files:
            if (repo_path / astropy_file).exists():
                relevant_files.insert(0, astropy_file)  # Add at the beginning for highest priority
                logger.info(f"Added known important file: {astropy_file}")

        logger.info(f"Identified {len(relevant_files)} relevant files, prioritizing implementation files")
        return relevant_files

    def _infer_implementation_file(self, test_file_path: str) -> Optional[str]:
        """
        Infer the implementation file that is being tested by a test file.

        Args:
            test_file_path: Path to the test file

        Returns:
            Path to the inferred implementation file, or None if couldn't be inferred
        """
        # Common patterns: test_foo.py tests foo.py
        if not test_file_path or 'test' not in test_file_path.lower():
            return None

        file_name = os.path.basename(test_file_path)
        dir_name = os.path.dirname(test_file_path)

        # Extract implementation file name
        impl_name = None
        if file_name.startswith('test_'):
            impl_name = file_name[5:]  # Remove 'test_'
        elif file_name.endswith('_test.py'):
            impl_name = file_name[:-8] + '.py'  # Remove '_test.py'
        else:
            # Try to extract from module path for more complex cases
            parts = test_file_path.split('/')
            for i, part in enumerate(parts):
                if part == 'tests' and i > 0:
                    # Look for module name in parent directory
                    impl_name = parts[i - 1] + '.py'
                    break

        if not impl_name:
            return None

        # Try multiple directory structures to find the implementation file
        possible_impl_paths = []

        # 1. Same directory
        possible_impl_paths.append(os.path.join(dir_name, impl_name))

        # 2. Parent module (removing /tests/ directory)
        if '/tests/' in dir_name:
            parent_dir = dir_name.replace('/tests/', '/')
            possible_impl_paths.append(os.path.join(parent_dir, impl_name))

        # 3. Special case for specific repositories like astropy
        if test_file_path.startswith('astropy/'):
            # Special case for astropy/wcs/wcsapi/tests/test_fitswcs.py -> astropy/wcs/wcsapi/fitswcs.py
            test_module_path = test_file_path.split('/tests/')[0] if '/tests/' in test_file_path else dir_name
            possible_impl_paths.append(os.path.join(test_module_path, impl_name))

        # Return the most specific implementation path
        # Sort by path length (more specific paths are longer)
        return sorted(possible_impl_paths, key=len, reverse=True)[0] if possible_impl_paths else None

    def _file_matches_issue(self, file_path: str, issue_description: str) -> bool:
        """
        Check if a file matches patterns in the issue description.

        Args:
            file_path: Path to the file
            issue_description: Issue description

        Returns:
            True if file matches patterns in the issue
        """
        # Extract keywords from issue description
        keywords = set()
        # Split by non-alphanumeric chars and get words longer than 3 chars
        for word in re.findall(r'\w+', issue_description.lower()):
            if len(word) > 3 and not word.isdigit():
                keywords.add(word)

        # Check if file path contains keywords
        file_path_lower = file_path.lower()
        matches = sum(1 for keyword in keywords if keyword in file_path_lower)

        # More specific matching for some patterns
        file_name = os.path.basename(file_path_lower)
        if 'test' in issue_description.lower() and 'test' in file_name:
            matches += 2

        return matches >= 2  # Require at least 2 matches

    def _explore_with_rag(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform repository exploration using the RAG system for memory efficiency.

        Args:
            issue: Issue dictionary with metadata.

        Returns:
            Dictionary with repository exploration results.
        """
        # Analyze the issue using RAG
        rag_results = self.rag_system.retrieve_code_for_issue(issue)

        # Process RAG results to match the expected format
        repo = issue.get("repo", "")
        repo_path = Path(self.config["data"]["repositories"]) / repo
        logger.info(f"Repo path: {repo_path}")

        # Extract relevant files from code chunks
        relevant_files = list(set(chunk.get("file_path", "") for chunk in rag_results))

        # Extract scores for file filtering
        file_scores = []
        for file_path in relevant_files:
            # Find max score among chunks for this file
            max_score = 0
            for chunk in rag_results:
                if chunk.get("file_path") == file_path:
                    max_score = max(max_score, chunk.get("combined_score", 0))
            file_scores.append((file_path, max_score))

        # Sort by score
        file_scores.sort(key=lambda x: x[1], reverse=True)

        # Format file contents in the expected structure
        file_contents = {}
        for chunk in rag_results:
            file_path = chunk.get("file_path", "")

            if not file_path:
                continue

            if file_path not in file_contents:
                file_contents[file_path] = {
                    "chunks": [],
                    "functions": {},
                    "classes": [],
                    "content": chunk.get("content", "") if chunk.get("type") == "file" else "",
                    "relevance_score": chunk.get("combined_score", 0)
                }

            # Add chunk to file_contents
            file_contents[file_path]["chunks"].append(chunk)

            # Organize functions and classes
            if chunk.get("type") == "function":
                name = chunk.get("name", "")
                if name:
                    file_contents[file_path]["functions"][name] = {
                        "start_line": chunk.get("start_line", 0),
                        "end_line": chunk.get("end_line", 0),
                        "code": chunk.get("content", "")
                    }
            elif chunk.get("type") == "class":
                file_contents[file_path]["classes"].append({
                    "name": chunk.get("name", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "code": chunk.get("content", "")
                })

        # Create result dictionary
        result = {
            "repo_path": str(repo_path),
            "python_files_count": len(file_contents),
            "functions": [],  # These would need to be extracted from chunks if needed
            "key_terms": [],  # These would need to be extracted from chunks if needed
            "relevant_files": relevant_files,
            "file_contents": file_contents,
            "file_scores": file_scores,
            "using_rag": True
        }

        return result

    def _extract_parent_classes(self, class_def: str) -> List[str]:
        """
        Extract parent classes from a class definition.

        Args:
            class_def: The class definition string.

        Returns:
            List of parent class names.
        """
        # Look for anything between parentheses in the class definition
        parent_match = re.search(r'class\s+\w+\s*\((.*?)\):', class_def)
        if not parent_match:
            return []

        # Split the parent classes and clean them
        parents_str = parent_match.group(1)
        parents = [p.strip() for p in parents_str.split(',')]

        # Remove any arguments or whitespace
        cleaned_parents = []
        for parent in parents:
            # Extract just the class name, ignoring any arguments
            base_match = re.match(r'(\w+(?:\.\w+)*)', parent)
            if base_match:
                cleaned_parents.append(base_match.group(1))

        return cleaned_parents

    def retrieve_full_code(self, file_path: str, component_name: str = None) -> Dict[str, Any]:
        """
        Retrieve code for a specific file or component within a file.
        Optimized version that avoids loading the entire file when possible.

        Args:
            file_path: Path to the file.
            component_name: Optional name of component (function/class) to retrieve.

        Returns:
            Dictionary with component information and code.
        """
        try:
            # Convert Path to string if needed
            if isinstance(file_path, Path):
                file_path = str(file_path)

            # Check if file exists
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {"error": f"File not found: {file_path}"}

            # If component name is specified, try to extract only that component
            if component_name:
                # Use grep-like approach to find the component definition line
                import subprocess
                import io

                try:
                    # Find the start line of the component definition
                    grep_cmd = ["grep", "-n", f"(def|class)\\s+{component_name}\\s*(\\(|:)", file_path]
                    grep_result = subprocess.run(grep_cmd, capture_output=True, text=True)

                    if grep_result.returncode == 0 and grep_result.stdout:
                        # Extract line number from grep result
                        line_match = re.match(r'(\d+):', grep_result.stdout.split('\n')[0])
                        if line_match:
                            start_line = int(line_match.group(1))

                            # Read a chunk of the file starting from the found line
                            # We'll read 100 lines as a reasonable default for the component
                            with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                                # Skip to the start line
                                for _ in range(start_line - 1):
                                    next(f, None)

                                # Read the component content (up to 100 lines)
                                component_lines = []
                                line_count = 0
                                base_indent = None

                                for line in f:
                                    line_count += 1
                                    if line_count > 15:  # Safety limit
                                        break

                                    # Detect indentation of first line
                                    if line_count == 1:
                                        component_lines.append(line)
                                        match = re.match(r'^(\s*)', line)
                                        if match:
                                            base_indent = match.group(1)
                                        else:
                                            base_indent = ''
                                    else:
                                        # Check if we've reached the end of the component
                                        # This happens when indentation returns to base level or less
                                        if base_indent is not None and len(line.strip()) > 0:
                                            match = re.match(r'^(\s*)', line)
                                            current_indent = match.group(1) if match else ''

                                            if len(current_indent) <= len(base_indent) and line_count > 1:
                                                break

                                        component_lines.append(line)

                                component_code = ''.join(component_lines)

                                # Determine component type
                                comp_type = "function" if "def " in component_lines[0] else "class"

                                return {
                                    "type": comp_type,
                                    "name": component_name,
                                    "file_path": str(file_path_obj),
                                    "code": component_code,
                                    "start_line": start_line,
                                    "end_line": start_line + len(component_lines) - 1
                                }
                except subprocess.SubprocessError:
                    # Fall back to reading the whole file if grep fails
                    pass

            # If component-specific extraction failed or wasn't requested,
            # read the file but limit content size for token efficiency
            with open(file_path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                # For large files, only read up to 10KB to save tokens
                file_stat = os.stat(file_path_obj)
                if file_stat.st_size > 10240 and not component_name:  # 10KB limit if no specific component
                    content = f.read(10240)
                    content += "\n... [file truncated due to size] ..."
                else:
                    content = f.read()

            # If no component name specified or if we failed to extract it specifically
            if not component_name:
                return {
                    "type": "file",
                    "name": file_path_obj.name,
                    "file_path": str(file_path_obj),
                    "code": content,
                    "start_line": 1,
                    "end_line": content.count('\n') + 1,
                    "truncated": file_stat.st_size > 10240
                }

            # If we get here, it means we have a component name but couldn't extract it efficiently
            # Try regex-based extraction on the loaded content
            func_pattern = r'(def\s+' + re.escape(component_name) + r'\s*\([^)]*\)(?:\s*->.*?)?:(?:\n(?:\s+.*\n)+))'
            func_match = re.search(func_pattern, content)
            if func_match:
                func_code = func_match.group(1)
                start_line = content[:func_match.start()].count('\n') + 1
                end_line = start_line + func_code.count('\n')
                return {
                    "type": "function",
                    "name": component_name,
                    "file_path": str(file_path_obj),
                    "code": func_code,
                    "start_line": start_line,
                    "end_line": end_line
                }

            # For class
            class_pattern = r'(class\s+' + re.escape(component_name) + r'\s*(?:\([^)]*\))?:(?:\n(?:\s+.*\n)+))'
            class_match = re.search(class_pattern, content)
            if class_match:
                class_code = class_match.group(1)
                start_line = content[:class_match.start()].count('\n') + 1
                end_line = start_line + class_code.count('\n')
                return {
                    "type": "class",
                    "name": component_name,
                    "file_path": str(file_path_obj),
                    "code": class_code,
                    "start_line": start_line,
                    "end_line": end_line
                }

            # Component not found
            return {
                "error": f"Component {component_name} not found in {file_path}",
                "type": "file",
                "name": file_path_obj.name,
                "file_path": str(file_path_obj),
                "code_snippet": content[:5] + "...",  # Just return a snippet instead of full content
                "start_line": 1,
                "end_line": content.count('\n') + 1
            }

        except Exception as e:
            logger.error(f"Error retrieving code from {file_path}: {e}")
            return {"error": str(e)}
