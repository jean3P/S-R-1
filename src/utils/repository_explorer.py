# src/utils/repository_explorer.py

import logging
import re
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add the import for the new RAG system
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
        Now with option to use RAG for memory efficiency.
        """
        logger.info("Exploring repository structure for relevant files")

        # If RAG system is available and enabled, use it for memory-efficient analysis
        if self.use_rag and self.rag_system:
            logger.info("Using memory-efficient RAG analysis")
            return self._explore_with_rag(issue)
        else:
            # Fall back to the original implementation
            logger.info("Using standard repository exploration")
            return self._explore_standard(issue)

    def _explore_with_rag(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform repository exploration using the RAG system for memory efficiency.

        Args:
            issue: Issue dictionary with metadata.

        Returns:
            Dictionary with repository exploration results.
        """
        # Analyze the issue using RAG
        rag_results = self.rag_system.analyze_issue(issue, top_k=10)

        # If RAG analysis failed, fall back to standard exploration
        if "error" in rag_results:
            logger.warning(f"RAG analysis failed: {rag_results['error']}. Falling back to standard exploration.")
            return self._explore_standard(issue)

        # Process RAG results to match the expected format
        repo = issue.get("repo", "")
        repo_path = Path(self.config["data"]["repositories"]) / repo

        # Extract scores from RAG results for file filtering
        file_scores = []
        for file_path in rag_results.get("relevant_files", []):
            # Find max score among chunks for this file
            max_score = 0
            for chunk in rag_results.get("relevant_chunks", []):
                if chunk.get("file_path") == file_path:
                    max_score = max(max_score, chunk.get("score", 0))
            file_scores.append((file_path, max_score))

        # Sort by score
        file_scores.sort(key=lambda x: x[1], reverse=True)

        # Format file contents in the expected structure
        file_contents = {}
        for file_path, file_info in rag_results.get("file_contents", {}).items():
            chunks = file_info.get("chunks", [])
            functions = file_info.get("functions", {})
            classes = file_info.get("classes", [])

            # Determine if we have a full file or just chunks
            has_full_content = False
            full_content = ""

            # Try to find a full file chunk
            for chunk in chunks:
                if chunk.get("type") == "file":
                    has_full_content = True
                    full_content = chunk.get("content", "")
                    break

            # If no full content, extract content from chunks if needed
            if not has_full_content:
                # For now, just use the function/class code
                combined_chunks = ""
                for chunk in chunks:
                    if "content" in chunk:
                        combined_chunks += chunk["content"] + "\n\n"
                full_content = combined_chunks

            # Create file info structure
            file_contents[file_path] = {
                "content": full_content,
                "lines_count": full_content.count('\n') + 1,
                "functions": functions,
                "classes": classes,
                "relevance_score": next((score for path, score in file_scores if path == file_path), 0)
            }

        # Create result dictionary
        result = {
            "repo_path": str(repo_path),
            "python_files_count": len(rag_results.get("relevant_files", [])),
            "imports": [],  # Not available directly from RAG
            "functions": rag_results.get("functions", []),
            "key_terms": rag_results.get("key_terms", []),
            "mentioned_files": [],  # Not available directly from RAG
            "relevant_files": rag_results.get("relevant_files", []),
            "file_contents": file_contents,
            "file_scores": file_scores,
            "using_rag": True
        }

        return result

    def _explore_standard(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        The original repository exploration implementation.
        Used as fallback when RAG is not available.
        """
        # Get repository path
        repo = issue.get("repo", "")
        repo_base_path = Path(self.config["data"]["repositories"]) / repo

        logger.info(f"Base repository path: {repo_base_path}")

        if not repo_base_path.exists():
            logger.warning(f"Repository path does not exist: {repo_base_path}")
            return {
                "error": f"Repository not found: {repo}",
                "repo_path": str(repo_base_path),
                "imports": [],
                "functions": [],
                "file_paths": [],
                "relevant_files": [],
                "file_contents": {}
            }

        # Clear CUDA cache before processing files
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Find all Python files in the repository
        all_python_files = list(repo_base_path.glob("**/*.py"))
        logger.info(f"Found {len(all_python_files)} Python files in the repository")

        # Extract key information from issue description
        from ..data.data_loader import SWEBenchDataLoader
        data_loader = SWEBenchDataLoader(self.config)
        issue_description = data_loader.get_issue_description(issue)
        logger.info(
            f"Issue Description: {issue_description[:100]}..." if len(issue_description) > 100 else issue_description)

        # Extract key terms for similarity matching
        key_terms = self._extract_key_terms(issue_description)
        logger.info(f"Extracted key terms: {key_terms}")

        # Extract file paths mentioned in the issue
        mentioned_files = self._extract_file_paths(issue_description)
        logger.info(f"Files mentioned in issue: {mentioned_files}")

        # First, try to find exact matches for mentioned files
        exact_matches = []
        for mentioned_file in mentioned_files:
            # Normalize path
            mentioned_path = Path(mentioned_file)
            # Try various path combinations
            for py_file in all_python_files:
                if py_file.name == mentioned_path.name:
                    # If the name matches, check if partial paths match
                    if str(mentioned_path) in str(py_file):
                        exact_matches.append(py_file)
                        logger.info(f"Found exact match for {mentioned_file}: {py_file}")

        # Calculate similarity scores for all Python files
        relevant_files = self._score_files_by_relevance(
            all_python_files,
            key_terms,
            issue_description,
            exact_matches=exact_matches
        )

        logger.info(f"Identified {len(relevant_files)} relevant files")

        # Extract functions mentioned in the issue description
        functions = self._extract_functions(issue_description)

        # Examine content for most relevant files
        file_contents = {}
        max_files_to_analyze = min(len(relevant_files), 10)  # Limit to top 10 files

        for i, (file_path, score) in enumerate(relevant_files[:max_files_to_analyze]):
            try:
                # Get the relative path from repo path
                rel_path = file_path.relative_to(repo_base_path)
                rel_path_str = str(rel_path)

                logger.info(f"Analyzing file {i + 1}/{max_files_to_analyze}: {rel_path_str} (score: {score:.2f})")

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Analyze the file content
                file_contents[rel_path_str] = {
                    "content": content,
                    "lines_count": len(content.split('\n')),
                    "functions": self._extract_function_definitions(content, functions),
                    "classes": self._extract_class_definitions(content),
                    "relevance_score": score
                }

                # Clear CUDA cache after processing each file
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                file_contents[str(rel_path)] = {"error": str(e)}

        # Create result dictionary
        result = {
            "repo_path": str(repo_base_path),
            "python_files_count": len(all_python_files),
            "imports": self._extract_imports(issue_description),
            "functions": functions,
            "key_terms": key_terms,
            "mentioned_files": mentioned_files,
            "relevant_files": [str(f.relative_to(repo_base_path)) for f, _ in relevant_files[:max_files_to_analyze]],
            "file_contents": file_contents,
            "using_rag": False
        }

        # Store file scores in the result dictionary for later filtering
        result["file_scores"] = [(str(f.relative_to(repo_base_path)), score) for f, score in
                                 relevant_files[:max_files_to_analyze]]

        return result

    # Keep all your existing helper methods from here down
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key technical terms from text for similarity matching."""
        # Remove code blocks as they might contain unrelated terms
        text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

        # Extract technical terms using regex
        # Look for terms that might be file names, class names, function names, etc.
        term_patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
            r'\b[a-z]+_[a-z_]+\b',  # snake_case
            r'\b[a-z]+\.[a-z]+\b',  # dot.notation
            r'[\w/\.-]+\.py\b',  # Python files
            r'import\s+([a-zA-Z0-9_.]+)',  # import statements
            r'from\s+([a-zA-Z0-9_.]+)',  # from statements
        ]

        terms = []
        for pattern in term_patterns:
            terms.extend(re.findall(pattern, text_without_code))

        # Add explicit terms from the text that might be relevant
        explicit_terms = re.findall(r'"([^"]+)"', text_without_code)
        explicit_terms.extend(re.findall(r"'([^']+)'", text_without_code))
        terms.extend(explicit_terms)

        # Look specifically for module paths
        module_paths = re.findall(r'(?:from|import)\s+([\w.]+(?:\.[\w.]+)*)', text_without_code)
        for path in module_paths:
            terms.extend(path.split('.'))

        # Remove duplicates, filter short terms, and normalize
        normalized_terms = []
        for term in set(terms):
            term = term.strip()
            if len(term) > 2 and term not in normalized_terms:
                normalized_terms.append(term)

                # Also add parts of compound terms
                if '.' in term:
                    parts = term.split('.')
                    for part in parts:
                        if len(part) > 2 and part not in normalized_terms:
                            normalized_terms.append(part)

        return normalized_terms

    def _extract_imports(self, text: str) -> List[Dict[str, str]]:
        """Extract imports from text."""
        imports = []
        # Match patterns like: from astropy.modeling import models as m
        import_pattern = r'from\s+([\w\.]+)\s+import\s+([\w\.,\s]+)(?:\s+as\s+(\w+))?'
        for match in re.finditer(import_pattern, text):
            module, objects, alias = match.groups()
            imports.append({
                "module": module,
                "objects": [obj.strip() for obj in objects.split(',')],
                "alias": alias
            })
        return imports

    def _extract_functions(self, text: str) -> List[str]:
        """Extract function references from text."""
        functions = []
        # Match patterns like: separability_matrix(model)
        func_pattern = r'(\w+)\s*\('
        for match in re.finditer(func_pattern, text):
            functions.append(match.group(1))
        return list(set(functions))  # Remove duplicates

    def _extract_file_paths(self, text: str) -> List[str]:
        """Extract file path references from text."""
        file_paths = []
        # Match patterns like: astropy/modeling/separable.py
        path_pattern = r'([\w\/\.-]+\.\w+)'
        for match in re.finditer(path_pattern, text):
            path = match.group(1)
            if '.' in path and not path.startswith(('http://', 'https://')):
                file_paths.append(path)
        return list(set(file_paths))  # Remove duplicates

    def _score_files_by_relevance(
            self,
            python_files: List[Path],
            key_terms: List[str],
            issue_description: str,
            exact_matches: List[Path] = None
    ) -> List[Tuple[Path, float]]:
        """
        Score files based on their relevance to the issue.

        Args:
            python_files: List of paths to Python files
            key_terms: List of key terms extracted from the issue
            issue_description: Full issue description
            exact_matches: List of files that exactly match mentioned paths

        Returns:
            List of (file_path, score) tuples, sorted by relevance
        """
        file_scores = {}

        # Ensure exact matches get top priority
        if exact_matches:
            for file_path in exact_matches:
                file_scores[file_path] = 1000  # Very high score to ensure they come first

        # Process each Python file
        for file_path in python_files:
            # Skip if already scored as exact match
            if file_path in file_scores:
                continue

            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Base score starts at 0
                score = 0

                # 1. Check filename relevance
                filename = file_path.name.lower()
                for term in key_terms:
                    term_lower = term.lower()
                    # Term in filename is very relevant
                    if term_lower in filename:
                        score += 10
                        # Even more relevant if it's the main part of the filename
                        if filename.startswith(term_lower) or filename.endswith(term_lower + '.py'):
                            score += 10

                # 2. Check directory name relevance
                dir_name = file_path.parent.name.lower()
                for term in key_terms:
                    if term.lower() in dir_name:
                        score += 5

                # 3. Check content relevance
                content_lower = content.lower()
                for term in key_terms:
                    term_lower = term.lower()
                    # Count occurrences in content (more occurrences = more relevant)
                    occurrences = content_lower.count(term_lower)
                    if occurrences > 0:
                        score += min(occurrences, 10)  # Cap to avoid extreme scores

                # 4. Check for function names mentioned in the issue
                functions = self._extract_functions(issue_description)
                for func in functions:
                    # Function definition is highly relevant
                    if re.search(r'def\s+' + re.escape(func) + r'\s*\(', content):
                        score += 20
                    # Function used in code is also relevant
                    elif re.search(r'\b' + re.escape(func) + r'\s*\(', content):
                        score += 10

                # 5. Boost score for non-test files
                if "test" not in filename and "test" not in str(file_path):
                    score *= 1.5

                # 6. Penalize very large files (less likely to be directly relevant)
                if len(content) > 100000:  # 100KB
                    score *= 0.7

                # 7. Penalize files with very generic names
                generic_names = ["__init__", "utils", "helpers", "common", "base"]
                if any(generic in filename for generic in generic_names):
                    score *= 0.8

                # Store the score if positive
                if score > 0:
                    file_scores[file_path] = score

            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")

        # Sort files by score and return
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        # Store the file scores for later filtering
        self.file_scores = sorted_files
        return sorted_files

    def _extract_function_definitions(self, content: str, function_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract function definitions and their locations in the file content."""
        function_defs = {}

        for func_name in function_names:
            # Look for function definition
            pattern = re.compile(
                r'def\s+' + re.escape(func_name) + r'\s*\([^)]*\)(?:\s*->.*?)?\s*:(?:\s*#.*?)?((?:\n\s+.*)+)',
                re.DOTALL)
            matches = pattern.finditer(content)

            for match in matches:
                start_pos = match.start()
                end_pos = match.end()

                # Get line numbers
                start_line = content[:start_pos].count('\n') + 1
                end_line = content[:end_pos].count('\n') + 1

                function_defs[func_name] = {
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": match.group(0)
                }

        return function_defs

    def _extract_class_definitions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract class definitions from file content.

        Args:
            content: The source code content of the file.

        Returns:
            List of dictionaries containing class information.
        """
        classes = []

        # Match class definitions
        class_pattern = r'class\s+(\w+)(?:\(.*?\))?:'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            start_pos = match.start()

            # Get line number
            line_num = content[:start_pos].count('\n') + 1

            # Try to find the end of the class
            class_body_start = content.find(':', match.end()) + 1

            # Look for the next class definition or end of file
            next_class_pos = len(content)
            next_class_match = re.search(r'\nclass\s+\w+', content[class_body_start:])
            if next_class_match:
                next_class_pos = class_body_start + next_class_match.start()

            # Estimate class end by looking for non-indented lines after the class
            class_content = content[class_body_start:next_class_pos]
            lines = class_content.split('\n')

            # Start with the assumption that the class extends to the next class or EOF
            end_line = line_num + len(lines)

            # Find where indentation ends for this class
            in_class = False
            class_indent = None

            for i, line in enumerate(lines):
                stripped = line.lstrip()

                # Skip empty lines
                if not stripped:
                    continue

                # Calculate indentation level
                indent_level = len(line) - len(stripped)

                # First non-empty line establishes the class indentation level
                if not in_class:
                    class_indent = indent_level
                    in_class = True
                    continue

                # If we find a line with less or equal indentation than the class definition,
                # and it's not a comment, decorator, or string continuation,
                # it's likely the end of the class
                if in_class and indent_level <= class_indent and not stripped.startswith(
                        '#') and not stripped.startswith(
                    '@'):
                    end_line = line_num + i
                    break

            # Extract methods within the class
            methods = []
            method_pattern = r'^\s+def\s+(\w+)\s*\('
            class_text = content[
                         start_pos:min(next_class_pos, content.find('\n', start_pos) + (end_line - line_num) * 20)]

            for method_match in re.finditer(method_pattern, class_text, re.MULTILINE):
                method_name = method_match.group(1)
                method_start_pos = method_match.start() + start_pos
                method_line = content[:method_start_pos].count('\n') + 1
                methods.append({
                    "name": method_name,
                    "line": method_line
                })

            classes.append({
                "name": class_name,
                "start_line": line_num,
                "end_line": end_line,
                "methods": methods,
                "parent_classes": self._extract_parent_classes(match.group(0))
            })

        return classes

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
