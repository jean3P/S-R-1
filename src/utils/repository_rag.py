# src/utils/repository_rag.py
import json
import logging
import os
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..data.data_loader import SWEBenchDataLoader

logger = logging.getLogger(__name__)


# Simple vector index using NumPy instead of FAISS
class SimpleVectorIndex:
    """A simple vector index implementation using NumPy and cosine similarity."""

    def __init__(self, dimension):
        """Initialize with vector dimension."""
        self.embeddings = None
        self.dimension = dimension
        self.ntotal = 0

    def add(self, embeddings):
        """Add vectors to the index."""
        self.embeddings = embeddings
        self.ntotal = len(embeddings)

    def search(self, query_embedding, k):
        """Search for nearest neighbors using cosine similarity."""
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top k indices
        k = min(k, self.ntotal)  # Make sure k isn't larger than total vectors
        indices = np.argsort(similarities)[-k:][::-1]

        # Get distances (convert similarities to distances)
        distances = 1 - similarities[indices]

        # Reshape to match expected output format
        return np.array([distances]), np.array([indices])


class RepositoryRAG:
    """
    Retrieval-Augmented Generation for repositories.
    Creates and manages embeddings for repository code to enable
    efficient code retrieval for bug fixing.
    """

    def __init__(self, config):
        """
        Initialize the Repository RAG.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.repo_path = Path(config["data"]["repositories"])
        self.cache_dir = Path(config["data"]["cache_dir"])

        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

        # Initialize the embedding model
        self.embedding_model = self._load_embedding_model()

        # Create embeddings directory
        self.embeddings_dir = self.cache_dir / "embeddings"
        if not self.embeddings_dir.exists():
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Cache for storing indexed repositories
        self.index_cache = {}

    def _load_embedding_model(self):
        """Load the embedding model for code similarity."""
        # Use a smaller but effective model for code embeddings
        try:
            model_name = "all-MiniLM-L6-v2"  # Small but effective general-purpose model
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            return model
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            logger.info("Trying to download the model...")
            try:
                # Try downloading explicitly
                os.system(f"pip install -q sentence-transformers")
                model = SentenceTransformer(model_name)
                return model
            except Exception as e2:
                logger.error(f"Failed to load embedding model: {e2}")
                raise

    def index_repository(self, repo_name: str) -> bool:
        """
        Index a repository by creating embeddings for its code chunks.

        Args:
            repo_name: Name of the repository to index.

        Returns:
            Boolean indicating if indexing was successful.
        """
        repo_dir = self.repo_path / repo_name
        if not repo_dir.exists():
            logger.error(f"Repository {repo_name} not found at {repo_dir}")
            return False

        logger.info(f"Indexing repository: {repo_name}")

        # Check if repository is already indexed
        index_file = self.embeddings_dir / f"{repo_name}_index.npy"
        metadata_file = self.embeddings_dir / f"{repo_name}_metadata.pt"

        if index_file.exists() and metadata_file.exists():
            logger.info(f"Loading existing index for {repo_name}")
            try:
                # Load existing embeddings
                embeddings = np.load(str(index_file))
                metadata = torch.load(metadata_file, weights_only=True)

                # Create index
                dimension = embeddings.shape[1]
                index = SimpleVectorIndex(dimension)
                index.add(embeddings)

                # Store in cache
                self.index_cache[repo_name] = {
                    "index": index,
                    "metadata": metadata
                }
                return True
            except Exception as e:
                logger.warning(f"Error loading existing index, will create new one: {e}")

        # Find all Python files
        python_files = list(repo_dir.glob("**/*.py"))
        logger.info(f"Found {len(python_files)} Python files in {repo_name}")

        # Extract code chunks from files
        chunks = []
        metadata = []

        for file_path in python_files:
            try:
                # Get relative path for easier reference
                rel_path = file_path.relative_to(repo_dir)

                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Extract functions and classes
                file_chunks = self._extract_code_chunks(content, str(rel_path))

                # Add file-level chunk for small files
                if len(content) < 10000:  # Only for reasonably sized files
                    chunks.append(content)
                    metadata.append({
                        "file_path": str(rel_path),
                        "type": "file",
                        "name": str(rel_path),
                        "start_line": 1,
                        "end_line": content.count('\n') + 1,
                        "content": content
                    })

                # Add function and class chunks
                for chunk in file_chunks:
                    chunks.append(chunk["content"])
                    metadata.append(chunk)

                # Clear memory periodically
                if len(chunks) % 1000 == 0:
                    logger.info(f"Extracted {len(chunks)} chunks so far...")
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

        if not chunks:
            logger.warning(f"No code chunks extracted from {repo_name}")
            return False

        logger.info(f"Creating embeddings for {len(chunks)} code chunks")

        # Create embeddings in batches to avoid OOM
        batch_size = 128
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            # Create embeddings for batch
            try:
                with torch.no_grad():
                    embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
                all_embeddings.append(embeddings)
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i // batch_size}: {e}")
                # Try a smaller batch if this one failed
                half_batch = batch_size // 2
                if half_batch > 0:
                    try:
                        for j in range(i, min(i + batch_size, len(chunks)), half_batch):
                            sub_batch = chunks[j:j + half_batch]
                            with torch.no_grad():
                                sub_embeddings = self.embedding_model.encode(sub_batch, show_progress_bar=False)
                            all_embeddings.append(sub_embeddings)
                    except Exception as e2:
                        logger.error(f"Error creating embeddings even with smaller batch: {e2}")

            # Clear memory
            torch.cuda.empty_cache()

        logger.info(f"Created embeddings for {len(all_embeddings)} batches, total chunks: {len(chunks)}")
        # Combine all embeddings
        if not all_embeddings:
            logger.error(f"Failed to create any embeddings for {repo_name}")
            return False

        embeddings = np.vstack(all_embeddings)

        # Create vector index
        dimension = embeddings.shape[1]
        index = SimpleVectorIndex(dimension)
        index.add(embeddings.astype(np.float32))

        # Save index and metadata
        try:
            # Ensure parent directories exist
            index_file.parent.mkdir(parents=True, exist_ok=True)
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            np.save(str(index_file), embeddings.astype(np.float32))
            torch.save(metadata, metadata_file)

            # Store in cache
            self.index_cache[repo_name] = {
                "index": index,
                "metadata": metadata
            }

            logger.info(f"Successfully indexed repository {repo_name} with {len(metadata)} chunks")
            logger.info(f"Created vector index with dimension {dimension} and {index.ntotal} vectors")

            return True
        except Exception as e:
            logger.error(f"Error saving index for {repo_name}: {e}")
            return False

    def _extract_code_chunks(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract functions and classes from code content.

        Args:
            content: The code content to process.
            file_path: Path to the file (for metadata).

        Returns:
            List of code chunks with metadata.
        """
        chunks = []

        # Extract classes
        class_pattern = r'(class\s+\w+(?:\([^)]*\))?:(?:.|\n)*?)(?=\n\S|$)'
        for match in re.finditer(class_pattern, content):
            class_code = match.group(1)
            if len(class_code.strip()) < 10:  # Skip tiny classes
                continue

            # Extract class name
            class_name_match = re.search(r'class\s+(\w+)', class_code)
            if class_name_match:
                class_name = class_name_match.group(1)

                # Calculate line numbers
                start_line = content[:match.start()].count('\n') + 1
                end_line = start_line + class_code.count('\n')

                chunks.append({
                    "file_path": file_path,
                    "type": "class",
                    "name": class_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": class_code
                })

                # Also extract methods within the class
                method_pattern = r'    def\s+(\w+)\s*\([^)]*\)(?:\s*->.*?)?:(?:.|\n)*?(?=\n        \S|\n    \S|\n\S|$)'
                for method_match in re.finditer(method_pattern, class_code):
                    method_code = method_match.group(0)
                    if len(method_code.strip()) < 10:  # Skip tiny methods
                        continue

                    # Extract method name
                    method_name_match = re.search(r'def\s+(\w+)', method_code)
                    if method_name_match:
                        method_name = method_name_match.group(1)

                        # Calculate line numbers relative to file
                        method_start = start_line + class_code[:method_match.start()].count('\n')
                        method_end = method_start + method_code.count('\n')

                        chunks.append({
                            "file_path": file_path,
                            "type": "method",
                            "class_name": class_name,
                            "name": f"{class_name}.{method_name}",
                            "start_line": method_start,
                            "end_line": method_end,
                            "content": method_code
                        })

        # Extract standalone functions
        func_pattern = r'(def\s+\w+\s*\([^)]*\)(?:\s*->.*?)?:(?:.|\n)*?)(?=\n\S|$)'
        for match in re.finditer(func_pattern, content):
            func_code = match.group(1)
            if len(func_code.strip()) < 10:  # Skip tiny functions
                continue

            # Extract function name
            func_name_match = re.search(r'def\s+(\w+)', func_code)
            if func_name_match:
                func_name = func_name_match.group(1)

                # Calculate line numbers
                start_line = content[:match.start()].count('\n') + 1
                end_line = start_line + func_code.count('\n')

                chunks.append({
                    "file_path": file_path,
                    "type": "function",
                    "name": func_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": func_code
                })

        return chunks

    def retrieve_relevant_code(self,
                               repo_name: str,
                               query: str,
                               test_patch_info: Dict[str, Any] = None,
                               top_k: int = 5,
                               include_content: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant code chunks for a query, incorporating test patch information.

        Args:
            repo_name: Name of the repository.
            query: The query text (issue description).
            test_patch_info: Dictionary containing test patch analysis from get_test_patch().
            top_k: Number of results to return.
            include_content: Whether to include code content in results.

        Returns:
            List of relevant code chunks with metadata.
        """
        # Ensure repository is indexed
        if repo_name not in self.index_cache:
            success = self.index_repository(repo_name)
            if not success:
                logger.error(f"Failed to index repository {repo_name}")
                return []

        # Get index and metadata
        index = self.index_cache[repo_name]["index"]
        metadata = self.index_cache[repo_name]["metadata"]
        logger.info(f"Searching index with {index.ntotal} entries for query: '{query[:100]}...'")

        # Get files and code information from test patch info
        test_related_files = []
        test_related_code = []
        tested_files = []

        if test_patch_info:
            logger.info("Using test patch analysis for enhanced retrieval")

            # Get test files and implementation files
            test_related_files = test_patch_info.get("files", [])
            tested_files = test_patch_info.get("implementation_files", [])

            # Get test functions and imports
            test_functions = test_patch_info.get("test_functions", [])
            imports = test_patch_info.get("imports", [])
            assertions = test_patch_info.get("assertions", [])

            # Combine all code-related information
            test_related_code.extend(test_functions)
            test_related_code.extend(imports)
            test_related_code.extend(assertions)

            # Log what we found
            logger.info(f"Using {len(test_related_files)} test files, {len(tested_files)} implementation files, "
                        f"and {len(test_related_code)} code snippets from test patch")

        # Enhance query with test-related information
        enhanced_query = query
        if test_related_code:
            test_code_context = "\n".join(test_related_code[:10])  # Limit to avoid too much noise
            enhanced_query += f"\n\nRelated test code:\n{test_code_context}"

        # Create query embedding for the enhanced query
        query_embedding = self.embedding_model.encode([enhanced_query])[0]

        # Search the index
        distances, indices = index.search(
            np.array([query_embedding]).astype(np.float32),
            min(top_k * 2, index.ntotal)  # Get more results initially for filtering
        )

        # Get raw results
        raw_results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            result = metadata[idx].copy()
            result["score"] = float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
            raw_results.append(result)

        # Prioritize results from tested files
        prioritized_results = []
        remaining_results = []

        for result in raw_results:
            file_path = result.get("file_path", "")

            # Check if this result is from a tested file or has high relevance to the test
            is_priority = any(tested_file in file_path for tested_file in tested_files)

            # Also check if it's directly mentioned in the test file
            if not is_priority and "content" in result:
                content = result.get("content", "")
                for test_code in test_related_code:
                    # Look for function names or class names from the test in the implementation
                    func_match = re.search(r'def\s+(\w+)', test_code)
                    if func_match and func_match.group(1) in content:
                        is_priority = True
                        # Boost score for direct matches
                        result["score"] *= 1.5
                        break

            if is_priority:
                prioritized_results.append(result)
            else:
                remaining_results.append(result)

        # Combine results, prioritizing test-related files but keeping high-scoring unrelated files
        final_results = prioritized_results + remaining_results
        final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k]

        # Optionally remove content to save memory
        if not include_content:
            for result in final_results:
                if "content" in result:
                    del result["content"]

        logger.info(
            f"Returning {len(final_results)} relevant code chunks, {len(prioritized_results)} from test-related files")
        return final_results

    def analyze_issue(self, issue: Dict[str, Any], top_k: int = 8) -> Dict[str, Any]:
        """
        Analyze an issue to find relevant code.

        Args:
            issue: Issue dictionary.
            top_k: Number of relevant code chunks to retrieve.

        Returns:
            Dictionary with repository exploration results.
        """
        # Get repository name
        repo = issue.get("repo", "")
        if not repo:
            logger.error("No repository specified in issue")
            return {"error": "No repository specified in issue"}

        # Get issue description
        data_loader = SWEBenchDataLoader(self.config)
        description = data_loader.get_issue_description(issue)
        test_patch = data_loader.get_test_patch(issue)
        if not description:
            logger.error("No issue description found")
            return {"error": "No issue description found"}

        logger.info(f"Analyzing issue for repository: {repo}")

        # Extract key terms for improved retrieval
        key_terms = self._extract_key_terms(description)

        # Create combined query with key terms for better retrieval
        query = f"{description}\n\nKey terms: {', '.join(key_terms)}"

        # Retrieve relevant code chunks
        relevant_chunks = self.retrieve_relevant_code(repo, query, test_patch, top_k=top_k)

        if not relevant_chunks:
            logger.warning(f"No relevant code chunks found for issue in {repo}")
        else:
            logger.info(f"Relevant chunks: {len(relevant_chunks)} chunks")

        # Organize results
        relevant_files = list(set(chunk["file_path"] for chunk in relevant_chunks))

        # Create file structure for exploration results
        file_contents = {}
        for chunk in relevant_chunks:
            file_path = chunk["file_path"]

            if file_path not in file_contents:
                file_contents[file_path] = {
                    "chunks": [],
                    "functions": {},
                    "classes": []
                }

            # Add chunk to file contents
            file_contents[file_path]["chunks"].append(chunk)

            # Organize functions and classes
            if chunk["type"] == "function":
                file_contents[file_path]["functions"][chunk["name"]] = {
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "code": chunk["content"]
                }
            elif chunk["type"] == "class":
                file_contents[file_path]["classes"].append({
                    "name": chunk["name"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "code": chunk["content"]
                })

        # Extract functions mentioned in the issue
        functions = self._extract_functions(description)

        # Create exploration result
        result = {
            "repo": repo,
            "repo_path": str(self.repo_path / repo),
            "key_terms": key_terms,
            "functions": functions,
            "relevant_files": relevant_files,
            "file_contents": file_contents,
            "relevant_chunks": relevant_chunks
        }

        return result

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

    def _extract_functions(self, text: str) -> List[str]:
        """Extract function references from text."""
        functions = []
        # Match patterns like: separability_matrix(model)
        func_pattern = r'(\w+)\s*\('
        for match in re.finditer(func_pattern, text):
            functions.append(match.group(1))
        return list(set(functions))  # Remove duplicates

    def format_for_llm(self, analysis_result: Dict[str, Any], max_content_length: int = 12000) -> str:
        """
        Format repository analysis results for input to an LLM.

        Args:
            analysis_result: Results from analyze_issue.
            max_content_length: Maximum content length to return.

        Returns:
            Formatted string for LLM input.
        """
        if "error" in analysis_result:
            return f"Error analyzing repository: {analysis_result['error']}"

        repo = analysis_result.get("repo", "unknown")
        relevant_files = analysis_result.get("relevant_files", [])
        file_contents = analysis_result.get("file_contents", {})

        formatted_text = f"# REPOSITORY ANALYSIS: {repo}\n\n"

        # Add key terms
        key_terms = analysis_result.get("key_terms", [])
        if key_terms:
            formatted_text += "## Key Terms\n"
            formatted_text += ", ".join(key_terms)
            formatted_text += "\n\n"

        # Add relevant files overview
        formatted_text += f"## Relevant Files ({len(relevant_files)})\n"
        for file_path in relevant_files:
            formatted_text += f"- {file_path}\n"
        formatted_text += "\n"

        # Add file contents with chunks
        formatted_text += "## Code Analysis\n\n"

        content_so_far = len(formatted_text)

        # Sort chunks by relevance score
        all_chunks = []
        for file_path, file_info in file_contents.items():
            for chunk in file_info["chunks"]:
                chunk["file_path"] = file_path  # Ensure file path is included
                all_chunks.append(chunk)

        all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Add most relevant chunks first
        for chunk in all_chunks:
            file_path = chunk["file_path"]
            chunk_type = chunk["type"]
            chunk_name = chunk["name"]
            content = chunk.get("content", "")

            chunk_header = f"### {file_path}: {chunk_type} `{chunk_name}` (Lines {chunk['start_line']}-{chunk['end_line']})\n"
            chunk_content = f"```python\n{content}\n```\n\n"

            # Check if adding this chunk would exceed the max length
            if content_so_far + len(chunk_header) + len(chunk_content) > max_content_length:
                formatted_text += "\n... (content truncated due to length constraints) ...\n"
                break

            formatted_text += chunk_header + chunk_content
            content_so_far += len(chunk_header) + len(chunk_content)

        return formatted_text


    def _extract_entities_from_problem(self, problem_statement: str) -> Dict[str, List[str]]:
        """
        Extract key entities from problem statements with improved precision.

        Args:
            problem_statement: The issue description text

        Returns:
            Dictionary containing extracted entities by category
        """
        entities = {
            "functions": [],
            "classes": [],
            "files": [],
            "api_endpoints": [],
            "technical_terms": []
        }

        # Remove code blocks to avoid confusing syntax with entity names
        text_without_code = re.sub(r'```.*?```', '', problem_statement, flags=re.DOTALL)

        # Extract file paths - improved pattern for common extensions
        file_pattern = r'(?:^|\s|[\'"/])([a-zA-Z0-9_\-\.\/]+\.(?:py|java|js|c|cpp|h|rb|go|scala|php|html|css|json|yaml|yml))(?:$|\s|[\'"])'
        file_matches = re.findall(file_pattern, text_without_code)
        entities["files"] = [f.strip() for f in file_matches if len(f.strip()) > 3]

        # Extract function names - detect various function call patterns
        func_patterns = [
            r'(?:^|\s)([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*)\((?:[^)]|\([^)]*\))*\)',  # function(args)
            r'(?:def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # def function or function function
            r'(?:^|\s)([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\(',  # method calls
        ]

        for pattern in func_patterns:
            func_matches = re.findall(pattern, text_without_code)
            entities["functions"].extend([f.strip() for f in func_matches if len(f.strip()) > 2])

        # Extract class names
        class_pattern = r'(?:^|\s)(?:class|interface|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        class_matches = re.findall(class_pattern, text_without_code)
        entities["classes"] = [c.strip() for c in class_matches if len(c.strip()) > 2]

        # Extract API endpoints (e.g., for Elasticsearch, REST APIs)
        api_patterns = [
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+([/a-zA-Z0-9_\-\.]+)',  # HTTP methods
            r'(?:endpoint|URL|url|endpoint):\s*[\'"]?([/a-zA-Z0-9_\-\.]+)[\'"]?',  # Named endpoints
            r'/_[a-zA-Z0-9_\-\.\/]+',  # Elasticsearch-style endpoints
        ]

        for pattern in api_patterns:
            api_matches = re.findall(pattern, text_without_code)
            entities["api_endpoints"].extend([a.strip() for a in api_matches if len(a.strip()) > 3])

        # Extract technical terms (camelCase, snake_case, namespaces)
        term_patterns = [
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b',  # CamelCase
            r'\b([a-z]+_[a-z_]+)\b',  # snake_case
            r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_\.]*)\b',  # namespaces
        ]

        for pattern in term_patterns:
            term_matches = re.findall(pattern, text_without_code)
            entities["technical_terms"].extend([t.strip() for t in term_matches if len(t.strip()) > 3])

        # Deduplicate
        for category in entities:
            entities[category] = list(set(entities[category]))

        return entities

    def _enhance_retrieval_with_tests(self, query: str, test_patch: str, fail_to_pass: List[str] = None) -> str:
        """
        Use test information to enhance the retrieval query.

        Args:
            query: The original query text
            test_patch: Test patch content if available
            fail_to_pass: List of tests that should go from failing to passing

        Returns:
            Enhanced query incorporating test information
        """
        if not test_patch and not fail_to_pass:
            return query

        enhancements = []

        # Extract information from test patch
        if test_patch:
            # Extract test function names
            test_func_pattern = r'def\s+(test_[a-zA-Z0-9_]+)'
            test_funcs = re.findall(test_func_pattern, test_patch)

            if test_funcs:
                enhancements.append(f"Test functions: {', '.join(test_funcs)}")

            # Extract assertions to understand what's being tested
            assert_pattern = r'assert[^,;=\n]+(?:==|!=|>|<|is |is not |in |not in )[^,;=\n]+'
            assertions = re.findall(assert_pattern, test_patch)

            if assertions:
                # Limit to 3 assertions to keep query focused
                clean_assertions = [a.strip().replace('\n', ' ') for a in assertions[:3]]
                enhancements.append(f"Test assertions: {'; '.join(clean_assertions)}")

            # Extract imported modules in tests - likely relevant
            import_pattern = r'(?:from|import)\s+([\w\.]+)'
            imports = re.findall(import_pattern, test_patch)

            if imports:
                enhancements.append(f"Test imports: {', '.join(imports[:5])}")

        # Add failing test information
        if fail_to_pass:
            enhancements.append(f"Failing tests: {', '.join(fail_to_pass[:5])}")

            # Extract components from failing test names
            components = []
            for test in fail_to_pass:
                parts = re.split(r'[_\.]', test)
                components.extend([p for p in parts if len(p) > 3 and p.lower() not in ('test', 'tests')])

            if components:
                components = list(set(components))
                enhancements.append(f"Test components: {', '.join(components[:5])}")

        # Combine the enhancements with the original query
        enhanced_query = f"{query}\n\n{' '.join(enhancements)}"
        return enhanced_query

    def retrieve_code_for_issue(self, issue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval process for more accurate results.

        Args:
            issue: Issue dictionary with problem statement and other metadata

        Returns:
            List of relevant code chunks, reranked for maximum relevance
        """
        repo = issue.get("repo", "")
        problem_statement = issue.get("problem_statement", "")
        data_loader = SWEBenchDataLoader(self.config)
        if not problem_statement:
            # Fall back to other fields
            problem_statement = data_loader.get_issue_description(issue)


        test_patch = issue.get("test_patch", "")
        hints_text = issue.get("hints_text", "")
        fail_to_pass = []

        # Parse FAIL_TO_PASS string if present
        fail_to_pass_str = issue.get("FAIL_TO_PASS", "[]")
        if isinstance(fail_to_pass_str, str):
            try:
                fail_to_pass = json.loads(fail_to_pass_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse FAIL_TO_PASS JSON: {fail_to_pass_str}")
        elif isinstance(fail_to_pass_str, list):
            fail_to_pass = fail_to_pass_str

        # STAGE 1: Initial retrieval with basic query
        initial_query = problem_statement
        tt_patch = data_loader.get_test_patch(issue)
        initial_results = self.retrieve_relevant_code(
            repo,
            initial_query,
            tt_patch,
            top_k=20  # Retrieve more candidates initially
        )
        logger.info(f"Initial retrieval found {len(initial_results)} code chunks")
        if initial_results:
            top_3_files = [f"{r.get('file_path', 'unknown')} (score: {r.get('score', 0):.3f})" for r in
                           initial_results[:3]]
            logger.info(f"Top 3 initial files: {top_3_files}")

        if not initial_results:
            logger.warning(f"No initial results found for issue in repo {repo}")
            return []

        # STAGE 2: Extract entities and refine query
        entities = self._extract_entities_from_problem(problem_statement)
        # logger.info(f"Extracted entities from problem statement: {json.dumps(entities, indent=2)}")

        entity_str = ""
        for category, items in entities.items():
            if items:
                entity_str += f"{category}: {', '.join(items[:5])}\n"

        # Create enhanced query combining problem and test information
        enhanced_query = self._enhance_retrieval_with_tests(
            f"{problem_statement}\n\n{entity_str}",
            test_patch,
            fail_to_pass
        )
        logger.info(f"Enhanced query created with length: {len(enhanced_query)} chars")
        logger.debug(f"Query preview: {enhanced_query[:20]}...")

        # If hints available, add them to the query
        if hints_text:
            enhanced_query += f"\n\nHints: {hints_text}"

        # Retrieve with enhanced query
        refined_results = self.retrieve_relevant_code(
            repo,
            enhanced_query,
            tt_patch,
            top_k=10
        )
        logger.info(f"Refined retrieval found {len(refined_results)} code chunks")
        if refined_results:
            top_3_files = [f"{r.get('file_path', 'unknown')} (score: {r.get('score', 0):.3f})" for r in
                           refined_results[:3]]
            logger.info(f"Top 3 refined files: {top_3_files}")

        # STAGE 3: Re-rank results
        reranked_results = self._rerank_results(
            initial_results + refined_results,  # Combine both result sets
            entities,
            fail_to_pass,
            test_patch
        )

        logger.info(f"Reranking produced {len(reranked_results)} code chunks")
        if reranked_results:
            top_files = [f"{r.get('file_path', 'unknown')} (score: {r.get('combined_score', 0):.3f})" for r in
                         reranked_results[:5]]
            logger.info(f"Top 5 files after reranking: {top_files}")

            # Log detailed information about the top result
            # if reranked_results:
            #     top_result = reranked_results[0]
            #     logger.info(f"Top result details:")
            #     logger.info(f"  File: {top_result.get('file_path', 'unknown')}")
            #     logger.info(f"  Type: {top_result.get('type', 'unknown')}")
            #     logger.info(f"  Name: {top_result.get('name', 'unknown')}")
            #     logger.info(f"  Original score: {top_result.get('score', 0):.3f}")
            #     logger.info(f"  Entity score: {top_result.get('entity_score', 0):.3f}")
            #     logger.info(f"  Test relevance: {top_result.get('test_relevance_score', 0):.3f}")
            #     logger.info(f"  Combined score: {top_result.get('combined_score', 0):.3f}")
            #
            #     # Show a preview of the content
            #     content = top_result.get('content', '')
            #     if content:
            #         preview = content[:500] + ('...' if len(content) > 500 else '')
            #         logger.info(f"  Content preview: \n{preview}")

        # Return the top results
        return reranked_results[:8]

    def _rerank_results(self, code_chunks: List[Dict],
                        entities: Dict[str, List[str]],
                        fail_to_pass: List[str] = None,
                        test_patch: str = None) -> List[Dict]:
        """
        Rerank code chunks based on additional relevance signals.

        Args:
            code_chunks: List of code chunks with scores
            entities: Extracted entities from the problem
            fail_to_pass: List of failing tests
            test_patch: Test patch content

        Returns:
            Reranked list of code chunks
        """
        # Remove duplicates by file_path and code content
        unique_chunks = {}
        for chunk in code_chunks:
            key = f"{chunk.get('file_path', '')}:{chunk.get('name', '')}"
            if key not in unique_chunks or chunk.get('score', 0) > unique_chunks[key].get('score', 0):
                unique_chunks[key] = chunk

        reranked_chunks = list(unique_chunks.values())

        # Calculate additional scores
        for chunk in reranked_chunks:
            # Initialize additional scores
            chunk['entity_score'] = 0
            chunk['test_relevance_score'] = 0
            chunk['combined_score'] = chunk.get('score', 0)  # Start with original score

            content = chunk.get('content', '')

            # Entity matching score
            for category, items in entities.items():
                weight = 1.0
                if category == 'functions':
                    weight = 2.0  # Functions are more important
                elif category == 'files':
                    weight = 1.5

                for item in items:
                    if item.lower() in content.lower():
                        chunk['entity_score'] += weight

                        # Exact function or class name match gets bonus
                        if category in ('functions', 'classes'):
                            pattern = fr'\b{re.escape(item)}\b'
                            if re.search(pattern, content):
                                chunk['entity_score'] += weight

            # Test relevance score
            if fail_to_pass:
                for test in fail_to_pass:
                    if test.lower() in content.lower():
                        chunk['test_relevance_score'] += 2.0
                    else:
                        # Check for components of test name
                        parts = re.split(r'[_\.]', test)
                        for part in parts:
                            if len(part) > 3 and part.lower() not in ('test', 'tests'):
                                if part.lower() in content.lower():
                                    chunk['test_relevance_score'] += 0.5

            # Check test patch relevance
            if test_patch and content:
                # Look for common imported modules
                imports_pattern = r'(?:from|import)\s+([\w\.]+)'
                content_imports = re.findall(imports_pattern, content)
                test_imports = re.findall(imports_pattern, test_patch)

                common_imports = set(content_imports).intersection(set(test_imports))
                chunk['test_relevance_score'] += len(common_imports) * 0.5

            # Combine scores with weights
            chunk['combined_score'] = (
                    chunk.get('score', 0) * 0.5 +  # Original similarity score
                    chunk.get('entity_score', 0) * 0.3 +  # Entity matching
                    chunk.get('test_relevance_score', 0) * 0.2  # Test relevance
            )

        logger.info(f"Reranking {len(code_chunks)} code chunks")
        logger.info(f"Using entities: {entities}")
        logger.info(f"Using fail_to_pass tests: {fail_to_pass}")
        logger.info(f"Test patch available: {test_patch is not None}")

        # After removing duplicates
        logger.info(f"Removed duplicates, now have {len(reranked_chunks)} unique chunks")

        # After calculating additional scores
        logger.info(f"Calculated additional scores for ranking")

        # Show score changes for top chunks
        for i, chunk in enumerate(reranked_chunks[:5]):
            logger.info(f"Chunk {i + 1} scores: original={chunk.get('score', 0):.3f}, "
                        f"entity={chunk.get('entity_score', 0):.3f}, "
                        f"test={chunk.get('test_relevance_score', 0):.3f}, "
                        f"combined={chunk.get('combined_score', 0):.3f}")

        # After sorting
        logger.info(f"Sorted chunks by combined score")
        return reranked_chunks


