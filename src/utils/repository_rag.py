# src/utils/repository_rag.py

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
                               top_k: int = 5,
                               include_content: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant code chunks for a query.

        Args:
            repo_name: Name of the repository.
            query: The query text (issue description).
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

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search the index
        distances, indices = index.search(
            np.array([query_embedding]).astype(np.float32),
            min(top_k, index.ntotal)
        )

        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue

            result = metadata[idx].copy()
            result["score"] = float(1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score

            # Optionally remove content to save memory
            if not include_content and "content" in result:
                del result["content"]

            results.append(result)

        return results

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
        if not description:
            logger.error("No issue description found")
            return {"error": "No issue description found"}

        logger.info(f"Analyzing issue for repository: {repo}")

        # Extract key terms for improved retrieval
        key_terms = self._extract_key_terms(description)

        # Create combined query with key terms for better retrieval
        query = f"{description}\n\nKey terms: {', '.join(key_terms)}"

        # Retrieve relevant code chunks
        relevant_chunks = self.retrieve_relevant_code(repo, query, top_k=top_k)

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
