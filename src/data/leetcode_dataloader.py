import logging
import os
import gzip
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class LeetCodeDataLoader:
    """
    Loader for the LeetCodeDataset from local repository.
    Specifically designed to load only the test split.
    """

    def __init__(self, config):
        """
        Initialize LeetCode data loader.

        Args:
            config: Configuration object.
        """
        self.config = config

        # Extract leetcode and huggingface specific configs
        self.leetcode_config = config.get("leetcode", {})
        self.hf_config = config.get("huggingface", {})

        # Set up cache directory
        self.cache_dir = Path(self.leetcode_config.get("cache_dir", "data/huggingface_cache"))
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set max problems to load
        self.max_problems = self.leetcode_config.get("max_problems", 100)

        # Get repository path from config or use default
        self.repo_path = config.get("data", {}).get("leetcode_repo_path", "./")
        if isinstance(self.repo_path, str):
            self.repo_path = Path(self.repo_path)

        # Support for specifying dataset version
        self.dataset_version = self.leetcode_config.get("version", "v0.3.1")

        # Load dataset
        self.dataset = self._load_dataset()

    @property
    def test_dataset_file(self):
        """Generate test dataset filename based on the specified version."""
        return f"LeetCodeDataset-{self.dataset_version}-test.jsonl.gz"

    def _normalize_problem_data(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize problem data to handle different schema versions.

        Args:
            problem: Raw problem data

        Returns:
            Normalized problem data
        """
        normalized = problem.copy()

        # Handle meta information
        meta = {}
        if "meta" in problem and problem["meta"]:
            # If meta is a string, try to parse it
            if isinstance(problem["meta"], str):
                try:
                    meta = json.loads(problem["meta"])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse meta string for problem {problem.get('task_id', 'unknown')}")
            elif isinstance(problem["meta"], dict):
                meta = problem["meta"]

        # Move top-level meta fields into meta dict if they exist
        meta_fields = ["question_id", "difficulty", "tags", "estimated_date",
                       "question_title", "starter_code", "problem_description", "lang_code"]

        for field in meta_fields:
            if field in problem and problem[field] is not None:
                meta[field] = problem[field]

        # Replace or create meta field
        normalized["meta"] = meta

        return normalized

    def _load_dataset(self):
        """
        Load only the test split of the LeetCode dataset from the local repository.
        """
        try:
            from datasets import Dataset

            # Construct path to the data file
            data_dir = self.repo_path / "data"
            file_path = data_dir / self.test_dataset_file

            logger.info(f"Loading test dataset from repository file: {file_path}")

            # First try the direct UBELIX path from your file structure
            ubelix_path = Path(
                f"/storage/homefs/jp22b083/SSI/S-R-1/data/repositories/LeetCodeDataset/data/{self.test_dataset_file}")
            if ubelix_path.exists():
                file_path = ubelix_path
                logger.info(f"Using UBELIX cluster path: {file_path}")
            elif not file_path.exists():
                # If file doesn't exist in the repo, try alternate paths
                logger.warning(f"Dataset file not found at {file_path}")

                # Try in the current directory
                if Path(self.test_dataset_file).exists():
                    file_path = Path(self.test_dataset_file)
                    logger.info(f"Using test dataset file in current directory: {file_path}")
                else:
                    # Look specifically for the test file in the data directory and its subdirectories
                    found = False
                    for root, _, files in os.walk(str(self.repo_path)):
                        for filename in files:
                            if filename == self.test_dataset_file:
                                file_path = Path(os.path.join(root, filename))
                                logger.info(f"Found test dataset file at: {file_path}")
                                found = True
                                break
                        if found:
                            break

                    # If still not found, try with the repo_path directly
                    if not found and os.path.exists(os.path.join(self.repo_path, self.test_dataset_file)):
                        file_path = Path(os.path.join(self.repo_path, self.test_dataset_file))
                        logger.info(f"Found test dataset file at root of repo: {file_path}")
                        found = True

                    if not found:
                        logger.error(f"Test dataset file not found: {self.test_dataset_file}")
                        return None

            # Create a generator that reads and normalizes lines from the gzipped file
            def generate_from_gzip():
                count = 0
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line.strip())
                                yield self._normalize_problem_data(item)

                                count += 1
                                if self.max_problems and count >= self.max_problems:
                                    break
                            except json.JSONDecodeError as e:
                                logger.warning(f"Error parsing JSON line: {str(e)}")
                                continue

            # Create dataset from generator
            dataset = Dataset.from_generator(generate_from_gzip)

            logger.info(f"Loaded {len(dataset)} problems from the test dataset")
            return dataset

        except Exception as e:
            logger.error(f"Error loading LeetCode test dataset: {str(e)}")
            return None

    def load_problem(self, problem_idx: int) -> Optional[Dict[str, Any]]:
        """
        Load a specific problem by index.

        Args:
            problem_idx: Index of the problem to load.

        Returns:
            Dictionary containing problem information.
        """
        if self.dataset is None or problem_idx >= len(self.dataset):
            logger.warning(f"Problem index {problem_idx} not found in dataset")
            return None

        problem = self.dataset[problem_idx]
        meta = problem.get("meta", {})

        # Prepare problem data
        problem_data = {
            "problem_id": problem.get("task_id", ""),
            "problem_title": meta.get("question_title", ""),
            "difficulty": meta.get("difficulty", ""),
            "tags": meta.get("tags", []),
            "prompt": problem.get("prompt", ""),
            "query": problem.get("query", ""),
            "entry_point": problem.get("entry_point", ""),
            "test": problem.get("test", ""),
            "input_output": problem.get("input_output", []),
            "reference_solution": problem.get("completion", ""),
            "estimated_date": meta.get("estimated_date", ""),
            "starter_code": meta.get("lang_code", meta.get("starter_code", ""))
        }

        # Add timeout and retry configurations
        problem_data["test_timeout"] = self.leetcode_config.get("test_timeout", 10)
        problem_data["max_test_retries"] = self.leetcode_config.get("max_test_retries", 2)

        return problem_data

    def load_problem_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific problem by task_id.

        Args:
            task_id: The LeetCode problem ID (e.g., "two-sum").

        Returns:
            Dictionary containing problem information.
        """
        if self.dataset is None:
            logger.warning("Dataset not loaded")
            return None

        # Find problem with matching task_id
        for i, problem in enumerate(self.dataset):
            if problem.get("task_id") == task_id:
                return self.load_problem(i)

        logger.warning(f"Problem with task_id {task_id} not found in dataset")
        return None

    def list_problems(self, difficulty: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List available problems, optionally filtered by difficulty.

        Args:
            difficulty: Filter by difficulty ("Easy", "Medium", "Hard").
            limit: Maximum number of problems to return.

        Returns:
            List of problem summaries.
        """
        if self.dataset is None:
            return []

        problems = []
        count = 0

        for problem in self.dataset:
            meta = problem.get("meta", {})

            # Get difficulty
            problem_difficulty = meta.get("difficulty", "")

            # Filter by difficulty if requested
            if difficulty and problem_difficulty != difficulty:
                continue

            # Add problem to list
            problems.append({
                "problem_id": problem.get("task_id", ""),
                "title": meta.get("question_title", problem.get("task_id", "")),
                "difficulty": problem_difficulty,
                "tags": meta.get("tags", []),
                "estimated_date": meta.get("estimated_date", "")
            })

            count += 1
            if count >= limit:
                break

        return problems
