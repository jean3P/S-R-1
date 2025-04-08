# src/datasets/swe_bench.py

from typing import Dict, Any, Optional, Iterator
import json
import os
from src.datasets.base_dataset import BaseDataset


class SWEBenchDataset(BaseDataset):
    """Dataset implementation for SWE-bench with memory optimization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SWE-bench dataset with memory optimizations.

        Args:
            config: Dataset configuration
        """
        super().__init__(config)

        # Extract additional configuration
        self.dataset_name = config.get("dataset_name", "princeton-nlp/SWE-bench_Verified")  # Updated to Verified
        self.retrieval_type = config.get("retrieval_type", "standard")
        self.repos_dir = config.get("repos_dir", "data/repositories")
        self.cache_dir = config.get("cache_dir", "data/datasets")

        # Memory optimization flag - only load problem identifiers first
        self.lazy_loading = config.get("lazy_loading", True)

        # Define validation schema
        self.schema = {
            "instance_id": {"type": str, "required": True},
            "patch": {"type": str, "required": True},
            "repo": {"type": str, "required": True},
            "base_commit": {"type": str, "required": True},
            "hints_text": {"type": str, "required": False},
            "created_at": {"type": str, "required": True},
            "test_patch": {"type": str, "required": False},
            "problem_statement": {"type": str, "required": True},
            "version": {"type": str, "required": True},
            "environment_setup_commit": {"type": str, "required": True},
            "FAIL_TO_PASS": {"type": str, "required": True},
            "PASS_TO_PASS": {"type": str, "required": True}
        }

        # Initialize data structures
        self.data = None
        self._loaded = False
        self._instance_ids = []
        self._file_path = config.get("file_path")

        if config.get("auto_load", True):
            # With lazy loading, just load instance IDs first
            if self.lazy_loading:
                self._load_instance_ids()
            else:
                self.load()

    def _load_instance_ids(self) -> None:
        """
        Load only the instance IDs from the dataset file to save memory.
        """
        try:
            if self._file_path and os.path.exists(self._file_path):
                self.logger.info(f"Loading instance IDs from: {self._file_path}")

                try:
                    # Stream-process the JSON file to extract only instance IDs
                    self._instance_ids = []
                    with open(self._file_path, 'r', encoding='utf-8') as f:
                        # First character should be '[' for a JSON array
                        if f.read(1) == '[':
                            # Reset file pointer
                            f.seek(0)

                            # Simple streaming JSON processing
                            import json
                            data = json.load(f)
                            self._instance_ids = [item.get('instance_id') for item in data if 'instance_id' in item]

                    self.logger.info(f"Loaded {len(self._instance_ids)} instance IDs")
                    self._loaded = True
                except Exception as e:
                    self.logger.error(f"Error loading instance IDs: {str(e)}")
                    # Fallback to full loading if we can't extract IDs
                    self.load()
            else:
                self.logger.warning(f"File not found: {self._file_path}, falling back to full load")
                self.load()
        except Exception as e:
            self.logger.error(f"Error in lazy loading: {str(e)}")
            # Fallback to full loading
            self.load()

    def load(self) -> None:
        """
        Load the dataset from Hugging Face or local file with memory optimizations.
        """
        try:
            # Check if we have a local file path first
            file_path = self.config.get("file_path")
            if file_path and os.path.exists(file_path):
                self.logger.info(f"Loading SWE-bench dataset from local file: {file_path}")

                try:
                    # Load the file with minimal memory usage
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.data = json.load(f)

                    # Extract instance IDs for later use
                    self._instance_ids = [item.get('instance_id') for item in self.data if 'instance_id' in item]

                    self._loaded = True
                    self.logger.info(f"Loaded {len(self.data)} examples from local file")
                    return
                except Exception as e:
                    self.logger.error(f"Error loading from file: {str(e)}")
                    # Continue to try Hugging Face

            # Otherwise load from Hugging Face datasets
            try:
                from datasets import load_dataset

                self.logger.info(f"Loading SWE-bench dataset from Hugging Face: {self.dataset_name}")

                # Try to load with streaming for memory efficiency
                try:
                    dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir, streaming=True)

                    # Convert streaming dataset to a list (memory efficient)
                    if 'train' in dataset:
                        self.data = list(dataset['train'])
                    else:
                        # Just use the first split available
                        first_split = next(iter(dataset))
                        self.data = list(dataset[first_split])
                except Exception as e:
                    self.logger.warning(f"Streaming load failed: {str(e)}. Falling back to standard loading.")
                    # Fall back to non-streaming load
                    dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)

                    # Convert to our format
                    if 'train' in dataset:
                        self.data = [example for example in dataset['train']]
                    else:
                        # Just use the first split available
                        first_split = next(iter(dataset))
                        self.data = [example for example in dataset[first_split]]

                # Extract instance IDs for later use
                self._instance_ids = [item.get('instance_id') for item in self.data if 'instance_id' in item]

                self._loaded = True
                self.logger.info(f"Loaded {len(self.data)} examples from Hugging Face")

                # Save to local file for future use (if not too large)
                if file_path:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.data, f, indent=2)
                    self.logger.info(f"Saved dataset to local file: {file_path}")

            except ImportError:
                self.logger.error("Could not import 'datasets' module. Please install it using: pip install datasets")
                raise
            except Exception as e:
                self.logger.error(f"Error loading dataset from Hugging Face: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"Error loading SWE-bench dataset: {str(e)}")
            raise

    def _load_specific_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific instance from the dataset file without loading everything."""
        if not self._file_path or not os.path.exists(self._file_path):
            self.logger.warning("No file path available or file not found")
            return None

        try:
            with open(self._file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if item.get('instance_id') == instance_id:
                        return item
                return None
        except Exception as e:
            self.logger.error(f"Error loading specific instance {instance_id}: {str(e)}")
            return None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the dataset examples with memory efficiency.

        Returns:
            Iterator over examples
        """
        if self.lazy_loading and self._instance_ids:
            # For lazy loading, we'll generate instances on demand
            for instance_id in self._instance_ids:
                instance = self._load_specific_instance(instance_id)
                if instance:
                    yield instance
        else:
            # Traditional approach
            if self.data is None:
                if not self._loaded:
                    self.load()
                if self.data is None:
                    return iter([])

            return iter(self.data)

    def __len__(self) -> int:
        """
        Return the number of examples in the dataset.

        Returns:
            Number of examples
        """
        if self.lazy_loading and self._instance_ids:
            return len(self._instance_ids)

        if self.data is None:
            if not self._loaded:
                self.load()
            if self.data is None:
                return 0

        return len(self.data)

    def get_problem_by_instance_id(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get a problem by instance_id with memory efficiency."""
        # Memory-efficient version for lazy loading
        if self.lazy_loading:
            if instance_id in self._instance_ids:
                return self._load_specific_instance(instance_id)
            return None

        # Traditional approach
        matches = self.filter(lambda example: example.get("instance_id") == instance_id)
        return matches[0] if matches else None

    def prepare_for_agent(self, instance_id: str) -> Dict[str, Any]:
        """Prepare a problem for use by an agent."""
        problem = self.get_problem_by_instance_id(instance_id)
        if not problem:
            raise ValueError(f"Problem with instance_id {instance_id} not found")

        # Ensure we have the required fields
        if not problem.get("repo"):
            raise ValueError(f"Problem {instance_id} is missing 'repo' field")
        if not problem.get("base_commit"):
            raise ValueError(f"Problem {instance_id} is missing 'base_commit' field")
        if not problem.get("environment_setup_commit"):
            raise ValueError(f"Problem {instance_id} is missing 'environment_setup_commit' field")

        # Parse test information safely
        try:
            fail_to_pass = json.loads(problem.get("FAIL_TO_PASS", "[]"))
            pass_to_pass = json.loads(problem.get("PASS_TO_PASS", "[]"))
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing test info for {instance_id}: {e}")
            fail_to_pass = []
            pass_to_pass = []

        # Format the problem as a task for the agent
        task = {
            "name": instance_id,
            "language": "python",
            "initial_prompt": self._format_prompt(problem),
            "repo_info": {
                "repo": problem.get("repo"),
                "base_commit": problem.get("base_commit"),
                "environment_setup_commit": problem.get("environment_setup_commit")
            },
            "test_info": {
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass
            }
        }

        # Log the task structure for debugging
        self.logger.info(f"Prepared task for {instance_id} with keys: {list(task.keys())}")
        self.logger.info(f"repo_info keys: {list(task['repo_info'].keys())}")
        self.logger.info(f"test_info keys: {list(task['test_info'].keys())}")

        return task

    def _format_prompt(self, problem: Dict[str, Any]) -> str:
        """Format a problem as a prompt for the agent."""
        prompt = f"# GitHub Issue: {problem.get('instance_id')}\n\n"
        prompt += problem.get("problem_statement", "")

        # Add hints if available
        if problem.get("hints_text"):
            prompt += f"\n\n# Additional context and hints:\n{problem.get('hints_text')}"

        # Add repository information
        prompt += f"\n\n# Repository: {problem.get('repo')}"
        prompt += f"\n# Base commit: {problem.get('base_commit')}"

        return prompt

    def save(self) -> None:
        """
        Save the dataset to a file with memory efficiency.

        Raises:
            IOError: If the dataset cannot be saved
        """
        if self.data is None and not self._instance_ids:
            self.logger.warning("No data to save")
            return

        file_path = self.config.get("file_path")
        if not file_path:
            self.logger.warning("No file path specified, cannot save dataset")
            return

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save either full data or just instance IDs depending on what's loaded
            if self.data is not None:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=2)
                self.logger.info(f"Saved {len(self.data)} examples to {file_path}")
            elif self._instance_ids:
                # Create minimal dataset with just instance IDs for later retrieval
                minimal_data = [{"instance_id": instance_id} for instance_id in self._instance_ids]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(minimal_data, f, indent=2)
                self.logger.info(f"Saved {len(self._instance_ids)} instance IDs to {file_path}")

        except Exception as e:
            self.logger.error(f"Error saving dataset: {str(e)}")
            raise IOError(f"Error saving dataset: {str(e)}")
