# src/datasets/json_dataset.py

import os
import json
from typing import Dict, Any, List, Iterator

from src.datasets.base_dataset import BaseDataset


class JSONDataset(BaseDataset):
    """Dataset implementation for JSON files."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the JSON dataset.

        Args:
            config: Dataset configuration
        """
        super().__init__(config)

        # Extract configuration
        self.file_path = config.get("file_path")
        if not self.file_path:
            raise ValueError("JSON dataset requires a file_path")

        self.auto_load = config.get("auto_load", True)
        self.key_field = config.get("key_field")
        self.data_field = config.get("data_field")

        # Initialize data structure
        self.data = None

        # Auto-load if configured
        if self.auto_load:
            self.load()

    def load(self) -> None:
        """
        Load the dataset from a JSON file.

        Raises:
            FileNotFoundError: If the JSON file does not exist
            ValueError: If the JSON format is invalid
        """
        if not os.path.exists(self.file_path):
            self.logger.error(f"JSON file not found: {self.file_path}")
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # Handle different JSON structures
            if self.data_field:
                # Extract data from a specific field
                if self.data_field in loaded_data:
                    loaded_data = loaded_data[self.data_field]
                else:
                    self.logger.warning(f"Data field '{self.data_field}' not found in JSON")
                    loaded_data = []

            # Convert to list if it's a dictionary
            if isinstance(loaded_data, dict):
                if self.key_field:
                    # Add the key as a field in each item
                    self.data = [
                        {self.key_field: key, **value}
                        for key, value in loaded_data.items()
                    ]
                else:
                    # Just use the values
                    self.data = list(loaded_data.values())
            elif isinstance(loaded_data, list):
                self.data = loaded_data
            else:
                self.logger.error(f"Unsupported JSON structure: {type(loaded_data)}")
                raise ValueError(f"Unsupported JSON structure: {type(loaded_data)}")

            self._loaded = True
            self.logger.info(f"Loaded {len(self.data)} examples from {self.file_path}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in {self.file_path}: {str(e)}")
            raise ValueError(f"Invalid JSON format in {self.file_path}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error loading JSON dataset: {str(e)}")
            raise

    def save(self) -> None:
        """
        Save the dataset to a JSON file.

        Raises:
            IOError: If the JSON file cannot be saved
        """
        if self.data is None:
            self.logger.warning("No data to save")
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            # Determine output format
            output_data = self.data

            # Convert to dictionary if key_field is specified
            if self.key_field and all(self.key_field in example for example in self.data):
                output_data = {
                    example[self.key_field]: {
                        k: v for k, v in example.items() if k != self.key_field
                    }
                    for example in self.data
                }

            # Wrap in data_field if specified
            if self.data_field:
                output_data = {self.data_field: output_data}

            # Write to file
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(self.data)} examples to {self.file_path}")

        except Exception as e:
            self.logger.error(f"Error saving JSON dataset: {str(e)}")
            raise IOError(f"Error saving JSON dataset: {str(e)}")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the dataset examples.

        Returns:
            Iterator over examples
        """
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
        if self.data is None:
            if not self._loaded:
                self.load()
            if self.data is None:
                return 0

        return len(self.data)

    def add_example(self, example: Dict[str, Any]) -> None:
        """
        Add an example to the dataset.

        Args:
            example: Example to add
        """
        if self.data is None:
            if not self._loaded:
                self.load()
            if self.data is None:
                self.data = []

        self.data.append(example)

    def remove_example(self, index: int) -> Dict[str, Any]:
        """
        Remove an example from the dataset.

        Args:
            index: Index of the example to remove

        Returns:
            Removed example

        Raises:
            IndexError: If the index is out of range
        """
        if self.data is None:
            if not self._loaded:
                self.load()
            if self.data is None:
                raise ValueError("Dataset not loaded")

        if index >= len(self.data):
            raise IndexError(f"Index {index} out of range for dataset with {len(self.data)} examples")

        return self.data.pop(index)

    def find(self, key_field: str, value: Any) -> List[Dict[str, Any]]:
        """
        Find examples that match a key-value pair.

        Args:
            key_field: Field to match
            value: Value to match

        Returns:
            List of matching examples
        """
        if self.data is None:
            if not self._loaded:
                self.load()
            if self.data is None:
                return []

        return [example for example in self.data if key_field in example and example[key_field] == value]
