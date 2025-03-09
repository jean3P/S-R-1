# src/datasets/base_dataset.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Iterator, Tuple


class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset.

        Args:
            config: Dataset configuration
        """
        from src.utils.logging import get_logger

        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.data = None
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """
        Load the dataset.

        Raises:
            FileNotFoundError: If the dataset file does not exist
            ValueError: If the dataset format is invalid
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Save the dataset.

        Raises:
            IOError: If the dataset cannot be saved
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the dataset examples.

        Returns:
            Iterator over examples
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of examples in the dataset.

        Returns:
            Number of examples
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an example by index.

        Args:
            idx: Index of the example

        Returns:
            Example

        Raises:
            IndexError: If the index is out of range
        """
        if self.data is None:
            if not self._loaded:
                self.load()
            if self.data is None:
                raise ValueError("Dataset not loaded")

        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.data)} examples")

        return self.data[idx]

    def filter(self, condition: callable) -> List[Dict[str, Any]]:
        """
        Filter the dataset based on a condition.

        Args:
            condition: Function that takes an example and returns a boolean

        Returns:
            Filtered examples
        """
        if self.data is None:
            if not self._loaded:
                self.load()
            if self.data is None:
                raise ValueError("Dataset not loaded")

        return [example for example in self.data if condition(example)]

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1,
              test_ratio: float = 0.1, shuffle: bool = True) -> Tuple[List[Dict[str, Any]], ...]:
        """
        Split the dataset into train, validation, and test sets.

        Args:
            train_ratio: Ratio of examples for training
            val_ratio: Ratio of examples for validation
            test_ratio: Ratio of examples for testing
            shuffle: Whether to shuffle the data before splitting

        Returns:
            Tuple of (train_data, val_data, test_data)

        Raises:
            ValueError: If the ratios don't sum to 1
        """
        import random

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1, got {train_ratio + val_ratio + test_ratio}")

        if self.data is None:
            if not self._loaded:
                self.load()
            if self.data is None:
                raise ValueError("Dataset not loaded")

        # Make a copy of the data
        data = list(self.data)

        # Shuffle if requested
        if shuffle:
            random.shuffle(data)

        # Calculate split indices
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split the data
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        return train_data, val_data, test_data

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary of statistics
        """
        if self.data is None:
            if not self._loaded:
                self.load()
            if self.data is None:
                raise ValueError("Dataset not loaded")

        stats = {
            "num_examples": len(self.data),
            "keys": set()
        }

        # Get all keys in the dataset
        for example in self.data:
            stats["keys"].update(example.keys())

        # Convert set to sorted list for better readability
        stats["keys"] = sorted(list(stats["keys"]))

        return stats