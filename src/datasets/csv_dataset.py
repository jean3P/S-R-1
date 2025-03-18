# src/datasets/csv_dataset.py

import os
import csv
from typing import Dict, Any, Iterator

from src.datasets.base_dataset import BaseDataset


class CSVDataset(BaseDataset):
    """Dataset implementation for CSV files."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CSV dataset.

        Args:
            config: Dataset configuration
        """
        super().__init__(config)

        # Extract configuration
        self.file_path = config.get("file_path")
        if not self.file_path:
            raise ValueError("CSV dataset requires a file_path")

        self.auto_load = config.get("auto_load", True)
        self.delimiter = config.get("delimiter", ",")
        self.quotechar = config.get("quotechar", '"')
        self.has_header = config.get("has_header", True)
        self.header = config.get("header", None)
        self.key_field = config.get("key_field")
        self.convert_types = config.get("convert_types", True)

        # Initialize data structure
        self.data = None

        # Auto-load if configured
        if self.auto_load:
            self.load()

    def load(self) -> None:
        """
        Load the dataset from a CSV file.

        Raises:
            FileNotFoundError: If the CSV file does not exist
            ValueError: If the CSV format is invalid
        """
        if not os.path.exists(self.file_path):
            self.logger.error(f"CSV file not found: {self.file_path}")
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        try:
            self.data = []

            # Open the CSV file
            with open(self.file_path, 'r', encoding='utf-8', newline='') as f:
                # Create CSV reader
                reader = csv.reader(f, delimiter=self.delimiter, quotechar=self.quotechar)

                # Read header if present
                if self.has_header:
                    header = next(reader)
                    # Strip whitespace from headers
                    header = [h.strip() for h in header]
                elif self.header:
                    header = self.header
                else:
                    # If no header provided, use column indices
                    first_row = next(reader)
                    header = [f"column_{i}" for i in range(len(first_row))]
                    # Reset file pointer to read first row as data
                    f.seek(0)
                    reader = csv.reader(f, delimiter=self.delimiter, quotechar=self.quotechar)
                    if self.has_header:
                        next(reader)  # Skip header row

                # Read data rows
                for row in reader:
                    # Skip empty rows
                    if not row:
                        continue

                    # Create a dictionary for the row
                    example = {}
                    for i, value in enumerate(row):
                        if i < len(header):
                            field_name = header[i]
                            # Convert types if configured
                            if self.convert_types:
                                value = self._convert_value(value)
                            example[field_name] = value

                    self.data.append(example)

            self._loaded = True
            self.logger.info(f"Loaded {len(self.data)} examples from {self.file_path}")

        except csv.Error as e:
            self.logger.error(f"Invalid CSV format in {self.file_path}: {str(e)}")
            raise ValueError(f"Invalid CSV format in {self.file_path}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error loading CSV dataset: {str(e)}")
            raise

    def save(self) -> None:
        """
        Save the dataset to a CSV file.

        Raises:
            IOError: If the CSV file cannot be saved
        """
        if self.data is None:
            self.logger.warning("No data to save")
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            # Open the CSV file
            with open(self.file_path, 'w', encoding='utf-8', newline='') as f:
                # Determine header
                if self.header:
                    header = self.header
                else:
                    # Get all keys from all examples
                    all_keys = set()
                    for example in self.data:
                        all_keys.update(example.keys())

                    # Sort keys for consistent output
                    header = sorted(list(all_keys))

                    # If key_field is specified, put it first
                    if self.key_field and self.key_field in header:
                        header.remove(self.key_field)
                        header.insert(0, self.key_field)

                # Create CSV writer
                writer = csv.writer(f, delimiter=self.delimiter, quotechar=self.quotechar, quoting=csv.QUOTE_MINIMAL)

                # Write header
                if self.has_header:
                    writer.writerow(header)

                # Write data rows
                for example in self.data:
                    row = []
                    for field in header:
                        row.append(example.get(field, ""))
                    writer.writerow(row)

            self.logger.info(f"Saved {len(self.data)} examples to {self.file_path}")

        except Exception as e:
            self.logger.error(f"Error saving CSV dataset: {str(e)}")
            raise IOError(f"Error saving CSV dataset: {str(e)}")

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

    def _convert_value(self, value: str) -> Any:
        """
        Convert a string value to an appropriate type.

        Args:
            value: String value to convert

        Returns:
            Converted value
        """
        # Trim whitespace
        value = value.strip()

        # Return empty string as is
        if value == "":
            return value

        # Try to convert to int
        try:
            int_value = int(value)
            return int_value
        except ValueError:
            pass

        # Try to convert to float
        try:
            float_value = float(value)
            return float_value
        except ValueError:
            pass

        # Handle true/false values
        if value.lower() in ('true', 'yes', 'y', '1'):
            return True
        if value.lower() in ('false', 'no', 'n', '0'):
            return False

        # Return string as is
        return value
