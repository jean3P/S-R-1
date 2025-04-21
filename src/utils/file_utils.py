# utils/file_utils.py
import os
import shutil
import tempfile
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


class FileUtils:
    """
    Utilities for file operations.
    """

    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """
        Ensure a directory exists.

        Args:
            directory: Directory path.

        Returns:
            Path object for the directory.
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            directory_path.mkdir(parents=True)
        return directory_path

    @staticmethod
    def write_json(data: Any, file_path: Union[str, Path]) -> bool:
        """
        Write data to a JSON file.

        Args:
            data: Data to write.
            file_path: Path to the file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            file_path = Path(file_path)
            FileUtils.ensure_directory(file_path.parent)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing JSON file: {str(e)}")
            return False

    @staticmethod
    def read_json(file_path: Union[str, Path]) -> Optional[Any]:
        """
        Read data from a JSON file.

        Args:
            file_path: Path to the file.

        Returns:
            Loaded data or None if error.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON file: {str(e)}")
            return None

    @staticmethod
    def write_file(content: str, file_path: Union[str, Path]) -> bool:
        """
        Write content to a file.

        Args:
            content: Content to write.
            file_path: Path to the file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            file_path = Path(file_path)
            FileUtils.ensure_directory(file_path.parent)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing file: {str(e)}")
            return False

    @staticmethod
    def read_file(file_path: Union[str, Path]) -> Optional[str]:
        """
        Read content from a file.

        Args:
            file_path: Path to the file.

        Returns:
            File content or None if error.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return None
