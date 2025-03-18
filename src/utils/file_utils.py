# src/utils/file_utils.py

import os
import json
import yaml
import tempfile
from typing import Dict, Any, List


def save_json(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to the file
        indent: Indentation level for the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a YAML file.

    Args:
        data: Data to save
        file_path: Path to the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load data from a YAML file.

    Args:
        file_path: Path to the file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_to_temp_file(content: str, suffix: str = ".py") -> str:
    """
    Save content to a temporary file.

    Args:
        content: Content to save
        suffix: File suffix

    Returns:
        Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_filename = tmp_file.name

    return tmp_filename


def remove_file(file_path: str) -> None:
    """
    Remove a file.

    Args:
        file_path: Path to the file
    """
    if os.path.exists(file_path):
        os.remove(file_path)


def ensure_directory(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)


def list_files(directory_path: str, pattern: str = "*") -> List[str]:
    """
    List files in a directory matching a pattern.

    Args:
        directory_path: Path to the directory
        pattern: Glob pattern to match

    Returns:
        List of file paths
    """
    import glob
    return glob.glob(os.path.join(directory_path, pattern))


def read_text_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read text from a file.

    Args:
        file_path: Path to the file
        encoding: File encoding

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def write_text_file(content: str, file_path: str, encoding: str = "utf-8") -> None:
    """
    Write text to a file.

    Args:
        content: Content to write
        file_path: Path to the file
        encoding: File encoding
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return os.path.getsize(file_path)


def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.

    Args:
        file_path: Path to the file

    Returns:
        File extension (without dot)
    """
    return os.path.splitext(file_path)[1].lstrip(".")
