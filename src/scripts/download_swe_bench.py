# src/scripts/download_swe_bench.py
#!/usr/bin/env python3

"""
Download SWE-bench Lite dataset from Hugging Face.

This script downloads the SWE-bench Lite dataset and prepares it for use
with the framework.
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, Any, List
import requests
from tqdm import tqdm
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset URLs
DATASET_URLS = {
    "standard": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/raw/main/data.json",
    "oracle": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite_oracle/raw/main/data.json",
    "bm25_13K": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite_bm25_13K/raw/main/data.json",
    "bm25_27K": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite_bm25_27K/raw/main/data.json"
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download SWE-bench Lite dataset")

    parser.add_argument("--dataset-type", type=str, default="standard",
                        choices=["standard", "oracle", "bm25_13K", "bm25_27K"],
                        help="Dataset type to download")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                        help="Directory to save the dataset")
    parser.add_argument("--repos-dir", type=str, default="data/repositories",
                        help="Directory to store repositories")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit the number of examples to download (0 for all)")

    return parser.parse_args()


def download_file(url: str, output_path: str) -> None:
    """
    Download a file from a URL.

    Args:
        url: URL to download from
        output_path: Path to save the file to
    """
    logger.info(f"Downloading {url} to {output_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    file_size = int(response.headers.get("content-length", 0))
    progress = tqdm(total=file_size, unit="B", unit_scale=True)

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))

    progress.close()


def download_dataset(dataset_type: str, output_dir: str, limit: int = 0) -> str:
    """
    Download the SWE-bench Lite dataset.

    Args:
        dataset_type: Type of dataset to download
        output_dir: Directory to save the dataset
        limit: Limit the number of examples (0 for all)

    Returns:
        Path to the downloaded dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get the URL for the specified dataset type
    url = DATASET_URLS.get(dataset_type)
    if not url:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    # Download the dataset
    dataset_path = os.path.join(output_dir, f"swe_bench_lite_{dataset_type}.json")
    download_file(url, dataset_path)

    # Limit the number of examples if specified
    if limit > 0:
        logger.info(f"Limiting dataset to {limit} examples")
        with open(dataset_path, "r") as f:
            data = json.load(f)

        limited_data = data[:limit]

        with open(dataset_path, "w") as f:
            json.dump(limited_data, f, indent=2)

    return dataset_path


def create_config_file(dataset_type: str, output_dir: str, repos_dir: str) -> None:
    """
    Create a configuration file for the dataset.

    Args:
        dataset_type: Type of dataset downloaded
        output_dir: Directory where the dataset is saved
        repos_dir: Directory where repositories will be stored
    """
    dataset_id = f"swe_bench_lite_{dataset_type}"

    config = {
        "id": dataset_id,
        "type": "swe_bench",
        "config": {
            "file_path": os.path.join(output_dir, f"{dataset_id}.json"),
            "auto_load": True,
            "retrieval_type": dataset_type,
            "repos_dir": repos_dir
        }
    }

    # Create the configs directory if it doesn't exist
    configs_dir = "configs/datasets"
    os.makedirs(configs_dir, exist_ok=True)

    # Write the configuration file
    config_path = os.path.join(configs_dir, f"{dataset_id}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    logger.info(f"Created configuration file: {config_path}")


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        # Download the dataset
        dataset_path = download_dataset(args.dataset_type, args.output_dir, args.limit)
        logger.info(f"Dataset downloaded to {dataset_path}")

        # Create the configuration file
        create_config_file(args.dataset_type, args.output_dir, args.repos_dir)

        logger.info("Setup completed successfully")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Add yaml import here to avoid dependency at the module level
    import yaml

    main()
