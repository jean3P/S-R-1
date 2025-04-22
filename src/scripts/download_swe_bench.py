# src/scripts/download_swe_bench.py
"""
Download SWE-bench dataset (train, dev, and test) using Hugging Face's datasets library.
"""

import os
import logging
from pathlib import Path
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_split_to_json(split, path: Path):
    df = split.to_pandas()
    df.to_json(path, orient="records", lines=False, force_ascii=False)
    logger.info(f"Saved {len(df)} records to {path}")


def download_swe_bench(output_dir: str = "data/swe-bench-verified") -> None:
    """
    Download the SWE-bench dataset (train, dev, test) and save them as JSON files.

    Args:
        output_dir: Directory where the dataset should be saved
    """
    logger.info("Downloading SWE-bench from Hugging Face...")
    dataset = load_dataset("princeton-nlp/SWE-bench")  # includes train, test, dev

    os.makedirs(output_dir, exist_ok=True)

    for split_name in dataset.keys():
        split = dataset[split_name]
        split_path = Path(output_dir) / f"swe_bench_{split_name}.json"
        logger.info(f"Saving {split_name} split to: {split_path}")
        save_split_to_json(split, split_path)

    logger.info("All splits downloaded and saved.")


if __name__ == "__main__":
    download_swe_bench()
