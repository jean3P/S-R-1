"""
Download and preprocess the LeetCode test dataset for use with the solution generator.
"""

import os
import sys
import json
import logging
import argparse
import gzip
import requests
from pathlib import Path

# Add parent directory to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess the LeetCode test dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/leetcode",
        help="Output directory for the preprocessed dataset"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/huggingface_cache",
        help="Cache directory for Huggingface datasets"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v0.3.1",
        help="Version of the dataset to download"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of problems to download"
    )
    parser.add_argument(
        "--direct-download",
        action="store_true",
        help="Download directly from GitHub rather than using Huggingface"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    return parser.parse_args()


def setup_logging(log_level):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )


def direct_download(version, output_dir):
    """Download the test dataset directly from GitHub."""
    github_base_url = "https://github.com/newfacade/LeetCodeDataset/raw/main/data/"
    file_name = f"LeetCodeDataset-{version}-test.jsonl.gz"
    url = f"{github_base_url}{file_name}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / file_name

    logging.info(f"Downloading {url} to {output_file}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Successfully downloaded {file_name}")
        return output_file
    except Exception as e:
        logging.error(f"Error downloading file: {str(e)}")
        return None


def download_dataset(cache_dir, version, limit, use_direct_download, output_dir):
    """Download the LeetCode test dataset."""
    if use_direct_download:
        file_path = direct_download(version, output_dir)
        if file_path is None:
            return None

        problems = []
        count = 0
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            problem = json.loads(line.strip())
                            problems.append(problem)
                            count += 1
                            if limit > 0 and count >= limit:
                                break
                        except json.JSONDecodeError:
                            continue

            logging.info(f"Loaded {len(problems)} problems from direct download")
            return problems
        except Exception as e:
            logging.error(f"Error processing direct download: {str(e)}")
            return None
    else:
        # Use Huggingface datasets
        try:
            from datasets import load_dataset

            logging.info(f"Downloading test split of LeetCode dataset version {version} (max {limit} problems)...")
            dataset = load_dataset(
                "newfacade/LeetCodeDataset",
                split="test",
                cache_dir=cache_dir
            )

            logging.info(f"Successfully downloaded dataset with {len(dataset)} problems")

            # Limit the number of problems if requested
            if limit > 0 and len(dataset) > limit:
                dataset = dataset.select(range(limit))
                logging.info(f"Limited dataset to {len(dataset)} problems")

            # Convert to list of dictionaries
            problems = [problem for problem in dataset]
            return problems

        except Exception as e:
            logging.error(f"Error downloading dataset from Huggingface: {str(e)}")
            return None


def process_dataset(problems, output_dir):
    """Process the dataset and save it to JSON."""
    if problems is None or len(problems) == 0:
        logging.error("No problems to process")
        return

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize and process the problems
    normalized_problems = []
    for i, problem in enumerate(problems):
        try:
            # Extract fields from the problem
            normalized = {
                "problem_id": problem.get("task_id", f"problem_{i}"),
                "prompt": problem.get("prompt", ""),
                "query": problem.get("query", ""),
                "entry_point": problem.get("entry_point", ""),
                "test": problem.get("test", ""),
                "input_output": problem.get("input_output", []),
                "reference_solution": problem.get("completion", "")
            }

            # Extract metadata
            meta = {}
            if "meta" in problem and problem["meta"]:
                if isinstance(problem["meta"], str):
                    try:
                        meta = json.loads(problem["meta"])
                    except json.JSONDecodeError:
                        pass
                elif isinstance(problem["meta"], dict):
                    meta = problem["meta"]

            # Handle top-level meta fields
            meta_fields = ["question_id", "difficulty", "tags", "estimated_date",
                           "question_title", "starter_code", "problem_description", "lang_code"]

            for field in meta_fields:
                if field in problem and problem[field] is not None:
                    meta[field] = problem[field]

            # Add processed metadata to normalized problem
            normalized["difficulty"] = meta.get("difficulty", "")
            normalized["tags"] = meta.get("tags", [])
            normalized["title"] = meta.get("question_title", normalized["problem_id"])
            normalized["estimated_date"] = meta.get("estimated_date", "")
            normalized["starter_code"] = meta.get("lang_code", meta.get("starter_code", ""))

            normalized_problems.append(normalized)
        except Exception as e:
            logging.error(f"Error processing problem {i}: {str(e)}")

    # Save the processed dataset
    output_file = output_dir / "leetcode_problems.json"
    with open(output_file, 'w') as f:
        json.dump(normalized_problems, f, indent=2)

    logging.info(f"Successfully processed and saved {len(normalized_problems)} problems to {output_file}")

    # Also save a simple problem list for reference
    problem_list = []
    for p in normalized_problems:
        problem_list.append({
            "problem_id": p["problem_id"],
            "title": p.get("title", p["problem_id"]),
            "difficulty": p.get("difficulty", ""),
            "tags": p.get("tags", [])
        })

    list_file = output_dir / "problem_list.json"
    with open(list_file, 'w') as f:
        json.dump(problem_list, f, indent=2)

    logging.info(f"Saved problem list to {list_file}")


def main():
    """Main function."""
    args = parse_args()
    setup_logging(args.log_level)

    # Ensure the datasets library is installed if not using direct download
    if not args.direct_download:
        try:
            import datasets
            logging.info(f"Using datasets library version {datasets.__version__}")
        except ImportError:
            logging.error("Huggingface datasets library not installed. Please install it with: pip install datasets")
            return

    # Download the dataset
    problems = download_dataset(args.cache_dir, args.version, args.limit, args.direct_download, args.output_dir)

    # Process the dataset
    process_dataset(problems, args.output_dir)


if __name__ == "__main__":
    main()
