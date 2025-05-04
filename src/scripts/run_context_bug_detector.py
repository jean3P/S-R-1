#src/scripts/run_context_bug_detector.py

"""
Run the Enhanced Context-Based Bug Detector on Astropy synthetic dataset issues.

This script implements the bug detection approach described in the paper
"Improving Bug Detection via Context-Based Code Representation Learning
and Attention-Based Neural Networks" by Li et al.
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path

# Add parent directory to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data.astropy_synthetic_dataloader import AstropySyntheticDataLoader
from src.utils.context_based_bug_detector import EnhancedContextBasedBugDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Enhanced Context-Based Bug Detector on Astropy synthetic dataset"
    )

    parser.add_argument(
        "--issues",
        type=str,
        help="Comma-separated list of issue IDs (branch names)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of issues to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/enhanced_bug_detector",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--include-pdg",
        action="store_true",
        help="Include Program Dependency Graph analysis"
    )
    parser.add_argument(
        "--include-dfg",
        action="store_true",
        help="Include Data Flow Graph analysis"
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Use attention mechanism for weighting buggy paths"
    )

    return parser.parse_args()


def setup_logging(log_level, output_dir):
    """Set up logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"enhanced_bug_detector_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging initialized at {log_file}")
    return log_file


def create_config(args):
    """Create a configuration dictionary based on command-line arguments."""
    config = {
        "data": {
            "astropy_dataset_path": "/storage/homefs/jp22b083/SSI/S-R-1",
            "repositories": "/storage/homefs/jp22b083/SSI/S-R-1/data/repositories",
            "cache_dir": "/storage/homefs/jp22b083/SSI/S-R-1/data/cache",
            "file_path": "/storage/homefs/jp22b083/SSI/S-R-1/astropy_synthetic_dataset.csv"
        },
        "models": {
            "device": "cpu",
            "embedding_dim": 100,
            "attention_heads": 4
        },
        "evaluation": {
            "results_dir": args.output
        },
        "features": {
            "use_pdg": args.include_pdg,
            "use_dfg": args.include_dfg,
            "use_attention": args.use_attention
        }
    }
    return config


def run_bug_detector(args):
    """Run the enhanced bug detector on specified issues."""
    # Create configuration
    config = create_config(args)

    # Set up logging
    log_file = setup_logging(args.log_level, config["evaluation"]["results_dir"])

    # Create results directory
    results_dir = Path(config["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data loader
    data_loader = AstropySyntheticDataLoader(config)

    # Initialize the bug detector
    bug_detector = EnhancedContextBasedBugDetector(config)

    # Log the features being used
    feature_msg = "Using features: "
    if config["features"]["use_pdg"]:
        feature_msg += "Program Dependency Graph, "
    if config["features"]["use_dfg"]:
        feature_msg += "Data Flow Graph, "
    if config["features"]["use_attention"]:
        feature_msg += "Attention Mechanism, "
    logging.info(feature_msg.rstrip(", "))

    # Load issue IDs
    if args.issues:
        issue_ids = args.issues.split(",")
    else:
        # Load all issues from dataset
        all_issues = data_loader.load_dataset()
        logging.info(f"Loaded {len(all_issues)} issues from dataset")
        # Take first N issues
        issue_limit = min(args.limit, len(all_issues))
        issue_ids = []
        for i, issue in enumerate(all_issues[:issue_limit]):
            issue_id = issue.get("branch_name") or issue.get("instance_id")
            if issue_id:
                issue_ids.append(issue_id)

    logging.info(f"Processing {len(issue_ids)} issues: {issue_ids}")

    # Process each issue
    results = []

    for issue_id in issue_ids:
        logging.info(f"Processing issue {issue_id}")

        try:
            # Track start time
            start_time = time.time()

            # Detect bug location
            bug_location = bug_detector.detect_bug_location(issue_id)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Add issue ID and timing information to result
            bug_location["issue_id"] = issue_id
            bug_location["processing_time"] = elapsed_time

            logging.info(f"Detected bug location for issue {issue_id} in {elapsed_time:.2f} seconds")
            logging.info(f"Bug type: {bug_location.get('bug_type', 'unknown')}")

            if bug_location.get("involves_multiple_methods"):
                logging.info("Bug involves multiple methods")

            # Add result
            results.append(bug_location)

            # Save individual result
            result_file = results_dir / f"bug_location_{issue_id}.json"
            with open(result_file, 'w') as f:
                json.dump(bug_location, f, indent=2)

            logging.info(f"Saved bug location to {result_file}")

        except Exception as e:
            logging.error(f"Error processing issue {issue_id}: {str(e)}", exc_info=True)
            results.append({
                "issue_id": issue_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            })

    # Save combined results
    combined_file = results_dir / "bug_locations.json"
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved combined results to {combined_file}")

    # Generate summary
    generate_summary(results, results_dir)

    return results


def generate_summary(results, results_dir):
    """Generate a summary of the results."""
    summary = {
        "total_issues": len(results),
        "successful_issues": 0,
        "errors": 0,
        "avg_confidence": 0.0,
        "avg_time": 0.0,
        "bug_types": {},
        "multiple_method_bugs": 0
    }

    # Calculate statistics
    success_count = 0
    error_count = 0
    total_confidence = 0.0
    total_time = 0.0
    bug_types = {}
    multiple_method_bugs = 0

    for result in results:
        if "error" in result:
            error_count += 1
            continue

        success_count += 1
        confidence = result.get("confidence", 0.0)
        total_confidence += confidence
        total_time += result.get("processing_time", 0.0)

        # Count bug types
        bug_type = result.get("bug_type", "unknown")
        bug_types[bug_type] = bug_types.get(bug_type, 0) + 1

        # Count bugs involving multiple methods
        if result.get("involves_multiple_methods", False):
            multiple_method_bugs += 1

    # Calculate averages
    summary["successful_issues"] = success_count
    summary["errors"] = error_count
    summary["bug_types"] = bug_types
    summary["multiple_method_bugs"] = multiple_method_bugs

    if success_count > 0:
        summary["avg_confidence"] = total_confidence / success_count
        summary["avg_time"] = total_time / len(results)

    # Save summary
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Saved summary to {summary_file}")

    # Print summary to console
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total issues processed: {summary['total_issues']}")
    print(f"Successfully detected locations: {summary['successful_issues']}")
    print(f"Errors: {summary['errors']}")
    print(f"Average confidence: {summary['avg_confidence']:.3f}")
    print(f"Average processing time: {summary['avg_time']:.2f} seconds")
    print(f"Bugs involving multiple methods: {summary['multiple_method_bugs']}")

    print("\nBug Types:")
    for bug_type, count in bug_types.items():
        print(f"  {bug_type}: {count}")

    return summary


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Run the bug detector
    results = run_bug_detector(args)

    # Exit with success
    sys.exit(0)
