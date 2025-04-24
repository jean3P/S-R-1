#!/usr/bin/env python
"""
Run the Context-Based Bug Detector on SWE-bench issues.

This script executes the context-based bug detector on specified SWE-bench issues
and outputs the detected bug locations for use in the bug fixing pipeline.
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

from src.utils.static_bug_detector import StaticAnalysisBugDetector

# Add the parent directory to sys.path if not already there
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config.config import Config
from src.data.data_loader import SWEBenchDataLoader

from src.utils.file_utils import FileUtils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Context-Based Bug Detector")

    parser.add_argument("--config", type=str, default="configs/experiments/bug_detector.yaml",
                        help="Path to configuration file")
    parser.add_argument("--issues", type=str, help="Comma-separated list of issue IDs")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of issues to process")
    parser.add_argument("--output", type=str, default="results/bug_detector",
                        help="Output directory for results")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    parser.add_argument("--validate", action="store_true",
                        help="Validate results against ground truth (if available)")
    parser.add_argument("--memory-efficient", action="store_true",
                        help="Use memory-efficient processing")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of results")

    return parser.parse_args()


def setup_logging(log_level, output_dir):
    """Set up logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"bug_detector_{time.strftime('%Y%m%d_%H%M%S')}.log"

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


def update_config_from_args(config, args):
    """Update configuration based on command line arguments."""
    # Update output directory
    if args.output:
        config["evaluation"]["results_dir"] = args.output

    # Set memory efficiency flag
    if args.memory_efficient:
        config["memory_efficient"] = True

    return config


def run_bug_detector(args):
    """Run the context-based bug detector on specified issues."""
    # Load configuration
    config_path = args.config
    config = Config(config_path)

    # Set up logging
    log_file = setup_logging(args.log_level, args.output)

    # Update config from args
    config = update_config_from_args(config, args)

    # Create results directory
    results_dir = Path(config["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data loader
    data_loader = SWEBenchDataLoader(config)

    # Initialize the bug detector
    bug_detector = StaticAnalysisBugDetector(config)

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
            issue_id = issue.get("instance_id") or issue.get("id")
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

            # Add result
            results.append(bug_location)

            # Save individual result
            result_file = results_dir / f"bug_location_{issue_id}.json"
            FileUtils.write_json(bug_location, result_file)
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
    FileUtils.write_json(results, combined_file)
    logging.info(f"Saved combined results to {combined_file}")

    # Generate summary
    generate_summary(results, results_dir, args.visualize)

    # Validate results if requested
    if args.validate:
        validate_results(results, data_loader, results_dir)

    return results


def generate_summary(results, results_dir, visualize=False):
    """Generate a summary of the results."""
    summary = {
        "total_issues": len(results),
        "successful_issues": 0,
        "errors": 0,
        "avg_confidence": 0.0,
        "avg_time": 0.0,
        "confidence_distribution": {
            "high": 0,  # > 0.8
            "medium": 0,  # 0.5-0.8
            "low": 0  # < 0.5
        }
    }

    # Calculate statistics
    success_count = 0
    error_count = 0
    total_confidence = 0.0
    total_time = 0.0

    for result in results:
        if "error" in result:
            error_count += 1
            continue

        success_count += 1
        confidence = result.get("confidence", 0.0)
        total_confidence += confidence
        total_time += result.get("processing_time", 0.0)

        # Count confidence distribution
        if confidence > 0.8:
            summary["confidence_distribution"]["high"] += 1
        elif confidence >= 0.5:
            summary["confidence_distribution"]["medium"] += 1
        else:
            summary["confidence_distribution"]["low"] += 1

    # Calculate averages
    summary["successful_issues"] = success_count
    summary["errors"] = error_count

    if success_count > 0:
        summary["avg_confidence"] = total_confidence / success_count
        summary["avg_time"] = total_time / len(results)

    # Save summary
    summary_file = results_dir / "summary.json"
    FileUtils.write_json(summary, summary_file)
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
    print("\nConfidence distribution:")
    print(f"  High (>0.8): {summary['confidence_distribution']['high']}")
    print(f"  Medium (0.5-0.8): {summary['confidence_distribution']['medium']}")
    print(f"  Low (<0.5): {summary['confidence_distribution']['low']}")

    # Generate visualizations if requested
    if visualize:
        try:
            generate_visualizations(results, results_dir)
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}", exc_info=True)

    return summary


def generate_visualizations(results, results_dir):
    """Generate visualizations of the results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
    except ImportError:
        logging.error("Visualization libraries not available. Install matplotlib, seaborn, and pandas.")
        return

    # Ensure visualization directory exists
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Extract data for plotting
    data = []
    for result in results:
        if "error" in result:
            continue

        data.append({
            "issue_id": result.get("issue_id", "Unknown"),
            "confidence": result.get("confidence", 0.0),
            "time": result.get("processing_time", 0.0),
            "file": os.path.basename(result.get("file", "Unknown"))
        })

    if not data:
        logging.warning("No valid data for visualization")
        return

    # Create DataFrame
    df = pd.DataFrame(data)

    # Set plot style
    sns.set(style="whitegrid")

    # Plot 1: Confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["confidence"], bins=10, kde=True)
    plt.title("Distribution of Bug Detection Confidence")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(viz_dir / "confidence_distribution.png")
    plt.close()

    # Plot 2: Confidence by issue
    plt.figure(figsize=(12, 7))
    sns.barplot(x="issue_id", y="confidence", data=df)
    plt.title("Bug Detection Confidence by Issue")
    plt.xlabel("Issue ID")
    plt.ylabel("Confidence Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(viz_dir / "confidence_by_issue.png")
    plt.close()

    # Plot 3: Processing time distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["time"], bins=10, kde=True)
    plt.title("Distribution of Processing Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(viz_dir / "time_distribution.png")
    plt.close()

    # Plot 4: Time vs Confidence scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="time", y="confidence", data=df)
    plt.title("Processing Time vs Confidence")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Confidence Score")
    plt.tight_layout()
    plt.savefig(viz_dir / "time_vs_confidence.png")
    plt.close()

    logging.info(f"Generated visualizations in {viz_dir}")


def validate_results(results, data_loader, results_dir):
    """Validate bug detection results against ground truth."""
    validation_results = {
        "total": len(results),
        "successful": 0,
        "file_match": 0,
        "function_match": 0,
        "both_match": 0,
        "details": []
    }

    for result in results:
        issue_id = result.get("issue_id")
        if "error" in result:
            validation_results["details"].append({
                "issue_id": issue_id,
                "error": result["error"],
                "status": "error"
            })
            continue

        # Get issue and ground truth solution
        issue = data_loader.load_issue(issue_id)
        if not issue:
            continue

        # Get ground truth solution patch
        patch = data_loader.get_solution_patch(issue)

        # Extract modified files from patch
        modified_files = extract_modified_files_from_patch(patch)

        # Check if detected file is in modified files
        detected_file = result.get("file", "")
        file_match = any(detected_file in mod_file for mod_file in modified_files)

        # Extract modified functions from patch (simplified)
        modified_functions = extract_modified_functions_from_patch(patch)

        # Check if detected function is in modified functions
        detected_function = result.get("function", "")
        function_match = detected_function in modified_functions

        # Update counts
        if file_match:
            validation_results["file_match"] += 1

        if function_match:
            validation_results["function_match"] += 1

        if file_match and function_match:
            validation_results["both_match"] += 1

        if file_match or function_match:
            validation_results["successful"] += 1

        # Add details
        validation_results["details"].append({
            "issue_id": issue_id,
            "detected_file": detected_file,
            "ground_truth_files": modified_files,
            "file_match": file_match,
            "detected_function": detected_function,
            "ground_truth_functions": modified_functions,
            "function_match": function_match,
            "confidence": result.get("confidence", 0.0),
            "status": "match" if (file_match or function_match) else "no_match"
        })

    # Calculate percentages
    total = validation_results["total"]
    if total > 0:
        validation_results["file_match_percent"] = (validation_results["file_match"] / total) * 100
        validation_results["function_match_percent"] = (validation_results["function_match"] / total) * 100
        validation_results["both_match_percent"] = (validation_results["both_match"] / total) * 100
        validation_results["success_percent"] = (validation_results["successful"] / total) * 100

    # Save validation results
    validation_file = results_dir / "validation_results.json"
    FileUtils.write_json(validation_results, validation_file)
    logging.info(f"Saved validation results to {validation_file}")

    # Print validation summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total issues: {validation_results['total']}")
    print(
        f"Successful detections: {validation_results['successful']} ({validation_results.get('success_percent', 0):.1f}%)")
    print(f"File matches: {validation_results['file_match']} ({validation_results.get('file_match_percent', 0):.1f}%)")
    print(
        f"Function matches: {validation_results['function_match']} ({validation_results.get('function_match_percent', 0):.1f}%)")
    print(
        f"Both file and function matches: {validation_results['both_match']} ({validation_results.get('both_match_percent', 0):.1f}%)")


def extract_modified_files_from_patch(patch):
    """Extract modified files from a patch."""
    if not patch:
        return []

    files = []
    # Look for file paths in diff headers
    file_pattern = r'(?:---|\+\+\+) [ab]/([^\n]+)'
    matches = re.findall(file_pattern, patch)
    files.extend(matches)

    return list(set(files))


def extract_modified_functions_from_patch(patch):
    """Extract modified functions from a patch (simplified)."""
    if not patch:
        return []

    functions = []
    # Look for function definitions in added/modified lines
    func_pattern = r'[\+\-]\s*def\s+(\w+)'
    matches = re.findall(func_pattern, patch)
    functions.extend(matches)

    # Also look for method calls that might indicate changes
    method_pattern = r'[\+\-]\s*\w+\.(\w+)\('
    matches = re.findall(method_pattern, patch)
    functions.extend(matches)

    return list(set(functions))


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Run the bug detector
    results = run_bug_detector(args)

    # Exit with success
    sys.exit(0)
