# src/utils/swe_bench_results.py

"""
SWE-Bench results analysis and reporting utilities.

This module provides functions for analyzing, comparing, and visualizing
the results of SWE-Bench tests, with a focus on self-reasoning performance.
"""

import os
import json
import glob
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import csv

from src.utils.logging import get_logger
from src.utils.file_utils import save_json, ensure_directory

logger = get_logger("swe_bench_results")


class SWEBenchResultsAnalyzer:
    """
    Analyzer for SWE-Bench test results.

    This class provides utilities for loading, analyzing, and comparing
    SWE-bench test results across different runs.
    """

    def __init__(self, results_dir: str = "results/swe_bench"):
        """
        Initialize the SWE-Bench results analyzer.

        Args:
            results_dir: Directory containing results
        """
        print(f"[DEBUG] Initializing SWEBenchResultsAnalyzer with results_dir='{results_dir}'")
        self.results_dir = results_dir
        self.cache = {}

    def load_instance_result(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Load results for a specific instance.

        Args:
            instance_id: ID of the instance

        Returns:
            Instance results or None if not found
        """
        print(f"[DEBUG] load_instance_result called with instance_id='{instance_id}'")
        # Check cache first
        if instance_id in self.cache:
            print("[DEBUG] Found in cache, returning cached result.")
            return self.cache[instance_id]

        # Look for result file
        result_path = os.path.join(self.results_dir, f"{instance_id}.json")
        print(f"[DEBUG] Looking for result file at '{result_path}'")
        if not os.path.exists(result_path):
            # Try to find in subdirectories
            pattern = os.path.join(self.results_dir, "**", f"{instance_id}.json")
            print(f"[DEBUG] result file not found, searching recursively with pattern '{pattern}'")
            matches = glob.glob(pattern, recursive=True)

            if not matches:
                print(f"[DEBUG] No matches found, returning None for instance '{instance_id}'")
                logger.warning(f"No results found for instance {instance_id}")
                return None

            # Use the most recent file if there are multiple matches
            result_path = max(matches, key=os.path.getmtime)
            print(f"[DEBUG] Found multiple matches, using most recent file: '{result_path}'")

        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)

            # Cache the result
            self.cache[instance_id] = result
            print(f"[DEBUG] Successfully loaded and cached result for '{instance_id}'")
            return result

        except Exception as e:
            logger.error(f"Error loading result for instance {instance_id}: {str(e)}")
            print(f"[DEBUG] Error loading result for '{instance_id}': {e}")
            return None

    def load_batch_results(self, batch_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load results for a batch of instances.

        Args:
            batch_file: Path to the batch results file (optional)

        Returns:
            Batch results or None if not found
        """
        print(f"[DEBUG] load_batch_results called with batch_file='{batch_file}'")
        if batch_file and os.path.exists(batch_file):
            print(f"[DEBUG] batch_file exists: '{batch_file}'")
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading batch results from {batch_file}: {str(e)}")
                print(f"[DEBUG] Error loading batch results from '{batch_file}': {e}")
                return None

        # Look for batch results files
        pattern = os.path.join(self.results_dir, "batch_results_*.json")
        print(f"[DEBUG] Looking for batch results using pattern='{pattern}'")
        matches = glob.glob(pattern)

        if not matches:
            logger.warning("No batch results found")
            print("[DEBUG] No batch results found, returning None.")
            return None

        # Use the most recent file
        latest_file = max(matches, key=os.path.getmtime)
        print(f"[DEBUG] Found batch files, using most recent: '{latest_file}'")

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading batch results from {latest_file}: {str(e)}")
            print(f"[DEBUG] Error loading batch results from '{latest_file}': {e}")
            return None

    def summarize_all_results(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize all results in the results directory.

        Args:
            output_file: Path to save the summary (optional)

        Returns:
            Summary statistics
        """
        print("[DEBUG] summarize_all_results called.")
        # Find all instance result files
        pattern = os.path.join(self.results_dir, "**", "*.json")
        all_files = glob.glob(pattern, recursive=True)
        print(f"[DEBUG] Found {len(all_files)} total JSON files under '{self.results_dir}' (including batch files).")

        # Filter out batch result files
        instance_files = [f for f in all_files if not os.path.basename(f).startswith("batch_results_")]
        print(f"[DEBUG] Filtered out batch files, leaving {len(instance_files)} instance result files.")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_instances": len(instance_files),
            "successful_instances": 0,
            "failed_instances": 0,
            "error_instances": 0,
            "success_rate": 0.0,
            "avg_iterations": 0.0,
            "instance_results": [],
            "categories": defaultdict(int),
            "iterations_distribution": defaultdict(int),
            "file_stats": {
                "files_changed": defaultdict(int),
                "components_affected": defaultdict(int)
            }
        }

        total_iterations = 0

        for file_path in instance_files:
            print(f"[DEBUG] Processing result file: '{file_path}'")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)

                instance_id = result.get("instance_id", os.path.basename(file_path).replace(".json", ""))
                success = result.get("success", False)
                status = result.get("status", "unknown")
                iterations = len(result.get("iterations", []))

                # Record basic info
                instance_summary = {
                    "instance_id": instance_id,
                    "success": success,
                    "status": status,
                    "iterations": iterations
                }

                summary["instance_results"].append(instance_summary)

                # Update counters
                if success:
                    summary["successful_instances"] += 1
                elif status == "error":
                    summary["error_instances"] += 1
                else:
                    summary["failed_instances"] += 1

                # Update iterations stats
                total_iterations += iterations
                summary["iterations_distribution"][iterations] += 1

                # Update category stats
                category = self._categorize_instance(instance_id)
                summary["categories"][category] += 1

                # Update file stats if available
                if "final_patch" in result and result["final_patch"]:
                    self._update_file_stats(summary["file_stats"], result["final_patch"])

            except Exception as e:
                logger.error(f"Error processing result file {file_path}: {str(e)}")
                print(f"[DEBUG] Error processing result file '{file_path}': {e}")
                summary["error_instances"] += 1

        # Calculate averages
        if summary["total_instances"] > 0:
            summary["success_rate"] = summary["successful_instances"] / summary["total_instances"]

        if total_iterations > 0:
            summary["avg_iterations"] = total_iterations / summary["total_instances"]

        # Convert defaultdicts to regular dicts for JSON serialization
        summary["categories"] = dict(summary["categories"])
        summary["iterations_distribution"] = dict(summary["iterations_distribution"])
        summary["file_stats"]["files_changed"] = dict(summary["file_stats"]["files_changed"])
        summary["file_stats"]["components_affected"] = dict(summary["file_stats"]["components_affected"])

        # Save summary if output file is specified
        if output_file:
            save_json(summary, output_file)
            logger.info(f"Summary saved to {output_file}")
            print(f"[DEBUG] Summary saved to '{output_file}'")

        print("[DEBUG] summarize_all_results completed.")
        return summary

    def compare_runs(self, run_dirs: List[str], output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare results across different runs.

        Args:
            run_dirs: List of directories containing run results
            output_file: Path to save the comparison (optional)

        Returns:
            Comparison results
        """
        print("[DEBUG] compare_runs called.")
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "runs": [],
            "common_instances": [],
            "only_in_run": {},
            "success_comparison": {},
            "iterations_comparison": {}
        }

        # Process each run
        all_instances = set()
        run_instances = {}

        for i, run_dir in enumerate(run_dirs):
            print(f"[DEBUG] Processing run {i+1} with directory '{run_dir}'")
            run_name = os.path.basename(run_dir)
            analyzer = SWEBenchResultsAnalyzer(run_dir)
            summary = analyzer.summarize_all_results()

            # Add run summary
            comparison["runs"].append({
                "name": run_name,
                "path": run_dir,
                "total_instances": summary["total_instances"],
                "successful_instances": summary["successful_instances"],
                "success_rate": summary["success_rate"]
            })

            # Extract instance IDs
            instance_ids = [result["instance_id"] for result in summary["instance_results"]]
            run_instances[run_name] = set(instance_ids)
            all_instances.update(instance_ids)

        # Find common instances
        if run_instances:
            common_instances = set.intersection(*run_instances.values())
        else:
            common_instances = set()
        comparison["common_instances"] = list(common_instances)
        print(f"[DEBUG] Found {len(common_instances)} common instances across all runs.")

        # Find instances only in specific runs
        for run_name, instances in run_instances.items():
            only_in_this_run = instances - set.union(
                *(other_instances for other_run, other_instances in run_instances.items() if other_run != run_name)
            )
            comparison["only_in_run"][run_name] = list(only_in_this_run)

        # Compare success and iterations for common instances
        for instance_id in common_instances:
            instance_comparison = {
                "success": {},
                "iterations": {}
            }

            for run_name, run_dir in zip([run["name"] for run in comparison["runs"]], run_dirs):
                analyzer = SWEBenchResultsAnalyzer(run_dir)
                result = analyzer.load_instance_result(instance_id)

                if result:
                    instance_comparison["success"][run_name] = result.get("success", False)
                    instance_comparison["iterations"][run_name] = len(result.get("iterations", []))

            comparison["success_comparison"][instance_id] = instance_comparison["success"]
            comparison["iterations_comparison"][instance_id] = instance_comparison["iterations"]

        # Save comparison if output file is specified
        if output_file:
            save_json(comparison, output_file)
            logger.info(f"Comparison saved to {output_file}")
            print(f"[DEBUG] Comparison results saved to '{output_file}'")

        print("[DEBUG] compare_runs completed.")
        return comparison

    def export_to_csv(self, output_file: str) -> None:
        """
        Export results to a CSV file.

        Args:
            output_file: Path to save the CSV file
        """
        print(f"[DEBUG] export_to_csv called with output_file='{output_file}'")
        # Find all instance result files
        pattern = os.path.join(self.results_dir, "**", "*.json")
        all_files = glob.glob(pattern, recursive=True)
        print(f"[DEBUG] Found {len(all_files)} total JSON files in '{self.results_dir}' for possible CSV export.")

        # Filter out batch result files
        instance_files = [f for f in all_files if not os.path.basename(f).startswith("batch_results_")]
        print(f"[DEBUG] After filtering out batch_results_* files, we have {len(instance_files)} instance files for CSV export.")

        # Define CSV headers
        headers = [
            "instance_id", "category", "success", "status", "iterations",
            "fail_to_pass_fixed", "fail_to_pass_total", "pass_to_pass_broken",
            "pass_to_pass_total", "files_changed", "patch_length", "duration"
        ]

        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                for file_path in instance_files:
                    print(f"[DEBUG] Writing row for '{file_path}'")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            result = json.load(f)

                        instance_id = result.get("instance_id", os.path.basename(file_path).replace(".json", ""))
                        category = self._categorize_instance(instance_id)

                        # Get test results from the last iteration if available
                        test_results = {}
                        iterations = result.get("iterations", [])
                        if iterations:
                            last_iteration = iterations[-1]
                            test_results = last_iteration.get("test_results", {})

                        # Create row
                        row = {
                            "instance_id": instance_id,
                            "category": category,
                            "success": result.get("success", False),
                            "status": result.get("status", "unknown"),
                            "iterations": len(iterations),
                            "fail_to_pass_fixed": test_results.get("fail_to_pass_fixed", 0),
                            "fail_to_pass_total": test_results.get("fail_to_pass_total", 0),
                            "pass_to_pass_broken": test_results.get("pass_to_pass_broken", 0),
                            "pass_to_pass_total": test_results.get("pass_to_pass_total", 0),
                            "files_changed": len(self._extract_files_from_patch(result.get("final_patch", ""))),
                            "patch_length": len(result.get("final_patch", "")),
                            "duration": result.get("duration", 0)
                        }

                        writer.writerow(row)

                    except Exception as e:
                        logger.error(f"Error processing result file {file_path} for CSV export: {str(e)}")
                        print(f"[DEBUG] Error processing '{file_path}' for CSV export: {e}")

            logger.info(f"Results exported to CSV: {output_file}")
            print(f"[DEBUG] CSV export completed successfully. Saved to '{output_file}'")

        except Exception as e:
            logger.error(f"Error exporting results to CSV: {str(e)}")
            print(f"[DEBUG] Error exporting results to CSV: {e}")

    def generate_report(self, output_dir: Optional[str] = None) -> str:
        """
        Generate a comprehensive report of the results.

        Args:
            output_dir: Directory to save the report (optional)

        Returns:
            Path to the generated report
        """
        print("[DEBUG] generate_report called.")
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "reports")

        ensure_directory(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"swe_bench_report_{timestamp}.md")

        # Generate summary
        summary = self.summarize_all_results()

        # Generate CSV export
        csv_file = os.path.join(output_dir, f"swe_bench_results_{timestamp}.csv")
        self.export_to_csv(csv_file)

        # Write report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# SWE-Bench Results Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall statistics
            f.write("## Overall Statistics\n\n")
            f.write(f"- Total instances: {summary['total_instances']}\n")
            f.write(f"- Successful instances: {summary['successful_instances']} ({summary['success_rate']:.2%})\n")
            f.write(f"- Failed instances: {summary['failed_instances']}\n")
            f.write(f"- Error instances: {summary['error_instances']}\n")
            f.write(f"- Average iterations: {summary['avg_iterations']:.2f}\n\n")

            # Categories
            f.write("## Categories\n\n")
            for category, count in sorted(summary["categories"].items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    f.write(f"- {category}: {count}\n")
            f.write("\n")

            # Iterations distribution
            f.write("## Iterations Distribution\n\n")
            for iterations, count in sorted(summary["iterations_distribution"].items()):
                if count > 0:
                    f.write(f"- {iterations} iterations: {count} instances\n")
            f.write("\n")

            # Instance details
            f.write("## Instance Details\n\n")
            f.write("| Instance ID | Success | Iterations | Category |\n")
            f.write("|------------|---------|------------|----------|\n")

            for instance in sorted(summary["instance_results"], key=lambda x: x["instance_id"]):
                instance_id = instance["instance_id"]
                category = self._categorize_instance(instance_id)
                f.write(
                    f"| {instance_id} | {'✅' if instance['success'] else '❌'} | {instance['iterations']} | {category} |\n")

        logger.info(f"Report generated: {report_file}")
        print(f"[DEBUG] Report generated and saved to '{report_file}'")
        return report_file

    def _categorize_instance(self, instance_id: str) -> str:
        """
        Categorize an instance based on its ID.

        Args:
            instance_id: ID of the instance

        Returns:
            Category name
        """
        if "__" in instance_id:
            # Format: repo__issue_number
            return instance_id.split("__")[0]

        # Default category
        return "unknown"

    def _update_file_stats(self, file_stats: Dict[str, Dict[str, int]], patch_content: str) -> None:
        """
        Update file statistics based on a patch.

        Args:
            file_stats: File statistics to update
            patch_content: Content of the patch
        """
        print("[DEBUG] _update_file_stats called to analyze patch_content.")
        # Extract files changed
        files_changed = self._extract_files_from_patch(patch_content)
        print(f"[DEBUG] Found {len(files_changed)} files changed in patch.")

        # Update counts
        for file_path in files_changed:
            file_stats["files_changed"][file_path] += 1
            print(f"[DEBUG] Incremented files_changed count for '{file_path}'. Current count={file_stats['files_changed'][file_path]}")

            # Extract component from file path
            if "/" in file_path:
                component = file_path.split("/")[0]
                file_stats["components_affected"][component] += 1
                print(f"[DEBUG] Incremented components_affected count for '{component}'. Current count={file_stats['components_affected'][component]}")

    def _extract_files_from_patch(self, patch_content: str) -> List[str]:
        """
        Extract files changed from a patch.

        Args:
            patch_content: Content of the patch

        Returns:
            List of file paths
        """
        print("[DEBUG] _extract_files_from_patch called.")
        files = []

        if not patch_content:
            print("[DEBUG] patch_content is empty, returning [].")
            return files

        # Look for file headers
        file_patterns = [
            r"diff --git a/([^\s]+) b/",
            r"\+\+\+ b/([^\s]+)",
            r"--- a/([^\s]+)"
        ]

        for pattern in file_patterns:
            matches = re.findall(pattern, patch_content)
            files.extend(matches)

        # Remove duplicates while preserving order
        unique_files = []
        for file_path in files:
            if file_path not in unique_files:
                unique_files.append(file_path)

        print(f"[DEBUG] Extracted file paths: {unique_files}")
        return unique_files
