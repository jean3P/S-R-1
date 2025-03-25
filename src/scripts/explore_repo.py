# src/scripts/explore_repo.py

# !/usr/bin/env python3
"""Tool to test repository exploration for rule files."""

import os
import sys
import argparse
from src.utils.repo_explorer import RepoExplorer
import json


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore repository for rule definitions")
    parser.add_argument("--repo", type=str, required=True, help="Path to the repository")
    parser.add_argument("--rule", type=str, required=True, help="Rule ID to find (e.g., L031)")
    parser.add_argument("--output", type=str, help="Path to save results as JSON")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    if not os.path.exists(args.repo):
        print(f"Repository path does not exist: {args.repo}")
        sys.exit(1)

    explorer = RepoExplorer(args.repo)
    result = explorer.find_rule_definition(args.rule)

    if not result:
        print(f"Could not find rule definition for {args.rule}")
        sys.exit(1)

    print(f"Found rule {args.rule} in file: {result['file_path']}")

    if result["line_number"]:
        print(f"Rule description at line: {result['line_number']}")

    if result["code_snippet"]:
        print("\nCode snippet:")
        print("-----------------------------")
        print(result["code_snippet"])
        print("-----------------------------")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
