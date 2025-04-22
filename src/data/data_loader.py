# src/data/data_loader.py 

import logging
import os
import re
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SWEBenchDataLoader:
    """
    Loader for SWE-bench-Verified dataset.
    """

    def __init__(self, config):
        """
        Initialize SWE-bench data loader.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.data_path = Path(config["data"]["swe_bench_path"])
        self.cache_dir = Path(config["data"]["cache_dir"])
        self.max_context_length = config["data"]["max_context_length"]

        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the SWE-bench-Verified dataset.

        Returns:
            List of dictionaries containing issue information.
        """
        # Try to find the dataset file
        # First check if file_path is specified in config
        if "file_path" in self.config["data"]:
            file_path = Path(self.config["data"]["file_path"])
            if file_path.exists():
                logger.info(f"Loading dataset from specified file_path: {file_path}")
                return self._load_json_dataset(file_path)

        # Then check the swe_bench_path
        dataset_path = self.data_path

        # Try different file extensions and formats
        possible_paths = [
            dataset_path / "swe_bench_test.json",
        ]

        for path in possible_paths:
            if path.exists() and path.is_file():
                logger.info(f"Loading dataset from: {path}")
                if path.suffix == '.jsonl':
                    return self._load_jsonl_dataset(path)
                else:
                    return self._load_json_dataset(path)

        # If we get here, we couldn't find the dataset
        raise FileNotFoundError(
            f"Dataset file not found. Tried: {[str(p) for p in possible_paths]}. "
            f"Please run the download script: python -m src.scripts.download_swe_bench"
        )

    def _load_jsonl_dataset(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load dataset from a JSONL file."""
        issues = []
        with open(file_path, 'r') as f:
            for line in f:
                issue_data = json.loads(line)
                issues.append(issue_data)

        logger.info(f"Loaded {len(issues)} issues from JSONL dataset")
        return issues

    def _load_json_dataset(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load dataset from a JSON file."""
        with open(file_path, 'r') as f:
            issues = json.load(f)

        logger.info(f"Loaded {len(issues)} issues from JSON dataset")
        return issues

    def load_issue(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific issue by ID.

        Args:
            issue_id: ID of the issue to load.

        Returns:
            Dictionary containing issue information.
        """
        issues = self.load_dataset()
        for issue in issues:
            # Check both instance_id and id fields to be compatible with different dataset formats
            if issue.get("instance_id") == issue_id or issue.get("id") == issue_id:
                return issue

        # If not found, try more advanced techniques
        issue_id_parts = issue_id.split("__")
        if len(issue_id_parts) > 1:
            # Try to match partial issue ids
            repo_name = issue_id_parts[0]
            issue_number = issue_id_parts[1]

            for issue in issues:
                if (issue.get("repo", "") == repo_name and
                        (str(issue.get("issue_number", "")) == issue_number or
                         str(issue.get("number", "")) == issue_number)):
                    return issue

        logger.warning(f"Issue {issue_id} not found in dataset")
        return None

    def get_issue_description(self, issue: Dict[str, Any]) -> str:
        """
        Extract the description from an issue with improved robustness.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the issue description.
        """
        # First check for problem_statement field (from SWE-bench dataset)
        if "problem_statement" in issue and issue["problem_statement"]:
            logger.debug(f"Using problem_statement field ({len(issue['problem_statement'])} chars)")
            return issue["problem_statement"]

        # Check for issue_description field that might be present in some formats
        if "issue_description" in issue and issue["issue_description"]:
            logger.debug(f"Using issue_description field ({len(issue['issue_description'])} chars)")
            return issue["issue_description"]

        # Fall back to traditional fields if problem_statement not found
        description = issue.get("description", "")
        title = issue.get("title", "")

        if title or description:
            combined = f"Title: {title}\n\nDescription:\n{description}"
            logger.debug(f"Using title and description fields ({len(combined)} chars)")
            return combined

        # Try to extract from raw issue
        raw_issue = issue.get("raw_issue", "")
        if raw_issue:
            logger.debug(f"Using raw_issue field ({len(raw_issue)} chars)")
            return f"Raw Issue:\n{raw_issue}"

        # Last resort: Look for any text fields
        for key in ["body", "prompt", "details", "test_description"]:
            if key in issue and issue[key]:
                logger.debug(f"Using {key} field ({len(issue[key])} chars)")
                return f"{key}:\n{issue[key]}"

        # Absolute fallback: use ID information to create a minimal description
        repo = issue.get("repo", "unknown")
        issue_id = issue.get("id", issue.get("instance_id", "unknown"))
        logger.warning(f"No description found for issue {issue_id}. Using minimal description.")

        # Create a minimal description with whatever information we have
        fallback = f"Fix issue in repository {repo}, issue ID: {issue_id}. "
        fallback += "Examine the codebase to identify and fix any bugs or inconsistencies in the implementation. "

        # Try to add whatever additional info we can find
        if "files_modified" in issue:
            files = issue.get("files_modified", [])
            fallback += f"The following files may need modification: {', '.join(files[:3])}"

            # If we have file names, try to guess what we're fixing based on them
            for file in files:
                if "test" in file.lower():
                    fallback += " The issue may be related to test failures."
                    break

        return fallback

    def get_codebase_context(self, issue: Dict[str, Any]) -> str:
        """
        Get context from the codebase related to an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the codebase context.
        """
        # Extract repo and file paths
        repo = issue.get("repo", "")
        repo_path = self.data_path / "repos" / repo

        # Get file paths from the issue
        file_paths = []
        if "files_modified" in issue:
            file_paths.extend(issue["files_modified"])
        if "files_created" in issue:
            file_paths.extend(issue["files_created"])
        if "files_deleted" in issue:
            file_paths.extend(issue["files_deleted"])

        # Read file contents
        context = ""
        for file_path in file_paths:
            try:
                full_path = repo_path / file_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    context += f"File: {file_path}\n\n{content}\n\n"
                else:
                    logger.warning(f"File not found: {full_path}")
                    context += f"File: {file_path} (not found)\n\n"
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                context += f"File: {file_path} (error: {e})\n\n"

        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "...[TRUNCATED]"

        return context

    def get_solution_patch(self, issue: Dict[str, Any]) -> str:
        """
        Get the solution patch for an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing the solution patch.
        """
        # First try to get the patch from the standard field
        if "patch" in issue and issue["patch"]:
            return issue["patch"]

        # Try alternative fields if patch is not present
        if "solution" in issue and issue["solution"]:
            return issue["solution"]

        # For some dataset formats, it might be in gold_patch
        if "gold_patch" in issue and issue["gold_patch"]:
            return issue["gold_patch"]

        logger.warning(f"No solution patch found for issue {issue.get('issue_id', 'unknown')}")
        return ""

    def get_hints(self, issue: Dict[str, Any]) -> Optional[str]:
        """
        Extract hints from an issue.

        Args:
            issue: Issue dictionary.

        Returns:
            String containing hints if available, None otherwise.
        """
        # Handle SWE-bench dataset format which might include hints_text
        if "hints_text" in issue:
            return issue.get("hints_text", "")

        # Handle hints that might be in comments field
        if "comments" in issue:
            return "Comments from the issue:\n" + issue.get("comments", "")

        return None

    def get_test_patch(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and analyze test patch from an issue to provide richer context.

        Args:
            issue: Issue dictionary.

        Returns:
            Dictionary containing test patch information, or empty dict if not available.
        """
        # Initialize return structure
        test_info = {
            "patch": None,
            "files": [],
            "implementation_files": [],
            "test_functions": [],
            "imports": []
        }

        # Extract test patch from the issue
        if "test_patch" in issue:
            test_patch = issue.get("test_patch", "")
            if not test_patch:
                return test_info

            test_info["patch"] = test_patch

            # Extract file paths from test patch
            file_pattern = r'(?:---|\+\+\+) [ab]/([^\n]+)'
            patch_files = re.findall(file_pattern, test_patch)
            test_info["files"] = list(set(patch_files))

            # Infer implementation files being tested
            for file_path in patch_files:
                if 'test' in file_path.lower():
                    # Try to infer the implementation file being tested
                    # Common patterns: test_foo.py tests foo.py
                    test_file_name = os.path.basename(file_path)
                    if test_file_name.startswith('test_'):
                        impl_name = test_file_name[5:]  # Remove 'test_'
                        impl_dir = os.path.dirname(file_path)

                        # Try multiple directory structures
                        possible_impl_dirs = [
                            impl_dir,  # Same directory
                            impl_dir.replace('/tests', ''),  # Parent module
                            re.sub(r'/tests.*', '', impl_dir)  # Root module
                        ]

                        for impl_dir in possible_impl_dirs:
                            impl_path = os.path.join(impl_dir, impl_name)
                            if impl_path not in test_info["implementation_files"]:
                                test_info["implementation_files"].append(impl_path)

            # Extract added/modified test functions
            test_func_pattern = r'\+\s*(def test_[^:]+:.*?)(?=\n[^+]|\Z)'
            test_funcs = re.findall(test_func_pattern, test_patch, re.DOTALL)
            test_info["test_functions"] = [func.strip() for func in test_funcs]

            # Extract imports for additional context
            import_pattern = r'\+\s*((?:from|import)\s+[^\n]+)'
            imports = re.findall(import_pattern, test_patch)
            test_info["imports"] = imports

            # Extract assertions to understand what's being tested
            assert_pattern = r'\+\s*(assert[^;]+)'
            assertions = re.findall(assert_pattern, test_patch)
            test_info["assertions"] = assertions

            # Log what we found
            logger.debug(f"Test patch analysis: {len(test_info['files'])} files, "
                         f"{len(test_info['implementation_files'])} impl files, "
                         f"{len(test_info['test_functions'])} test functions")

        return test_info

    def get_fail_to_pass_tests(self, issue: Dict[str, Any]) -> List[str]:
        """
        Extract tests that should go from failing to passing.

        Args:
            issue: Issue dictionary.

        Returns:
            List of test identifiers.
        """
        # Handle SWE-bench dataset format which includes FAIL_TO_PASS
        if "FAIL_TO_PASS" in issue:
            fail_to_pass = issue.get("FAIL_TO_PASS", "[]")
            # It might be a JSON string or already parsed
            if isinstance(fail_to_pass, str):
                try:
                    return json.loads(fail_to_pass)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse FAIL_TO_PASS as JSON: {fail_to_pass}")
                    return []
            elif isinstance(fail_to_pass, list):
                return fail_to_pass

        return []

    def get_pass_to_pass_tests(self, issue: Dict[str, Any]) -> List[str]:
        """
        Extract tests that should continue to pass.

        Args:
            issue: Issue dictionary.

        Returns:
            List of test identifiers.
        """
        # Handle SWE-bench dataset format which includes PASS_TO_PASS
        if "PASS_TO_PASS" in issue:
            pass_to_pass = issue.get("PASS_TO_PASS", "[]")
            # It might be a JSON string or already parsed
            if isinstance(pass_to_pass, str):
                try:
                    return json.loads(pass_to_pass)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse PASS_TO_PASS as JSON: {pass_to_pass}")
                    return []
            elif isinstance(pass_to_pass, list):
                return pass_to_pass

        return []

    def prepare_repository_for_analysis(self, issue):
        """
        Prepare repository for code analysis by checking out the base commit.

        Args:
            issue: Issue dictionary containing metadata

        Returns:
            Boolean indicating success
        """
        repo = issue.get("repo", "")
        base_commit = issue.get("base_commit", "")

        if not base_commit:
            logger.warning(f"No base_commit specified for issue {issue.get('instance_id', '')}")
            return False

        repo_path = Path(self.config["data"]["repositories"]) / repo

        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {repo_path}")
            return False

        # Checkout the base commit for analysis
        try:
            import subprocess
            subprocess.run(
                ["git", "checkout", base_commit, "-f"],
                cwd=repo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Repository {repo} checked out to base_commit {base_commit}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout base_commit {base_commit}: {e}")
            return False

    def prepare_repository_for_testing(self, issue):
        """
        Prepare repository for testing by setting up the environment at the right commit.

        Args:
            issue: Issue dictionary containing metadata

        Returns:
            Boolean indicating success
        """
        repo = issue.get("repo", "")
        env_commit = issue.get("environment_setup_commit", "")

        # If environment_setup_commit is not specified, fall back to base_commit
        if not env_commit:
            env_commit = issue.get("base_commit", "")
            logger.info(f"No environment_setup_commit specified, using base_commit")

        if not env_commit:
            logger.warning(f"No environment commit specified for issue {issue.get('instance_id', '')}")
            return False

        repo_path = Path(self.config["data"]["repositories"]) / repo

        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {repo_path}")
            return False

        # Checkout the environment setup commit
        try:
            import subprocess
            subprocess.run(
                ["git", "checkout", env_commit, "-f"],
                cwd=repo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Run any additional environment setup steps if needed
            self._setup_environment(repo_path, issue)

            logger.info(f"Repository {repo} environment prepared at commit {env_commit}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup environment at commit {env_commit}: {e}")
            return False

    def _setup_environment(self, repo_path, issue):
        """
        Perform additional environment setup steps based on issue metadata.
        Creates and manages a clean environment for each issue.

        Args:
            repo_path: Path to the repository
            issue: Issue dictionary containing metadata
        """
        # Get unique identifier for this issue
        issue_id = issue.get("instance_id", issue.get("id", "unknown"))

        # Create a directory for virtual environments if it doesn't exist
        venv_dir = self.cache_dir / "venvs"
        if not venv_dir.exists():
            venv_dir.mkdir(parents=True)

        # Create a unique venv name for this issue
        venv_name = f"issue_{issue_id.replace('/', '_').replace('-', '_')}"
        venv_path = venv_dir / venv_name

        # Check if we should create a new environment
        create_new_env = not venv_path.exists()

        try:
            import subprocess
            import sys
            import os
            from shutil import rmtree

            # If environment exists but we want to recreate it
            if venv_path.exists() and create_new_env:
                logger.info(f"Removing existing environment for issue {issue_id}")
                rmtree(venv_path)

            # Create a new virtual environment if needed
            if create_new_env:
                logger.info(f"Creating new environment for issue {issue_id}")

                # Create virtual environment
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Determine pip path inside the venv
                if os.name == 'nt':  # Windows
                    pip_path = venv_path / "Scripts" / "pip"
                else:  # Unix/Linux/Mac
                    pip_path = venv_path / "bin" / "pip"

                # Upgrade pip in the new environment
                subprocess.run(
                    [str(pip_path), "install", "--upgrade", "pip"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Check for setup.py and pip-requirements/requirements.txt in the repo
                setup_file = repo_path / "setup.py"
                req_file = repo_path / "requirements.txt"
                pip_req_file = repo_path / "pip-requirements"

                # Check if we need to prioritize setup.py installation
                use_setup_py = setup_file.exists()

                # Check if pip-requirements file indicates to use setup.py instead
                if pip_req_file.exists():
                    with open(pip_req_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if "pip install" in content and "." in content:
                            use_setup_py = True
                            logger.info(f"pip-requirements file indicates to use setup.py")

                # Install the package using setup.py if appropriate
                if use_setup_py:
                    try:
                        logger.info(f"Installing package via setup.py for issue {issue_id}")
                        # First try with all optional dependencies
                        try:
                            subprocess.run(
                                [str(pip_path), "install", "-e", f"{str(repo_path)}[all]"],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=600  # 10 minutes timeout
                            )
                            logger.info(f"Successfully installed package with all extras")
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                            # If that fails, try basic install
                            logger.info(f"Falling back to basic package install")
                            subprocess.run(
                                [str(pip_path), "install", "-e", str(repo_path)],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=600  # 10 minutes timeout
                            )
                            logger.info(f"Successfully installed package with basic dependencies")
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                        logger.warning(f"Failed to install package: {e}")

                # If setup.py approach didn't work or isn't appropriate, try requirements.txt
                elif req_file.exists():
                    try:
                        logger.info(f"Installing dependencies from requirements.txt for issue {issue_id}")
                        subprocess.run(
                            [str(pip_path), "install", "-r", str(req_file)],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=600  # 10 minutes timeout
                        )
                        logger.info(f"Successfully installed dependencies from requirements.txt")
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                        logger.warning(f"Failed to install dependencies from requirements.txt: {e}")

                else:
                    logger.info(f"No setup.py or requirements.txt found for issue {issue_id}")

            # Store environment path in issue for later use
            issue["environment_path"] = str(venv_path)

            return True
        except Exception as e:
            logger.error(f"Error setting up environment for issue {issue_id}: {e}")
            return False

    def cleanup_environments(self, max_age_days=7):
        """
        Clean up old virtual environments.

        Args:
            max_age_days: Maximum age of environments to keep in days
        """
        import time
        from shutil import rmtree

        venv_dir = self.cache_dir / "venvs"
        if not venv_dir.exists():
            return

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60

        for venv_path in venv_dir.iterdir():
            if not venv_path.is_dir():
                continue

            try:
                # Get last modified time
                mtime = venv_path.stat().st_mtime
                age = current_time - mtime

                if age > max_age_seconds:
                    logger.info(f"Removing old environment: {venv_path.name}")
                    rmtree(venv_path)
            except Exception as e:
                logger.warning(f"Error cleaning up environment {venv_path}: {e}")
