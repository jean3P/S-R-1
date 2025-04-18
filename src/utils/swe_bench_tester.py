# src/utils/swe_bench_tester.py

"""
SWE-Bench testing utilities for self-reasoning agents.

This module provides tools for testing SWE-Bench dataset rows, including
repository management, patch application, test execution, and result tracking.
"""

import os
import subprocess
import tempfile
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.utils.file_utils import save_json, ensure_directory
from src.utils.logging import get_logger

logger = get_logger("swe_bench_tester")


class SWEBenchTester:
    def __init__(self,
                 repos_dir: str = "data/repositories",
                 results_dir: str = "results/swe_bench",
                 timeout: int = 300,
                 max_iterations: int = 3,
                 use_cache: bool = True):
        self.repos_dir = repos_dir
        self.results_dir = results_dir
        self.timeout = timeout
        self.max_iterations = max_iterations
        self.use_cache = use_cache

        ensure_directory(repos_dir)
        ensure_directory(results_dir)
        self.repo_cache = {}

    def test_instance(self, instance: Dict[str, Any], agent: Any = None) -> Dict[str, Any]:
        instance_id = instance.get("instance_id")
        repo_name = instance.get("repo")
        base_commit = instance.get("base_commit")
        patch = instance.get("patch", "")
        problem_statement = instance.get("problem_statement", "")

        fail_to_pass_tests = json.loads(instance.get("FAIL_TO_PASS", "[]"))
        pass_to_pass_tests = json.loads(instance.get("PASS_TO_PASS", "[]"))

        results = {
            "instance_id": instance_id,
            "repo": repo_name,
            "base_commit": base_commit,
            "timestamp": datetime.now().isoformat(),
            "status": "started",
            "iterations": [],
            "final_patch": None,
            "success": False,
            "error": None,
            "duration": 0
        }

        start_time = time.time()

        try:
            repo_path = self._setup_repository(repo_name, base_commit)
            if not repo_path:
                raise Exception(f"Could not set up repository {repo_name}")

            patch_to_apply = patch
            if agent:
                task = {
                    "name": instance_id,
                    "language": "python",
                    "initial_prompt": problem_statement,
                    "repo_info": {
                        "repo": repo_name,
                        "base_commit": base_commit,
                        "environment_setup_commit": instance.get("environment_setup_commit")
                    },
                    "test_info": {
                        "fail_to_pass": fail_to_pass_tests,
                        "pass_to_pass": pass_to_pass_tests
                    }
                }

                for iteration in range(1, self.max_iterations + 1):
                    if iteration == 1:
                        patch_to_apply = self._generate_patch(agent, problem_statement, task)
                    else:
                        prev = results["iterations"][-1]
                        patch_to_apply = self._refine_patch(agent, problem_statement, prev["patch"], prev["test_results"].get("output", ""), prev["test_results"].get("errors", ""), task)

                    success, message = self._apply_patch(repo_path, patch_to_apply)
                    test_results = {"success": success, "output": message, "errors": ""}

                    results["iterations"].append({
                        "iteration": iteration,
                        "patch": patch_to_apply,
                        "patch_applied": success,
                        "apply_message": message,
                        "test_results": test_results,
                        "success": success
                    })

                    if success:
                        results["success"] = True
                        results["final_patch"] = patch_to_apply
                        break
                    else:
                        self._reset_repository(repo_path, base_commit)
            else:
                success, message = self._apply_patch(repo_path, patch_to_apply)
                test_results = {"success": success, "output": message, "errors": ""}

                results["iterations"].append({
                    "iteration": 1,
                    "patch": patch_to_apply,
                    "patch_applied": success,
                    "apply_message": message,
                    "test_results": test_results,
                    "success": success
                })

                results["success"] = success
                results["final_patch"] = patch_to_apply

            results["status"] = "completed"
            results["duration"] = time.time() - start_time

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            results["duration"] = time.time() - start_time

        results_path = os.path.join(self.results_dir, f"{instance_id}.json")
        save_json(results, results_path)
        return results

    def _setup_repository(self, repo_name: str, commit_hash: str) -> Optional[str]:
        safe_repo_name = repo_name.replace("/", "_")
        repo_path = os.path.join(self.repos_dir, safe_repo_name)

        if self.use_cache and safe_repo_name in self.repo_cache:
            if self._reset_repository(repo_path, commit_hash):
                return self.repo_cache[safe_repo_name]

        if os.path.exists(repo_path):
            if self._reset_repository(repo_path, commit_hash):
                self.repo_cache[safe_repo_name] = repo_path
                return repo_path
            else:
                import shutil
                shutil.rmtree(repo_path)

        try:
            subprocess.run(["git", "clone", f"https://github.com/{repo_name}.git", repo_path], check=True)
            if self._reset_repository(repo_path, commit_hash):
                self.repo_cache[safe_repo_name] = repo_path
                return repo_path
        except Exception as e:
            logger.error(f"Failed to clone repo {repo_name}: {e}")
        return None

    def _reset_repository(self, repo_path: str, commit_hash: str) -> bool:
        try:
            subprocess.run(["git", "-C", repo_path, "clean", "-fdx"], check=True)
            subprocess.run(["git", "-C", repo_path, "reset", "--hard", commit_hash], check=True)
            return True
        except subprocess.SubprocessError:
            return False

    def _apply_patch(self, repo_path: str, patch_content: str) -> Tuple[bool, str]:
        if not patch_content.strip():
            return False, "Empty patch content"

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as patch_file:
                patch_file.write(patch_content)
                patch_path = patch_file.name

            result = subprocess.run(["git", "-C", repo_path, "apply", patch_path], capture_output=True, text=True)
            os.unlink(patch_path)

            if result.returncode == 0:
                return True, "Patch applied successfully"
            else:
                return False, result.stderr.strip()
        except Exception as e:
            return False, str(e)

    def _generate_patch(self, agent: Any, problem: str, task: Dict[str, Any]) -> str:
        try:
            result = agent.reflect(problem, task)
            return result.get("best_solution") if isinstance(result, dict) else str(result)
        except Exception as e:
            return f"# ERROR: {str(e)}"

    def _refine_patch(self, agent: Any, problem: str, prev_patch: str, output: str, errors: str, task: Dict[str, Any]) -> str:
        prompt = f"{problem}\n\n# Previous Patch\n{prev_patch}\n\n# Output\n{output}\n\n# Errors\n{errors}\n"
        try:
            result = agent.reflect(prompt, task)
            return result.get("best_solution") if isinstance(result, dict) else str(result)
        except Exception as e:
            return prev_patch


