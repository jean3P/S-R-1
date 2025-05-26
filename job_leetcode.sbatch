#!/bin/bash
###############################################################################
#  UBELIX – LeetCode Solution Generator with Self-Reflection
###############################################################################
#SBATCH --job-name=leetcode-solver-h100
#SBATCH --partition=gpu-invest          # investor pool (pre‑emptable)
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:4               # 2 GPUs → 32CPU, 180GB max
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=90G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out.log
###############################################################################
# job_leetcode.sbatch
#–– Software stack ––----------------------------------------------------------
module purge
module unload Python || true           # avoid default Python module

source .venv/bin/activate              # Python3.9 venv with +cu12 wheels

#––Runtime env vars ––--------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=$PWD
export TOKENIZERS_PARALLELISM=false
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ $line =~ ^#.*$ || -z $line ]] && continue
        # Export the variable (this addresses SC2163)
        eval export "$line"
    done < .env
else
    echo "Warning: .env file not found!"
    export HF_TOKEN=""  # Empty fallback
fi

# Add a check to verify the token was loaded
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Please check your .env file."
    exit 1
fi
export FLASH_ATTENTION_FORCE_DISABLE=1                 # <== DISABLE flash-attn

#–– Diagnostics ––------------------------------------------------------------
echo "=== UBELIX H100 job on $(hostname) ==="
date
nvidia-smi
echo "======================================"

#–– Directories ––------------------------------------------------------------
mkdir -p configs/{models,prompts,experiments,datasets}
mkdir -p data/{repositories,cache,huggingface_cache}
mkdir -p results offload_folder jobs_logs
mkdir -p src/evaluation

#–– Timestamp & logging ––----------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
EXP=leetcode_solver
LOGFILE=jobs_logs/${EXP}_${TS}.log
RESULTS=results/${EXP}_${TS}
mkdir -p "$RESULTS"
exec > >(tee -a "$LOGFILE") 2>&1

#–– Install required packages ––---------------------------------------------
echo "Installing required packages..."
pip install -q datasets requests pandas matplotlib seaborn evaluate

#–– LeetCode Dataset Path Setup ––-------------------------------------------
echo -e "\n\n============================================================"
echo -e "=== Setting up LeetCode dataset paths ==="
echo -e "============================================================\n"

# Set the repository path for the LeetCodeDataset
LEETCODE_REPO_PATH="/storage/homefs/jp22b083/SSI/S-R-1/data/repositories/LeetCodeDataset"
LEETCODE_DATA_DIR="${LEETCODE_REPO_PATH}/data"

# Verify dataset files exist
if [ -d "$LEETCODE_DATA_DIR" ]; then
    echo "Found LeetCode dataset directory at: $LEETCODE_DATA_DIR"
    ls -la "$LEETCODE_DATA_DIR" | grep "LeetCodeDataset.*test.jsonl.gz"

    # Count test dataset files
    TEST_FILES=$(ls -1 "$LEETCODE_DATA_DIR"/LeetCodeDataset-*-test.jsonl.gz 2>/dev/null | wc -l)
    if [ "$TEST_FILES" -gt 0 ]; then
        echo "✓ Found $TEST_FILES test dataset files in the repository"
    else
        echo "✗ No test dataset files found in $LEETCODE_DATA_DIR"
    fi
else
    echo "✗ LeetCode dataset directory not found at: $LEETCODE_DATA_DIR"
    echo "Will try to use the LeetCodeDataLoader directly"
fi

#–– Download and preprocess LeetCode dataset ––-----------------------------
echo -e "\n\n============================================================"
echo -e "=== Downloading and preprocessing LeetCode dataset ==="
echo -e "============================================================\n"

# First try using our download script
if [ -f "src/scripts/download_leetcode_dataset.py" ]; then
    echo "Using download_leetcode_dataset.py script to download the dataset"
    python src/scripts/download_leetcode_dataset.py \
        --output-dir "data/leetcode" \
        --cache-dir "data/huggingface_cache" \
        --version "v0.3.1" \
        --limit 60 \
        --log-level "INFO"
else
    # Fallback to direct dataset loading in Python
    echo "Direct dataset download with Python..."
    python - <<'PY'
import os
import json
import gzip
from pathlib import Path

print("Loading LeetCode test dataset...")
cache_dir = "data/huggingface_cache"
output_dir = "data/leetcode"
repo_path = "/storage/homefs/jp22b083/SSI/S-R-1/data/repositories/LeetCodeDataset"
data_dir = os.path.join(repo_path, "data")
test_file = "LeetCodeDataset-v0.3.1-test.jsonl.gz"
file_path = os.path.join(data_dir, test_file)

# Create directories if needed
Path(cache_dir).mkdir(parents=True, exist_ok=True)
Path(output_dir).mkdir(parents=True, exist_ok=True)

try:
    # Check if the file exists in the repository
    if os.path.exists(file_path):
        print(f"Found test dataset file at {file_path}")

        # Load and normalize problems
        problems = []
        count = 0
        limit = 50  # Limit to 50 problems

        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        problem = json.loads(line.strip())

                        # Normalize problem data
                        normalized = {
                            "problem_id": problem.get("task_id", f"problem_{count}"),
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

                        problems.append(normalized)

                        count += 1
                        if count >= limit:
                            break
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}")
                        continue

        print(f"Loaded {len(problems)} problems from the test dataset")

        # Save problems to JSON file
        output_file = os.path.join(output_dir, "leetcode_problems.json")
        with open(output_file, 'w') as f:
            json.dump(problems, f, indent=2)

        print(f"Saved {len(problems)} problems to {output_file}")

        # Save a problem list for reference
        problem_list = []
        for p in problems:
            problem_list.append({
                "problem_id": p["problem_id"],
                "title": p.get("title", p["problem_id"]),
                "difficulty": p.get("difficulty", ""),
                "tags": p.get("tags", [])
            })

        list_file = os.path.join(output_dir, "problem_list.json")
        with open(list_file, 'w') as f:
            json.dump(problem_list, f, indent=2)

        print(f"Saved problem list to {list_file}")
    else:
        print(f"Test dataset file not found at {file_path}")
        print("Attempting to load from Hugging Face datasets...")

        # Try using datasets library
        try:
            from datasets import load_dataset

            dataset = load_dataset("newfacade/LeetCodeDataset", split="test", cache_dir=cache_dir)
            print(f"Loaded {len(dataset)} problems from Hugging Face")

            # Process the first 50 problems to save to JSON
            problems = []
            for i, example in enumerate(dataset):
                if i >= 50:  # Limit to 50 problems
                    break

                # Extract problem data
                problem = {
                    "problem_id": example.get("task_id", f"problem_{i}"),
                    "prompt": example.get("prompt", ""),
                    "query": example.get("query", ""),
                    "entry_point": example.get("entry_point", ""),
                    "test": example.get("test", ""),
                    "input_output": example.get("input_output", []),
                    "reference_solution": example.get("completion", "")
                }

                # Extract metadata if available
                if "meta" in example and example["meta"]:
                    meta = example["meta"]
                    problem["difficulty"] = meta.get("difficulty", "")
                    problem["tags"] = meta.get("tags", [])
                    problem["title"] = meta.get("question_title", problem["problem_id"])

                problems.append(problem)

            # Save problems to JSON file
            output_file = os.path.join(output_dir, "leetcode_problems.json")
            with open(output_file, 'w') as f:
                json.dump(problems, f, indent=2)

            print(f"Saved {len(problems)} problems to {output_file}")

            # Save a problem list for reference
            problem_list = []
            for p in problems:
                problem_list.append({
                    "problem_id": p["problem_id"],
                    "title": p.get("title", p["problem_id"]),
                    "difficulty": p.get("difficulty", ""),
                    "tags": p.get("tags", [])
                })

            list_file = os.path.join(output_dir, "problem_list.json")
            with open(list_file, 'w') as f:
                json.dump(problem_list, f, indent=2)

            print(f"Saved problem list to {list_file}")

        except Exception as e:
            print(f"Error loading dataset from Hugging Face: {e}")
            print("Warning: Dataset loading failed, the solver may not work properly")

except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Warning: Dataset loading failed, the solver may not work properly")
PY
fi

# Verify dataset is available
echo "Checking dataset availability..."
if [ -f "data/leetcode/leetcode_problems.json" ]; then
    echo "✓ LeetCode dataset preprocessing successful"
    # Count problems and get a sample of problem IDs
    python - <<'PY'
import json
import os

try:
    with open("data/leetcode/leetcode_problems.json", 'r') as f:
        problems = json.load(f)

    print(f"Found {len(problems)} preprocessed LeetCode problems")

    # Get a sample of problem IDs
    sample_ids = [p.get("problem_id", "") for p in problems[:5]]
    print(f"Sample problem IDs: {', '.join(sample_ids)}")

    # Count problems by difficulty
    difficulties = {}
    for p in problems:
        diff = p.get("difficulty", "Unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1

    print("Problems by difficulty:")
    for diff, count in difficulties.items():
        print(f"  {diff}: {count}")

except Exception as e:
    print(f"Error analyzing dataset: {e}")
PY
else
    echo "✗ LeetCode dataset preprocessing failed"
    echo "Will try to use the LeetCodeDataLoader directly"
fi

# Create the code evaluator file if it doesn't exist
if [ ! -f "src/evaluation/code_evaluator.py" ]; then
    echo -e "\n\n============================================================"
    echo -e "=== Creating Code Evaluator module ==="
    echo -e "============================================================\n"

    cat > src/evaluation/code_evaluator.py << 'EOF'
"""
Code evaluation module using HuggingFace's code_eval metric.
Provides standardized pass@k evaluation for code generation.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
import tempfile
import traceback

logger = logging.getLogger(__name__)

class CodeEvaluator:
    """
    Evaluator for code solutions using Hugging Face's code_eval metric.
    """

    def __init__(self, config):
        """
        Initialize the code evaluator.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.evaluation_config = config.get("evaluation", {})

        # Get code_eval specific configuration
        self.code_eval_config = self.evaluation_config.get("code_eval", {})
        self.k_values = self.code_eval_config.get("k_values", [1, 3, 5, 10])
        self.num_workers = self.code_eval_config.get("num_workers", 4)
        self.timeout = self.code_eval_config.get("timeout", 3.0)

        # Set environment variable to allow code execution (with warning)
        if not os.environ.get("HF_ALLOW_CODE_EVAL"):
            logger.warning(
                "Setting HF_ALLOW_CODE_EVAL=1. This permits execution of untrusted model-generated code. "
                "Make sure this is running in a sandbox environment."
            )
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        # Load the code_eval metric
        try:
            from evaluate import load
            self.code_eval = load("code_eval")
            logger.info("Successfully loaded code_eval metric")
        except Exception as e:
            logger.error(f"Failed to load code_eval metric: {str(e)}")
            self.code_eval = None

    def evaluate_solutions(self, problem_data: Dict[str, Any], solutions: List[str]) -> Dict[str, Any]:
        """
        Evaluate a list of solutions against the problem's test cases.

        Args:
            problem_data: Problem data dictionary
            solutions: List of solution code strings

        Returns:
            Dictionary with evaluation metrics
        """
        if self.code_eval is None:
            logger.error("Cannot evaluate solutions: code_eval metric not loaded")
            return {"error": "code_eval metric not loaded"}

        if not solutions:
            logger.warning("No solutions to evaluate")
            return {"error": "no solutions provided"}

        # Create test cases from problem data
        test_cases = self._create_test_cases(problem_data)
        if not test_cases:
            logger.error("Failed to create test cases")
            return {"error": "failed to create test cases"}

        # Format solutions for code_eval
        formatted_solutions = [solutions]  # code_eval expects a list of lists

        try:
            # Compute pass@k using code_eval
            logger.info(f"Evaluating {len(solutions)} solutions with pass@{self.k_values}")
            pass_at_k, results = self.code_eval.compute(
                references=test_cases,
                predictions=formatted_solutions,
                k=self.k_values,
                num_workers=self.num_workers,
                timeout=self.timeout
            )

            # Format and return results
            evaluation = {
                "pass_at_k": pass_at_k,
                "detailed_results": results,
                "solutions_evaluated": len(solutions),
                "test_cases": len(test_cases)
            }

            return evaluation

        except Exception as e:
            logger.error(f"Error during code_eval evaluation: {str(e)}")
            logger.debug(traceback.format_exc())
            return {"error": str(e)}

    def _create_test_cases(self, problem_data: Dict[str, Any]) -> List[str]:
        """
        Create HuggingFace code_eval compatible test cases from problem data.

        Args:
            problem_data: Problem data dictionary

        Returns:
            List of test case strings
        """
        test_cases = []
        entry_point = problem_data.get("entry_point", "")

        # Extract method name from entry point
        method_name = ""
        if "." in entry_point:
            method_name = entry_point.split(".")[-1]

        # If we have test code, create a test wrapper
        if problem_data.get("test"):
            test_wrapper = f"""
def check_solution(candidate):
    import unittest
    import sys
    from io import StringIO

    # Store original stdout
    original_stdout = sys.stdout

    # Create a string buffer to capture output
    test_output = StringIO()
    sys.stdout = test_output

    try:
        # Execute test code
        {problem_data.get("test", "")}

        # Execute extra assertions
        {self._create_assertions_from_io(problem_data, method_name)}

        # If we reach here, all tests passed
        return True
    except AssertionError as e:
        # Test failure
        return False
    except Exception as e:
        # Other error
        return False
    finally:
        # Restore stdout
        sys.stdout = original_stdout

# Test the solution
result = check_solution({method_name})
assert result == True
"""
            test_cases.append(test_wrapper)

        # Create additional test cases from input-output pairs
        if problem_data.get("input_output"):
            for io_pair in problem_data.get("input_output", []):
                try:
                    input_data = io_pair.get("input", [])
                    expected_output = io_pair.get("output")

                    # Format inputs for assertion
                    if isinstance(input_data, list):
                        inputs_str = ", ".join(str(i) for i in input_data)
                    elif isinstance(input_data, dict):
                        inputs_str = ", ".join(f"{k}={v}" for k, v in input_data.items())
                    else:
                        inputs_str = str(input_data)

                    assertion = f"assert {method_name}({inputs_str}) == {expected_output}"
                    test_cases.append(assertion)
                except Exception as e:
                    logger.warning(f"Failed to create test case from I/O pair: {str(e)}")

        return test_cases

    def _create_assertions_from_io(self, problem_data: Dict[str, Any], method_name: str) -> str:
        """
        Create assertion statements from input-output pairs.

        Args:
            problem_data: Problem data dictionary
            method_name: Method name to test

        Returns:
            String with assertion statements
        """
        assertions = []

        for io_pair in problem_data.get("input_output", []):
            try:
                input_data = io_pair.get("input", [])
                expected_output = io_pair.get("output")

                # Format inputs for assertion
                if isinstance(input_data, list):
                    inputs_str = ", ".join(str(i) for i in input_data)
                elif isinstance(input_data, dict):
                    inputs_str = ", ".join(f"{k}={v}" for k, v in input_data.items())
                else:
                    inputs_str = str(input_data)

                assertion = f"assert {method_name}({inputs_str}) == {expected_output}"
                assertions.append(assertion)
            except Exception as e:
                logger.warning(f"Failed to create assertion from I/O pair: {str(e)}")

        return "\n".join(assertions)
EOF

    echo "Created src/evaluation/code_evaluator.py"

    # Create an __init__.py file in the evaluation directory
    touch src/evaluation/__init__.py
    echo "Created src/evaluation/__init__.py"
fi

#–– Clear GPU memory ––-------------------------------------------------------
python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache")
PY

#–– Benchmark configs ––------------------------------------------------------
MODELS=("deepseek-r1-distill" "qwen2-5-coder" "qwq-preview")
NUM_PROBLEMS=10                       # Number of problems to process
INITIAL_K=3                          # Initial solutions to generate
BRANCH_FACTOR=3                      # Solutions to generate per failed solution
MAX_DEPTH=3                          # Maximum tree depth
DIFFICULTIES=("Easy" "Medium" "Hard")  # Problem difficulty (Easy, Medium, Hard)
MEMORY_EFFICIENT=true                  # Use memory-efficient processing

# Set additional arguments based on config
ARGS=()
$MEMORY_EFFICIENT && ARGS+=(--memory-efficient)

#–– Copy config file to experiments directory ––-----------------------------
cp configs/experiments/leetcode_solver.yaml configs/experiments/leetcode_solver_run_${TS}.yaml

# Update the config file with dataset paths
echo "Updating config file with dataset paths..."
cat >> configs/experiments/leetcode_solver_run_${TS}.yaml << EOF

# Dataset paths added by job script
data:
  cache_dir: "data/cache"
  huggingface_cache: "data/huggingface_cache"
  leetcode_dir: "data/leetcode"
  leetcode_repo_path: "/storage/homefs/jp22b083/SSI/S-R-1/data/repositories/LeetCodeDataset"

leetcode:
  version: "v0.3.1"
  preprocessed_file: "data/leetcode/leetcode_problems.json"
  initial_solutions: $INITIAL_K
  branch_factor: $BRANCH_FACTOR
  max_depth: $MAX_DEPTH
  early_stopping: false  # Explicitly set to false in config

evaluation:
  use_code_eval: true
  code_eval:
    k_values: [1, 3, 5, 10]
    num_workers: 4
    timeout: 3.0
EOF

#–– Run LeetCode solver for each model and difficulty ––-------------------
for MODEL in "${MODELS[@]}"; do
    echo -e "\n\n============================================================"
    echo -e "=== Running LeetCode Solution Generator with $MODEL ==="
    echo -e "============================================================\n"

    # Create output directory for this model
    OUTDIR=$RESULTS/$MODEL
    mkdir -p "$OUTDIR"

    for DIFFICULTY in "${DIFFICULTIES[@]}"; do
        echo -e "\n\n------------------------------------------------------------"
        echo -e "--- Processing $NUM_PROBLEMS $DIFFICULTY problems ---"
        echo -e "------------------------------------------------------------\n"

        # Clear GPU memory before running each difficulty
        python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache")
PY

        # Run the solution generator with current model and difficulty
        python -m src.scripts.run_leetcode_solver \
            --config configs/experiments/leetcode_solver_run_${TS}.yaml \
            --model "$MODEL" \
            --difficulty "$DIFFICULTY" \
            --limit $NUM_PROBLEMS \
            --initial-k $INITIAL_K \
            --branch-factor $BRANCH_FACTOR \
            --max-depth $MAX_DEPTH \
            --output "$OUTDIR" \
            --log-level "DEBUG" \
            --disable-quantization \
            --disable-flash-attention \
            --use-code-eval \
            "${ARGS[@]}"

        echo -e "\n=== Finished $DIFFICULTY problems for $MODEL: $(date) ==="
        nvidia-smi
    done

    echo -e "\n=== Finished running all difficulties for $MODEL: $(date) ==="
done

#–– Generate comparison report ––--------------------------------------------
echo -e "\n\n============================================================"
echo -e "=== Generating model comparison report ==="
echo -e "============================================================\n"

# Create a comparison script
cat > compare_models.py << 'EOF'
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

results_dir = sys.argv[1]
models = sys.argv[2].split(',')

# Prepare data collection
results = []
code_eval_results = []

# For each model
for model in models:
    model_dir = Path(results_dir) / model

    # Get combined results file
    combined_file = model_dir / "combined_results.json"

    if not combined_file.exists():
        print(f"Warning: Combined results file not found for model {model}")
        continue

    try:
        with open(combined_file, 'r') as f:
            model_results = json.load(f)

        for problem_result in model_results:
            problem_id = problem_result.get("problem_id", "unknown")

            result = {
                "model": model,
                "problem_id": problem_id,
                "problem_title": problem_result.get("problem_title", problem_id),
                "status": problem_result.get("status", "unknown"),
                "processing_time": problem_result.get("processing_time", 0),
                "rounds": problem_result.get("rounds", 0),
                "total_candidates": problem_result.get("total_candidates", 0),
                "difficulty": problem_result.get("difficulty", "Unknown")
            }

            results.append(result)

            # Extract code_eval results if available
            if "code_eval_results" in problem_result and "pass_at_k" in problem_result["code_eval_results"]:
                for k, value in problem_result["code_eval_results"]["pass_at_k"].items():
                    code_eval_results.append({
                        "model": model,
                        "problem_id": problem_id,
                        "k": k,
                        "pass_rate": value * 100  # Convert to percentage
                    })
    except Exception as e:
        print(f"Error processing results for {model}: {e}")

# Create dataframe
if not results:
    print("No results found!")
    sys.exit(1)

df = pd.DataFrame(results)

# Save the raw data
csv_path = Path(results_dir) / "model_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"Saved raw data to {csv_path}")

# Create comparison visualizations
plt.figure(figsize=(12, 8))

# Success rate by model
success_rate = df.groupby('model')['status'].apply(
    lambda x: (x == 'solved').mean() * 100
).reset_index()
success_rate.columns = ['model', 'success_rate']

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='model', y='success_rate', data=success_rate)
plt.title('Success Rate by Model (%)')
plt.ylabel('Success Rate (%)')
plt.xlabel('Model')
plt.ylim(0, 100)
plt.xticks(rotation=45)

for i, v in enumerate(success_rate['success_rate']):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center')

plt.tight_layout()
plt.savefig(Path(results_dir) / "success_rate_comparison.png")

# Success rate by model and difficulty
if 'difficulty' in df.columns and not df['difficulty'].isna().all():
    success_by_diff = df.groupby(['model', 'difficulty'])['status'].apply(
        lambda x: (x == 'solved').mean() * 100
    ).reset_index()
    success_by_diff.columns = ['model', 'difficulty', 'success_rate']

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='model', y='success_rate', hue='difficulty', data=success_by_diff)
    plt.title('Success Rate by Model and Difficulty (%)')
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Model')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "success_rate_by_difficulty.png")

# Average time by model
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='model', y='processing_time', data=df)
plt.title('Average Processing Time by Model (seconds)')
plt.ylabel('Time (seconds)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(Path(results_dir) / "processing_time_comparison.png")

# Number of candidates by model
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='model', y='total_candidates', data=df)
plt.title('Average Candidates Generated by Model')
plt.ylabel('Candidates')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(Path(results_dir) / "candidates_comparison.png")

# Code Eval metrics if available
if code_eval_results:
    code_eval_df = pd.DataFrame(code_eval_results)

    # Create a pass@k comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='model', y='pass_rate', hue='k', data=code_eval_df)
    plt.title('Pass@k Metrics by Model (%)')
    plt.ylabel('Pass Rate (%)')
    plt.xlabel('Model')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "pass_at_k_comparison.png")

    # Save code_eval metrics to CSV
    code_eval_csv = Path(results_dir) / "code_eval_metrics.csv"
    code_eval_df.to_csv(code_eval_csv, index=False)
    print(f"Saved code_eval metrics to {code_eval_csv}")

print(f"Generated comparison visualizations in {results_dir}")
EOF

# Run the comparison script
python compare_models.py "$RESULTS" "$(IFS=,; echo "${MODELS[*]}")"

echo "=== Job finished: $(date) ==="
nvidia-smi
