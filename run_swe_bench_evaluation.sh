#!/bin/bash
# SWE-Bench Evaluation Script for GTX 1080 GPUs
# This script runs evaluation without SLURM batch commands

# Activate virtual environment (adjust path if needed)
source .venv/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Using all four 1080 GPUs

# Log machine information for reproducibility
echo "=== Job Information ==="
date
hostname
nvidia-smi
echo "======================="

# Create necessary directories
mkdir -p configs/models
mkdir -p configs/agents
mkdir -p configs/prompts
mkdir -p configs/evaluators
mkdir -p configs/experiments
mkdir -p configs/datasets
mkdir -p data/datasets
mkdir -p data/repositories
mkdir -p results/swe_bench
mkdir -p offload_folder
mkdir -p jobs_logs

# Define SWE-bench experiment name
EXPERIMENT_NAME="qwen_swe_bench"

# Define main experiment timestamp and results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/swe_bench_run_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Create results directory for Qwen
QWEN_RESULTS="${RESULTS_DIR}/qwen_coder"
mkdir -p "${QWEN_RESULTS}"

# Create log file
LOG_FILE="jobs_logs/swe_bench_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Create SWE-bench dataset configuration
cat > configs/datasets/swe_bench_lite.yaml << EOF
id: "swe_bench_lite"
type: "swe_bench"
config:
  dataset_name: "princeton-nlp/SWE-bench_Lite"
  retrieval_type: "standard"
  file_path: "data/datasets/swe_bench_lite.json"
  repos_dir: "data/repositories"
  cache_dir: "data/datasets"
  auto_load: true
  lazy_loading: true
EOF
echo "Created SWE-bench dataset configuration"

# Create SWE-bench evaluator configuration
cat > configs/evaluators/swe_bench_eval.yaml << EOF
id: "swe_bench_eval"
type: "swe_bench_evaluator"
config:
  timeout: 300
  repos_dir: "data/repositories"
  venv_dir: "data/venvs"
EOF
echo "Created SWE-bench evaluator configuration"

# Create prompt configuration for SWE-bench
cat > configs/prompts/swe_bench_prompt.yaml << EOF
id: "swe_bench_prompt"
type: "swe_bench_prompt"
config:
  system_message: "You are an expert software developer tasked with fixing GitHub issues. You understand how to read issue descriptions, identify the underlying problems, and create patches to fix them. Follow best software engineering practices and ensure your patches are minimal and focused on solving the specific issue."
  templates:
    generation: |
      # GitHub Issue: {issue_id}

      {problem_statement}

      # Repository Information
      Repository: {repo}
      Base commit: {base_commit}

      # Task
      Your task is to create a patch that fixes this issue. The patch should be in the format of a git diff.
      Focus on creating a minimal change that addresses the issue while maintaining the code's integrity.

      Please provide a patch in git diff format.

    reflection: |
      # GitHub Issue: {issue_id}

      {problem_statement}

      # Your Previous Solution
      {solution}

      # Test Results
      {output}

      # Errors
      {errors}

      # Task
      Based on the test results above, please refine your solution. The patch should be in the format of a git diff.
      Focus on creating a minimal change that addresses the issue while maintaining the code's integrity.
      Make sure your solution passes all the required tests.

      Please provide your refined patch in git diff format.
  default_variables:
    issue_id: "unknown"
    repo: "unknown"
    base_commit: "unknown"
EOF
echo "Created SWE-bench prompt configuration"

# Create Qwen model configuration - optimized for GTX 1080 with 8GB VRAM
cat > configs/models/qwen_coder.yaml << EOF
id: "qwen_coder"
type: "huggingface"
config:
  model_name: "Qwen/Qwen2-7B-Instruct"
  device_map: "auto"  # Will distribute across available GPUs
  use_fp16: true
  use_4bit: true
  use_8bit: false      # Enable 8-bit quantization for VRAM efficiency
  max_length: 2048
  temperature: 0.2
  top_p: 0.9
  repetition_penalty: 1.1
  cache_dir: "data/model_cache"
  offload_folder: "offload_folder"
  enable_offloading: true
  low_cpu_mem_usage: true
  torch_dtype: "float16"
EOF
echo "Created Qwen model configuration"

# Create agent configuration
cat > configs/agents/patch_refinement.yaml << EOF
id: "patch_refinement"
type: "patch_refinement"
config:
  max_iterations: 2
  early_stop_on_success: true
  save_results: true
  output_dir: "results"
EOF
echo "Created agent configuration"

# Create SWE-bench experiment configuration
cat > configs/experiments/${EXPERIMENT_NAME}.yaml << EOF
name: "${EXPERIMENT_NAME}"
description: "SWE-bench evaluation using Qwen model on GTX 1080 GPUs"
agent:
  id: "patch_refinement"
model:
  id: "qwen_coder"
prompt:
  id: "swe_bench_prompt"
evaluator:
  id: "swe_bench_eval"
task:
  name: "swe_bench_task"
  language: "python"
  initial_prompt: "This is a placeholder. The actual prompt will be generated from the SWE-bench dataset."
EOF
echo "Created SWE-bench experiment configuration"

# List available registered prompt types
echo "Checking registered prompt types..."
python -m src.main list --type prompts

# List available registered evaluator types
echo "Checking registered evaluator types..."
python -m src.main list --type evaluators

# Run SWE-bench evaluation with just one instance initially
MAX_INSTANCES=1

echo "=== Starting SWE-bench evaluation ==="
echo "Model: Qwen/Qwen2-7B-Instruct"
echo "Max instances: ${MAX_INSTANCES}"
echo "Using 4x GTX 1080 GPUs with model distributed across them"

# Run the evaluation
python -m src.main swe-bench \
  --experiment "${EXPERIMENT_NAME}" \
  --dataset "swe_bench_lite" \
  --output-dir "${QWEN_RESULTS}" \
  --max-instances ${MAX_INSTANCES} \
  --log-level INFO \
  --batch-size 1

# Check if the evaluation was successful
if [ $? -eq 0 ]; then
  echo "Evaluation completed successfully!"

  # If successful with one instance, try more
  echo "Trying with 2 instances..."
  MAX_INSTANCES=2
  QWEN_RESULTS_2="${RESULTS_DIR}/qwen_coder_2instances"
  mkdir -p "${QWEN_RESULTS_2}"

  python -m src.main swe-bench \
    --experiment "${EXPERIMENT_NAME}" \
    --dataset "swe_bench_lite" \
    --output-dir "${QWEN_RESULTS_2}" \
    --max-instances ${MAX_INSTANCES} \
    --log-level INFO \
    --batch-size 1
else
  echo "Evaluation failed. Check the logs for errors."
fi

echo "=== SWE-bench evaluation complete! ==="
