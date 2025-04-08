#!/bin/bash
# SWE-Bench Evaluation Script for GTX 1080 GPUs with ImprovedCodeRefinementAgent
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
mkdir -p src/context

# Define improved SWE-bench experiment name
EXPERIMENT_NAME="tot_patch_experiment"

# Define main experiment timestamp and results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/improved_swe_bench_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Create results directory for Qwen with improved agent
IMPROVED_RESULTS="${RESULTS_DIR}/improved_agent"
mkdir -p "${IMPROVED_RESULTS}"

# Create log file
LOG_FILE="jobs_logs/improved_swe_bench_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Create SWE-bench dataset configuration
cat > configs/datasets/swe_bench_verified.yaml << EOF
id: "swe_bench_verified"
type: "swe_bench"
config:
  dataset_name: "princeton-nlp/SWE-bench_Lite"
  retrieval_type: "standard"
  file_path: "data/datasets/swe_bench_verified.json"
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

      {repository_context}

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

      # Continuity Context
      {continuity}

      # Task
      Based on the test results above, please refine your solution. The patch should be in the format of a git diff.
      Focus on creating a minimal change that addresses the issue while maintaining the code's integrity.
      Make sure your solution passes all the required tests.

      Please provide your refined patch in git diff format.
  default_variables:
    issue_id: "unknown"
    repo: "unknown"
    base_commit: "unknown"
    continuity: ""
    repository_context: ""
EOF
echo "Created SWE-bench prompt configuration with improved context handling"

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
  temperature: 0.1
  top_p: 0.3
  repetition_penalty: 1.1
  cache_dir: "data/model_cache"
  offload_folder: "offload_folder"
  enable_offloading: true
  low_cpu_mem_usage: true
  torch_dtype: "float16"
EOF
echo "Created Qwen model configuration"

# Create improved agent configuration
cat > configs/agents/improved_code_refinement.yaml << EOF
id: "improved_code_refinement"
type: "improved_code_refinement"  # Must match the agent class name without "Agent"
description: "Improved agent that generates and refines code through self-reflection with token-efficient architecture"

config:
  # General configuration
  max_iterations: 3
  save_results: false
  early_stop_on_success: true
  output_dir: "results/improved_patches"
  repos_dir: "data/repositories"

  # Model generation parameters
  temperature: 0.2
  max_tokens: 4000

  # Token optimization settings
  summarizer_config:
    include_docstrings: true
    include_signatures: true
    include_imports: true

  context_manager_config:
    relevance_threshold: 0.1
    max_files: 10
    include_imports: true
    include_related_functions: true

  disclosure_config:
    max_depth: 3
    max_context_per_file: 2000

  memory_manager_config:
    max_history_items: 5
    max_memory_tokens: 1000

  # Evaluation options
  test_timeout: 1000
  enable_file_exploration: true

  # Logging and debugging
  verbose_logging: true
  save_intermediate_patches: true
EOF
echo "Created improved code refinement agent configuration"

# Create SWE-bench experiment configuration with Tree of Thought patch agent
cat > configs/experiments/${EXPERIMENT_NAME}.yaml << EOF
name: "${EXPERIMENT_NAME}"
description: "SWE-bench evaluation using Tree of Thought patch agent for exploring multiple reasoning paths"
agent:
  id: "tree_of_thought_patch"
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
echo "Created Tree of Thought patch experiment configuration"


# List available registered prompt types
echo "Checking registered prompt types..."
python -m src.main list --type prompts

# List available registered evaluator types
echo "Checking registered evaluator types..."
python -m src.main list --type evaluators

# List available registered agent types
echo "Checking registered agent types..."
python -m src.main list --type agents

# Run SWE-bench evaluation with improved agent (just one instance initially)
MAX_INSTANCES=1

echo "=== Starting SWE-bench evaluation with Tree of Thought patch agent ==="
echo "Model: Qwen/Qwen2-7B-Instruct"
echo "Agent: TreeOfThoughtPatchAgent"
echo "Max instances: ${MAX_INSTANCES}"
echo "Using 4x GTX 1080 GPUs with model distributed across them"

# Run the evaluation
python -m src.scripts.run_reasoning_experiment \
  --name "${EXPERIMENT_NAME}" \
  --dataset "swe_bench_verified" \
  --agent "tree_of_thought_patch" \
  --model "qwen_coder" \
  --prompt "swe_bench_prompt" \
  --evaluator "swe_bench_eval" \
  --max-instances ${MAX_INSTANCES} \
  --output-dir "${IMPROVED_RESULTS}" \
  --log-level INFO

echo "=== Tree of Thought Patch Agent SWE-bench evaluation complete! ==="
