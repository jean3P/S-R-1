#!/bin/bash
#SBATCH --job-name=S-R-1_test         # Job name
#SBATCH --output=./jobs_logs/sr1_%j.log  # Output log file
#SBATCH --partition=gpu                # Submit to the GPU partition
#SBATCH --gres=gpu:h100:1              # Request one NVIDIA H100 GPU
#SBATCH --ntasks=1                     # Single task
#SBATCH --cpus-per-task=4              # Request 4 CPU cores per task
#SBATCH --mem=90GB                     # Total memory for the job
#SBATCH --time=01:00:00                # Job time limit: 1 hour

# Activate virtual environment
source /storage/homefs/jp22b083/SSI/S-R-1/.venv/bin/activate

# Navigate to the project directory
cd /storage/homefs/jp22b083/SSI/S-R-1 || exit

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

# Log machine information for reproducibility
echo "=== Job Information ==="
date
hostname
nvidia-smi
echo "======================="

# Create necessary directories if they don't exist
mkdir -p configs/models
mkdir -p configs/agents
mkdir -p configs/prompts
mkdir -p configs/evaluators
mkdir -p configs/experiments
mkdir -p results
mkdir -p problems

# Define experiment names and problem file
EXPERIMENT_NAME_QWEN="qwen_coder_experiment"
EXPERIMENT_NAME_QWQ="qwq_experiment"
EXPERIMENT_NAME_DS="deepseek_experiment"
PROBLEM_FILE="./problems/coding_problem.txt"

# Define main experiment timestamp and results directory
MAIN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/experiment_run_${MAIN_TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Each model will get its own timestamped directory with clear model identifier
# This preserves timing info while making results easier to compare

# Create a sample problem file if it doesn't exist
if [ ! -f "$PROBLEM_FILE" ]; then
    cat > "$PROBLEM_FILE" << EOF
Write a Python function is_prime(n) that checks if a number is prime.
The function should:
- Return True for prime numbers and False otherwise
- Handle negative numbers and zero (return False for both)
- Be efficient for large numbers
- Include proper documentation and error handling

Examples:
- is_prime(7) should return True
- is_prime(10) should return False
- is_prime(1) should return False
- is_prime(0) should return False
- is_prime(-5) should return False
EOF
    echo "Created sample problem file"
fi

# Create Qwen model configuration if it doesn't exist
if [ ! -f "configs/models/qwen_coder.yaml" ]; then
    cat > configs/models/qwen_coder.yaml << EOF
id: "qwen_coder"
type: "huggingface"
config:
  model_name: "Qwen/Qwen2.5-Coder-32B-Instruct"
  device_map: "auto"
  use_fp16: true
  use_8bit: false
  max_length: 4096
  temperature: 0.2
  top_p: 0.9
  repetition_penalty: 1.1
EOF
    echo "Created Qwen model configuration"
fi

# Create QwQ model config
if [ ! -f "configs/models/qwq_preview.yaml" ]; then
    cat > configs/models/qwq_preview.yaml << EOF
id: "qwq_preview"
type: "huggingface"
config:
  model_name: "Qwen/QwQ-32B-Preview"
  device_map: "auto"
  use_fp16: true
  use_8bit: false
  max_length: 4096
  temperature: 0.2
  top_p: 0.9
  repetition_penalty: 1.1
EOF
    echo "Created QwQ model configuration"
fi

# Create DeepSeek model config
if [ ! -f "configs/models/deepseek_qwen.yaml" ]; then
    cat > configs/models/deepseek_qwen.yaml << EOF
id: "deepseek_qwen"
type: "huggingface"
config:
  model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  device_map: "auto"
  use_fp16: true
  use_8bit: false
  max_length: 4096
  temperature: 0.2
  top_p: 0.9
  repetition_penalty: 1.1
EOF
    echo "Created DeepSeek model configuration"
fi

# Create agent configuration if it doesn't exist
if [ ! -f "configs/agents/code_refinement.yaml" ]; then
    cat > configs/agents/code_refinement.yaml << EOF
id: "code_refinement"
type: "code_refinement"
config:
  max_iterations: 3
  early_stop_on_success: false
  save_results: true
  output_dir: "results"
EOF
    echo "Created agent configuration"
fi

# Create prompt configuration if it doesn't exist
if [ ! -f "configs/prompts/code_gen.yaml" ]; then
    cat > configs/prompts/code_gen.yaml << EOF
id: "code_gen"
type: "code_generation"
config:
  system_message: "You are an expert software developer with deep knowledge of programming languages, algorithms, and best practices. Your task is to write efficient, readable, and well-documented code."
  templates:
    generation: |
      # TASK: {prompt}
      # LANGUAGE: {language}
      {constraints}
      {examples}
      # Please write a solution that meets all requirements:
    reflection: |
      {solution}

      # Execution Output:
      {output}

      # Execution Errors:
      {errors}

      # Based on the above, please refine the solution. Focus on:
      # 1. Fixing any errors
      # 2. Improving efficiency
      # 3. Enhancing readability
      # 4. Adding proper documentation
      # 5. Handling edge cases
      # Provide the complete refined solution:
  default_variables:
    language: "python"
    constraints: ""
    examples: ""
EOF
    echo "Created prompt configuration"
fi

# Create evaluator configuration if it doesn't exist
if [ ! -f "configs/evaluators/python_exec.yaml" ]; then
    cat > configs/evaluators/python_exec.yaml << EOF
id: "python_exec"
type: "python_executor"
config:
  timeout: 30
  install_dependencies: false
  forbidden_modules:
    - "os.system"
    - "subprocess"
    - "shutil.rmtree"
  include_test_cases: true
  test_cases:
    - input: 7
      expected: true
    - input: 10
      expected: false
    - input: 1
      expected: false
    - input: 0
      expected: false
    - input: -5
      expected: false
EOF
    echo "Created evaluator configuration"
fi

# ------------------------------------------------------------------------
# Create experiment configs if they don't already exist
# ------------------------------------------------------------------------

# 1) Qwen Coder Experiment
if [ ! -f "configs/experiments/${EXPERIMENT_NAME_QWEN}.yaml" ]; then
    echo "Creating experiment configuration for Qwen coder..."
    python -m src.main create \
      --name "${EXPERIMENT_NAME_QWEN}" \
      --agent code_refinement \
      --model qwen_coder \
      --prompt code_gen \
      --evaluator python_exec \
      --task "$PROBLEM_FILE" \
      --output "configs/experiments/${EXPERIMENT_NAME_QWEN}.yaml"
fi

# 2) QwQ Experiment
if [ ! -f "configs/experiments/${EXPERIMENT_NAME_QWQ}.yaml" ]; then
    echo "Creating experiment configuration for QwQ..."
    python -m src.main create \
      --name "${EXPERIMENT_NAME_QWQ}" \
      --agent code_refinement \
      --model qwq_preview \
      --prompt code_gen \
      --evaluator python_exec \
      --task "$PROBLEM_FILE" \
      --output "configs/experiments/${EXPERIMENT_NAME_QWQ}.yaml"
fi

# 3) DeepSeek Experiment
if [ ! -f "configs/experiments/${EXPERIMENT_NAME_DS}.yaml" ]; then
    echo "Creating experiment configuration for DeepSeek..."
    python -m src.main create \
      --name "${EXPERIMENT_NAME_DS}" \
      --agent code_refinement \
      --model deepseek_qwen \
      --prompt code_gen \
      --evaluator python_exec \
      --task "$PROBLEM_FILE" \
      --output "configs/experiments/${EXPERIMENT_NAME_DS}.yaml"
fi

# ------------------------------------------------------------------------
# Run each experiment
# ------------------------------------------------------------------------

echo "Starting Qwen Coder experiment..."
QWEN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
QWEN_RESULTS="${RESULTS_DIR}/qwen_coder_${QWEN_TIMESTAMP}"
python -m src.main run \
  --config "configs/experiments/${EXPERIMENT_NAME_QWEN}.yaml" \
  --output-dir "${QWEN_RESULTS}" \
  --log-level INFO

echo "Starting QwQ experiment..."
QWQ_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
QWQ_RESULTS="${RESULTS_DIR}/qwq_preview_${QWQ_TIMESTAMP}"
python -m src.main run \
  --config "configs/experiments/${EXPERIMENT_NAME_QWQ}.yaml" \
  --output-dir "${QWQ_RESULTS}" \
  --log-level INFO

echo "Starting DeepSeek experiment..."
DEEPSEEK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEEPSEEK_RESULTS="${RESULTS_DIR}/deepseek_qwen_${DEEPSEEK_TIMESTAMP}"
python -m src.main run \
  --config "configs/experiments/${EXPERIMENT_NAME_DS}.yaml" \
  --output-dir "${DEEPSEEK_RESULTS}" \
  --log-level INFO

# Function for professional file copying with proper error handling
copy_experiment_results() {
    local experiment_name=$1
    local target_dir=$2
    local timestamp=$(date +"%Y%m%d")

    echo "INFO: Looking for result files for experiment '${experiment_name}'..."

    # Create experiment directory if it doesn't exist
    mkdir -p "$target_dir"

    # Find the most recent result file for this experiment (exact match with experiment name)
    local result_files=($(find "results/" -maxdepth 1 -name "${experiment_name}_*.json" -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | cut -d' ' -f2-))

    if [[ ${#result_files[@]} -eq 0 ]]; then
        echo "WARNING: No exact result files found for ${experiment_name}. Trying alternative pattern..."

        # Try an alternative pattern that might match
        result_files=($(find "results/" -maxdepth 1 -name "*${experiment_name}*.json" -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | cut -d' ' -f2-))

        if [[ ${#result_files[@]} -eq 0 ]]; then
            echo "ERROR: No result files found for experiment '${experiment_name}' - comparison may be incomplete"
            return 1
        fi
    fi

    # Get the most recent file (first in the sorted list)
    local source_file="${result_files[0]}"
    local dest_file="${target_dir}/$(basename "$source_file")"

    # Log what we found
    echo "INFO: Found result file: $source_file"

    # Copy the file with proper error handling
    if cp "$source_file" "$dest_file"; then
        echo "INFO: Successfully copied $(basename "$source_file") to experiment directory"

        # Create a symlink to the experiment_results.json filename that the compare script expects
        ln -sf "$(basename "$source_file")" "${target_dir}/experiment_results.json"
        echo "INFO: Created experiment_results.json symlink"

        # List files in target directory to verify
        ls -la "${target_dir}/"

        return 0
    else
        echo "ERROR: Failed to copy result file for ${experiment_name}"
        return 1
    fi
}

# Copy result files to experiment directories for comparison
echo "Copying result files to experiment directories for comparison..."

# Copy each model's results to its directory with proper error handling
copy_experiment_results "${EXPERIMENT_NAME_QWEN}" "${QWEN_RESULTS}"
copy_experiment_results "${EXPERIMENT_NAME_QWQ}" "${QWQ_RESULTS}"
copy_experiment_results "${EXPERIMENT_NAME_DS}" "${DEEPSEEK_RESULTS}"

# ------------------------------------------------------------------------
# Compare model results - using the new specific results directory
# ------------------------------------------------------------------------

echo "All experiments complete! Now comparing results..."
python -m src.utils.compare_models --results-dir "${RESULTS_DIR}"

# Save the comparison to a dedicated file
COMPARISON_FILE="${RESULTS_DIR}/model_comparison.txt"
python -m src.utils.compare_models --results-dir "${RESULTS_DIR}" --output "${COMPARISON_FILE}"
echo "Comparison saved to ${COMPARISON_FILE}"

# Also save a timestamped copy in the root results directory for easy access
cp "${COMPARISON_FILE}" "results/comparison_${MAIN_TIMESTAMP}.txt"
echo "Comparison also saved to results/comparison_${MAIN_TIMESTAMP}.txt"

# Create a metadata file with experiment details
cat > "${RESULTS_DIR}/experiment_metadata.json" << EOF
{
  "experiment_id": "${MAIN_TIMESTAMP}",
  "execution_date": "$(date -Iseconds)",
  "models": [
    {
      "name": "qwen_coder",
      "timestamp": "${QWEN_TIMESTAMP}",
      "results_dir": "${QWEN_RESULTS}"
    },
    {
      "name": "qwq_preview",
      "timestamp": "${QWQ_TIMESTAMP}",
      "results_dir": "${QWQ_RESULTS}"
    },
    {
      "name": "deepseek_qwen",
      "timestamp": "${DEEPSEEK_TIMESTAMP}",
      "results_dir": "${DEEPSEEK_RESULTS}"
    }
  ],
  "problem_file": "${PROBLEM_FILE}"
}
EOF

echo "Created experiment metadata at ${RESULTS_DIR}/experiment_metadata.json"
