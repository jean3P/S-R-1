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

# Set environment variables for better GPU utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_OFFLINE=0  # Set to 1 if you're running without internet
export PYTHONPATH=$(pwd)       # Add the current directory to Python path
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

# Define experiment name and problem file
EXPERIMENT_NAME="qwen_coder_experiment"
PROBLEM_FILE="./problems/coding_problem.txt"

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

# Create agent configuration if it doesn't exist
if [ ! -f "configs/agents/code_refinement.yaml" ]; then
    cat > configs/agents/code_refinement.yaml << EOF
id: "code_refinement"
type: "code_refinement"
config:
  max_iterations: 3
  early_stop_on_success: true
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

# Check if experiment config exists, otherwise create it
if [ ! -f "configs/experiments/${EXPERIMENT_NAME}.yaml" ]; then
    echo "Creating experiment configuration..."
    # Use the -m flag to run the module correctly
    python -m src.main create \
      --name "${EXPERIMENT_NAME}" \
      --agent code_refinement \
      --model qwen_coder \
      --prompt code_gen \
      --evaluator python_exec \
      --task "$PROBLEM_FILE" \
      --output "configs/experiments/${EXPERIMENT_NAME}.yaml"
fi

# Run the experiment
echo "Starting experiment execution..."
# Use the -m flag to run the module correctly
python -m src.main run \
  --config "configs/experiments/${EXPERIMENT_NAME}.yaml" \
  --output-dir "results/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)" \
  --log-level INFO

echo "Job completed"