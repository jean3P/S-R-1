#!/bin/bash
# SWE-Bench Self-Reflection Benchmark Script
# This script runs a benchmark experiment using self-reflection for each model
# on the SWE-bench dataset

# Activate virtual environment (adjust path if needed)
source .venv/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=$(pwd)
export HF_TOKEN=
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Using all available GPUs

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
mkdir -p results/self_reflection_benchmark
mkdir -p offload_folder
mkdir -p jobs_logs

# Define experiment name and timestamp
EXPERIMENT_NAME="self_reflection_benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
NUM_ITERATIONS=5

# Create log file
LOG_FILE="jobs_logs/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Download SWE-bench dataset if it doesn't exist
echo "=== Checking for SWE-bench dataset ==="
DATASET_DIR="data/datasets"
DATASET_FILE="${DATASET_DIR}/swe_bench_verified.json"

if [ ! -f "$DATASET_FILE" ]; then
  echo "SWE-bench dataset not found. Downloading..."
  
  # Install required dependencies for download
  pip install requests tqdm
  
  # Run the download script
  python -m src.scripts.download_swe_bench --output "$DATASET_FILE" --lite
  
  if [ $? -ne 0 ]; then
    echo "Error downloading dataset. Please check the logs."
    exit 1
  fi
  
  echo "Dataset downloaded to $DATASET_FILE"
else
  echo "SWE-bench dataset found at $DATASET_FILE"
fi

# Create a symlink to ensure the dataset is accessible from the expected path
mkdir -p src/data/swe-bench-verified
if [ ! -f "src/data/swe-bench-verified/swe_bench_verified.json" ]; then
  echo "Creating symlink to dataset..."
  ln -sf "$(realpath $DATASET_FILE)" "src/data/swe-bench-verified/swe_bench_verified.json"
fi

echo "=== Starting Self-Reflection Benchmark Experiment ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Results directory: ${RESULTS_DIR}"

# Define models to benchmark
MODELS=("deepseek-r1-distill" )
#MODELS=("deepseek-r1-distill" "qwen2-5-coder" "qwq-preview")
# Set maximum number of instances to process
MAX_INSTANCES=1

# Run the self-reflection benchmark
echo ""
echo "=== Running self-reflection benchmark ==="
echo "Using ${NUM_ITERATIONS} reflection iterations"
echo "Max instances: ${MAX_INSTANCES}"

python -m src.scripts.run_self_reflection_benchmark \
  --config configs/experiments/swe_bench_experiment.yaml \
  --model all \
  --reasoning "chain_of_thought" \
  --iterations ${NUM_ITERATIONS} \
  --limit ${MAX_INSTANCES} \
  --output "${RESULTS_DIR}" \
  --log-file "${RESULTS_DIR}/run.log" \
  --debug

echo ""
echo "=== Self-Reflection Benchmark Complete! ==="
echo "Results saved to: ${RESULTS_DIR}"
echo "Log file: ${LOG_FILE}"

# Install required dependencies for visualization
echo ""
echo "=== Installing required dependencies ==="
pip install seaborn matplotlib pandas numpy python-dotenv

# Generate visualizations of the results
echo ""
echo "=== Generating visualizations ==="
python -m src.statistics.benchmark_report \
  --input "${RESULTS_DIR}/self_reflection_benchmark_*.json" \
  --output "${RESULTS_DIR}/visualizations" \
  --format "png"

echo "Visualizations saved to: ${RESULTS_DIR}/visualizations"
