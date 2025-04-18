#!/bin/bash
# SWE-Bench Chain of Thought Benchmark Script
# This script runs a benchmark experiment using chain of thought reasoning for each model
# on the first instance of the SWE dataset

# Activate virtual environment (adjust path if needed)
source .venv/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=$(pwd)
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
mkdir -p results/chain_of_thought_benchmark
mkdir -p offload_folder
mkdir -p jobs_logs

# Define experiment name and timestamp
EXPERIMENT_NAME="chain_of_thought_benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Create log file
LOG_FILE="jobs_logs/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Starting Chain of Thought Benchmark Experiment ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Results directory: ${RESULTS_DIR}"

# Define models to benchmark
MODELS=("deepseek-7b" "qwen-7b" "qwen-coder-7b" "qwq-7b")

# Set maximum number of instances to process (just 1 for this benchmark)
MAX_INSTANCES=1

# Run benchmark for each model
for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "=== Running benchmark for model: ${MODEL} ==="
  
  # Create model-specific results directory
  MODEL_RESULTS_DIR="${RESULTS_DIR}/${MODEL}"
  mkdir -p "${MODEL_RESULTS_DIR}"
  
  echo "Using Chain of Thought reasoning"
  echo "Max instances: ${MAX_INSTANCES}"
  
  # Run the evaluation with chain_of_thought reasoning
  python -m src.scripts.run_experiment \
    --config configs/experiments/swe_bench_experiment.yaml \
    --model "${MODEL}" \
    --reasoning "chain_of_thought" \
    --limit ${MAX_INSTANCES} \
    --output "${MODEL_RESULTS_DIR}" \
    --log-file "${MODEL_RESULTS_DIR}/run.log" \
    --debug
  
  echo "Completed benchmark for ${MODEL}"
done

# Generate comparative report
echo ""
echo "=== Generating comparative benchmark report ==="

# Run the benchmark report generator
python -m src.scripts.benchmark_runner \
  --config configs/experiments/swe_bench_experiment.yaml \
  --skip-solving \
  --output "${RESULTS_DIR}/report" \
  --log-file "${RESULTS_DIR}/report.log"

echo ""
echo "=== Chain of Thought Benchmark Complete! ==="
echo "Results saved to: ${RESULTS_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Report directory: ${RESULTS_DIR}/report"

# Install required dependencies
echo ""
echo "=== Installing required dependencies ==="
pip install seaborn matplotlib pandas numpy

# Check if visualization was successful
if [ -d "${RESULTS_DIR}/report" ] && [ -f "${RESULTS_DIR}/report/results.json" ]; then
  # Generate visualizations of the results
  echo ""
  echo "=== Generating visualizations ==="
  python -m src.statistics.benchmark_report \
    --input "${RESULTS_DIR}/report/results.json" \
    --output "${RESULTS_DIR}/visualizations" \
    --format "png"

  echo "Visualizations saved to: ${RESULTS_DIR}/visualizations"
else
  echo ""
  echo "=== Skipping visualizations - report data not available ==="
fi
