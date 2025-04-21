#!/bin/bash
# SWE-Bench Chain of Thought Benchmark Script
# This script runs a benchmark experiment using chain of thought reasoning for each model
# on the first instance of the SWE dataset

# Activate virtual environment (adjust path if needed)
source .venv/bin/activate

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=$(pwd)
export HF_TOKEN=
export TOKENIZERS_PARALLELISM=false
USE_LLM_GUIDANCE=true
USE_RAG=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
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
mkdir -p data/cache/embeddings
mkdir -p results/chain_of_thought_benchmark
mkdir -p offload_folder
mkdir -p jobs_logs

# Define experiment name and timestamp
EXPERIMENT_NAME="chain_of_thought_benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
NUM_ITERATIONS=3
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

# Install necessary dependencies for RAG system
echo "=== Installing dependencies for RAG system ==="
pip install -q sentence-transformers scikit-learn

echo "=== Starting Chain of Thought Benchmark Experiment ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Results directory: ${RESULTS_DIR}"

# Define models to benchmark
MODELS=("deepseek-r1-distill")
#MODELS=("deepseek-r1-distill" "qwen2-5-coder" "qwq-preview")

# Set maximum number of instances to process (just 1 for this benchmark)
MAX_INSTANCES=1

# Add LLM guidance flag if enabled
LLM_GUIDANCE_ARGS=""
if [ "$USE_LLM_GUIDANCE" = true ]; then
  LLM_GUIDANCE_ARGS="--use-llm-guidance --guidance-iterations 3"
fi

# Add RAG flag if enabled
RAG_ARGS=""
if [ "$USE_RAG" = true ]; then
  RAG_ARGS="--memory-efficient --max-context-length 4096"
  echo "Using memory-efficient RAG mode"
fi

if [ -x "$(command -v nvidia-smi)" ]; then
  echo "Clearing CUDA cache before starting..."
  python -c "import torch; torch.cuda.empty_cache()" || true
fi

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
    --iterations ${NUM_ITERATIONS} \
    --output "${MODEL_RESULTS_DIR}" \
    --log-file "${MODEL_RESULTS_DIR}/run.log" \
    --debug \
    --max-new-tokens 1024 \
    $RAG_ARGS \
    $LLM_GUIDANCE_ARGS

  echo "Completed benchmark for ${MODEL}"

  if [ -x "$(command -v nvidia-smi)" ]; then
    echo "Clearing CUDA cache after ${MODEL}..."
    python -c "import torch; torch.cuda.empty_cache()" || true
  fi
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
pip install seaborn matplotlib pandas numpy python-dotenv

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

# Log memory and performance statistics
if [ -x "$(command -v nvidia-smi)" ]; then
  echo ""
  echo "=== GPU Memory Statistics ==="
  nvidia-smi

  echo ""
  echo "=== Memory-Efficient RAG Performance ==="
  if [ "$USE_RAG" = true ]; then
    echo "RAG mode was enabled for this run"
    echo "Check the logs for retrieval statistics"
  else
    echo "RAG mode was not enabled for this run"
  fi
fi
