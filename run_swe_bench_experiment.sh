#!/bin/bash
# Run SWE-Bench self-reasoning experiments across multiple models
# Activate virtual environment (adjust path if needed)
source .venv/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Using all four 1080 GPUs

# Default values
MODELS=("qwen_coder")
#MODELS=("qwen_coder" "deepseek_qwen" "qwq_preview")
AGENT="improved_code_refinement"
MAX_INSTANCES=1
MAX_ITERATIONS=1
REPO_DIR="data/repositories"
TIMEOUT=300
DATASET="swe_bench_verified"
PROMPT="swe_bench_prompt"
EVALUATOR="swe_bench_eval"
LOG_LEVEL="INFO"
ANALYZE=true
INSTANCE_ID=""
BASELINE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --agent)
      AGENT="$2"
      shift 2
      ;;
    --max-instances)
      MAX_INSTANCES="$2"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --repos-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --evaluator)
      EVALUATOR="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --no-analyze)
      ANALYZE=false
      shift
      ;;
    --instance-id)
      INSTANCE_ID="$2"
      shift 2
      ;;
    --baseline)
      BASELINE="--baseline"
      shift
      ;;
    --help)
      echo "Run SWE-Bench self-reasoning experiments across multiple models"
      echo ""
      echo "Options:"
      echo "  --agent AGENT            Agent ID to use (default: improved_code_refinement)"
      echo "  --max-instances N        Maximum number of instances to test (default: 1)"
      echo "  --max-iterations N       Maximum number of self-reasoning iterations (default: 3)"
      echo "  --repos-dir DIR          Directory to store repositories (default: data/repositories)"
      echo "  --timeout SECONDS        Timeout for test execution in seconds (default: 300)"
      echo "  --dataset DATASET        Dataset name (default: swe_bench_verified)"
      echo "  --prompt PROMPT          Prompt template ID (default: swe_bench_prompt)"
      echo "  --evaluator EVALUATOR    Evaluator ID (default: swe_bench_eval)"
      echo "  --log-level LEVEL        Logging level (default: INFO)"
      echo "  --no-analyze             Disable results analysis"
      echo "  --instance-id ID         Specific instance ID to test"
      echo "  --baseline               Run baseline test with original patches"
      echo "  --help                   Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p $REPO_DIR
mkdir -p results/experiments
mkdir -p logs/model_comparisons

# Current date and time for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/model_comparisons/comparison_run_${TIMESTAMP}.log"

echo "=== STARTING MODEL COMPARISON RUN ===" | tee -a $LOG_FILE
echo "Date: $(date)" | tee -a $LOG_FILE
echo "Dataset: $DATASET" | tee -a $LOG_FILE
echo "Models to test: ${MODELS[*]}" | tee -a $LOG_FILE
echo "Instance ID: ${INSTANCE_ID:-'All instances'}" | tee -a $LOG_FILE
echo "=========================================" | tee -a $LOG_FILE

# Run the experiment for each model
for MODEL in "${MODELS[@]}"; do
  echo "" | tee -a $LOG_FILE
  echo "=== TESTING MODEL: $MODEL ===" | tee -a $LOG_FILE
  echo "Starting at: $(date)" | tee -a $LOG_FILE
  echo "-----------------------------------" | tee -a $LOG_FILE

  # Clear CUDA cache between model runs
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

  # Name for this specific experiment run
  NAME="swe_bench_${MODEL}_${TIMESTAMP}"

  # Build the command
  CMD="python -m src.scripts.run_reasoning_experiment"
  CMD="$CMD --name $NAME"
  CMD="$CMD --model $MODEL"
  CMD="$CMD --agent $AGENT"
  CMD="$CMD --dataset $DATASET"
  CMD="$CMD --prompt $PROMPT"
  CMD="$CMD --evaluator $EVALUATOR"
  CMD="$CMD --max-instances $MAX_INSTANCES"
  CMD="$CMD --max-iterations $MAX_ITERATIONS"
  CMD="$CMD --repos-dir $REPO_DIR"
  CMD="$CMD --timeout $TIMEOUT"
  CMD="$CMD --log-level $LOG_LEVEL"

  # Add optional arguments
  if [ "$ANALYZE" = false ]; then
    CMD="$CMD --no-analyze"
  fi

  if [ ! -z "$INSTANCE_ID" ]; then
    CMD="$CMD --instance-id $INSTANCE_ID"
  fi

  if [ ! -z "$BASELINE" ]; then
    CMD="$CMD $BASELINE"
  fi

  # Create SWE-bench Verified dataset configuration
cat > configs/datasets/swe_bench_verified.yaml << EOF
id: "swe_bench_verified"
type: "swe_bench"
config:
  dataset_name: "princeton-nlp/SWE-bench_Verified"
  retrieval_type: "standard"
  file_path: "data/datasets/swe_bench_verified.json"
  repos_dir: "data/repositories"
  cache_dir: "data/datasets"
  auto_load: true
  lazy_loading: true
EOF
echo "Created SWE-bench Verified dataset configuration"

echo "  --dataset DATASET        Dataset name (default: swe_bench_verified)"


  # Print the command
  echo "Running: $CMD" | tee -a $LOG_FILE
  echo "-----------------------------------" | tee -a $LOG_FILE

  # Run the experiment and capture output
  $CMD 2>&1 | tee -a $LOG_FILE

  exit_code=${PIPESTATUS[0]}

  if [ $exit_code -ne 0 ]; then
    echo "Experiment for model $MODEL failed with exit code $exit_code" | tee -a $LOG_FILE
    echo "Continuing with next model..." | tee -a $LOG_FILE
  else
    echo "Experiment for model $MODEL completed successfully" | tee -a $LOG_FILE
  fi

  echo "Finished at: $(date)" | tee -a $LOG_FILE
  echo "=========================================" | tee -a $LOG_FILE

  # Sleep a bit between runs to ensure clean transitions
  sleep 10
done

echo "" | tee -a $LOG_FILE
echo "=== ALL MODEL EXPERIMENTS COMPLETED ===" | tee -a $LOG_FILE
echo "Date: $(date)" | tee -a $LOG_FILE
echo "See detailed results in the results/experiments directory" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
