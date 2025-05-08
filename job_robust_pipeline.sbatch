#!/bin/bash
###############################################################################
#  UBELIX – Robust Bug Fixing Pipeline with DeepSeek on 2 × NVIDIA H100
###############################################################################
#SBATCH --job-name=robust-bugfix-h100
#SBATCH --partition=gpu-invest          # investor pool (pre‑emptable)
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:2               # 2 GPUs → 32CPU, 180GB max
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=90G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out.log
###############################################################################

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
mkdir -p data/{repositories,cache}
mkdir -p results offload_folder jobs_logs

#–– Timestamp & logging ––----------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
EXP=robust_bug_fixing_pipeline
LOGFILE=jobs_logs/${EXP}_${TS}.log
RESULTS=results/${EXP}_${TS}
mkdir -p "$RESULTS"
exec > >(tee -a "$LOGFILE") 2>&1

#–– Extra deps for validation pipeline ––-------------------------------------
pip install -q sentence-transformers scikit-learn seaborn matplotlib pandas numpy

#–– Clear GPU memory ––-------------------------------------------------------
python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache")
PY

#–– Benchmark configs ––------------------------------------------------------
#MODELS=("qwq-preview")
MODELS=("deepseek-r1-distill" "qwen2-5-coder" "qwq-preview")
MAX_ITERATIONS=5                        # Maximum iterations per bug
MAX_ISSUES=1                           # Maximum number of issues to process
MEMORY_EFFICIENT=true                  # Use memory-efficient processing

# Set additional arguments based on config
ARGS=()
$MEMORY_EFFICIENT && ARGS+=(--memory-efficient)

#–– Copy config file to experiments directory ––-----------------------------
cp configs/experiments/robust_pipeline.yaml configs/experiments/robust_pipeline_run_${TS}.yaml

#–– Run robust bug fixing pipeline for each model ––-------------------------
for MODEL in "${MODELS[@]}"; do
    echo -e "\n\n============================================================"
    echo -e "=== Running Robust Bug Fixing Pipeline with $MODEL ==="
    echo -e "============================================================\n"

    # Clear GPU memory before running each model
    python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache")
PY

    # Create output directory for this model
    OUTDIR=$RESULTS/$MODEL
    mkdir -p "$OUTDIR"

    # Run the pipeline with current model
    python -m src.scripts.main \
        --config configs/experiments/robust_pipeline_run_${TS}.yaml \
        --model "$MODEL" \
        --limit $MAX_ISSUES \
        --max-iterations $MAX_ITERATIONS \
        --output "$OUTDIR" \
        --log-level "DEBUG" \
        --disable-quantization \
        --disable-flash-attention \
        "${ARGS[@]}"

    echo -e "\n=== Finished running $MODEL: $(date) ==="
    nvidia-smi
done

#–– Generate comparison report ––--------------------------------------------
echo -e "\n\n============================================================"
echo -e "=== Generating model comparison report ==="
echo -e "============================================================\n"

# Create a simple comparison script
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

# For each model
for model in models:
    model_dir = Path(results_dir) / model
    if not model_dir.exists():
        print(f"Warning: Directory for model {model} not found at {model_dir}")
        continue

    # Look for bug results
    bug_results = list(model_dir.glob("bugs/*.json"))

    for bug_file in bug_results:
        try:
            with open(bug_file, 'r') as f:
                bug_data = json.load(f)

            # Extract key metrics
            bug_id = bug_data.get("bug_id", "unknown")
            status = bug_data.get("status", "unknown")
            iterations = bug_data.get("iterations", 0)
            processing_time = bug_data.get("processing_time", 0)

            # Get statistics if available
            stats = bug_data.get("stats", {})
            syntax_failures = stats.get("syntax_failures", 0)
            test_failures = stats.get("test_failures", 0)
            time_to_valid_patch = stats.get("time_to_valid_patch", 0)
            time_to_solution = stats.get("time_to_solution", 0)

            # Store result
            results.append({
                "model": model,
                "bug_id": bug_id,
                "status": status,
                "iterations": iterations,
                "processing_time": processing_time,
                "syntax_failures": syntax_failures,
                "test_failures": test_failures,
                "time_to_valid_patch": time_to_valid_patch,
                "time_to_solution": time_to_solution
            })
        except Exception as e:
            print(f"Error processing {bug_file}: {e}")

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
    lambda x: (x == 'success_test_passed').mean() * 100
).reset_index()
success_rate.columns = ['model', 'success_rate']

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='model', y='success_rate', data=success_rate)
plt.title('Success Rate by Model (%)')
plt.ylabel('Success Rate (%)')
plt.xlabel('Model')
plt.xticks(rotation=45)

for i, v in enumerate(success_rate['success_rate']):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center')

plt.tight_layout()
plt.savefig(Path(results_dir) / "success_rate_comparison.png")

# Time to solution by model
valid_times = df[df['time_to_solution'] > 0]
if not valid_times.empty:
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='model', y='time_to_solution', data=valid_times)
    plt.title('Average Time to Solution by Model (seconds)')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "time_to_solution_comparison.png")

# Number of iterations by model
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='model', y='iterations', data=df)
plt.title('Average Iterations by Model')
plt.ylabel('Iterations')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(Path(results_dir) / "iterations_comparison.png")

# Syntax failures by model
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='model', y='syntax_failures', data=df)
plt.title('Average Syntax Failures by Model')
plt.ylabel('Syntax Failures')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(Path(results_dir) / "syntax_failures_comparison.png")

print(f"Generated comparison visualizations in {results_dir}")
EOF

# Run the comparison script
python compare_models.py "$RESULTS" "$(IFS=,; echo "${MODELS[*]}")"

echo "=== Job finished: $(date) ==="
nvidia-smi
