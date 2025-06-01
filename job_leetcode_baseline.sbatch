#!/bin/bash
###############################################################################
#  UBELIX – LeetCode Baseline Experiments (All Difficulties)
###############################################################################
#SBATCH --job-name=leetcode-baseline-all
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=90G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out.log
###############################################################################

# Module and environment setup
module purge
module unload Python || true
source .venv/bin/activate

# Runtime environment variables
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=512
export TRANSFORMERS_OFFLINE=0
export PYTHONPATH=$PWD
export TOKENIZERS_PARALLELISM=false
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    while IFS= read -r line || [ -n "$line" ]; do
        [[ $line =~ ^#.*$ || -z $line ]] && continue
        eval export "$line"
    done < .env
else
    echo "Warning: .env file not found!"
    export HF_TOKEN=""
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set. Please check your .env file."
    exit 1
fi
export FLASH_ATTENTION_FORCE_DISABLE=1

# Diagnostics
echo "=== UBELIX H100 Baseline Experiments job on $(hostname) ==="
date
nvidia-smi
echo "======================================"

# Directories
mkdir -p configs/{models,prompts,experiments,datasets}
mkdir -p data/{repositories,cache,huggingface_cache}
mkdir -p results offload_folder jobs_logs
mkdir -p src/evaluation

# Timestamp & logging
TS=$(date +%Y%m%d_%H%M%S)
EXP=leetcode_baseline
LOGFILE=jobs_logs/${EXP}_${TS}.log
RESULTS=results/${EXP}_${TS}
mkdir -p "$RESULTS"
exec > >(tee -a "$LOGFILE") 2>&1

# Install required packages
echo "Installing required packages..."
pip install -q datasets requests pandas matplotlib seaborn evaluate

# Clear GPU memory
python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache")
PY

# Copy config file to experiments directory
cp configs/experiments/leetcode_solver.yaml configs/experiments/leetcode_baseline_${TS}.yaml

# Update the config file with baseline settings
echo "Updating config file with baseline settings..."
cat >> configs/experiments/leetcode_baseline_${TS}.yaml << EOF

# Dataset paths added by job script
data:
  cache_dir: "data/cache"
  huggingface_cache: "data/huggingface_cache"
  leetcode_dir: "data/leetcode"
  leetcode_repo_path: "/storage/homefs/jp22b083/SSI/S-R-1/data/repositories/LeetCodeDataset"

# Baseline specific settings
baseline:
  num_solutions: 10  # Generate 10 independent solutions for Pass@k evaluation

# LeetCode specific settings
leetcode:
  version: "v0.3.1"
  test_timeout: 10
  max_test_retries: 2
  use_isolated_env: true

# Evaluation settings
evaluation:
  use_code_eval: true
  code_eval:
    k_values: [1, 3, 5, 10]
    num_workers: 4
    timeout: 25.0

# Memory settings
memory_efficient: true
EOF

# Models and problem counts configuration
MODELS=("deepseek-r1-distill" "qwen2-5-coder" "qwq-preview")
declare -A PROBLEM_COUNTS
PROBLEM_COUNTS["Easy"]=18
PROBLEM_COUNTS["Medium"]=20
PROBLEM_COUNTS["Hard"]=10

# Summary tracking
declare -A TOTAL_SOLVED
declare -A TOTAL_PROBLEMS

# Run baseline experiments for each model and difficulty
for MODEL in "${MODELS[@]}"; do
    echo -e "\n\n============================================================"
    echo -e "=== Running Baseline Experiments with $MODEL ==="
    echo -e "============================================================\n"

    MODEL_DIR=$RESULTS/$MODEL
    mkdir -p "$MODEL_DIR"

    # Initialize counters for this model
    TOTAL_SOLVED[$MODEL]=0
    TOTAL_PROBLEMS[$MODEL]=0

    for DIFFICULTY in "Easy" "Medium" "Hard"; do
        NUM_PROBLEMS=${PROBLEM_COUNTS[$DIFFICULTY]}

        echo -e "\n\n------------------------------------------------------------"
        echo -e "--- Processing $NUM_PROBLEMS $DIFFICULTY problems ---"
        echo -e "------------------------------------------------------------\n"

        # Clear cache before each difficulty
        python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache")
PY

        # Run baseline solver
        python -m src.scripts.run_baseline_solver \
            --config configs/experiments/leetcode_baseline_${TS}.yaml \
            --model "$MODEL" \
            --difficulty "$DIFFICULTY" \
            --limit $NUM_PROBLEMS \
            --output "$MODEL_DIR/$DIFFICULTY" \
            --use-code-eval

        # Extract statistics from the summary file
        SUMMARY_FILE="$MODEL_DIR/$DIFFICULTY/baseline_summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            SOLVED=$(python -c "import json; print(json.load(open('$SUMMARY_FILE'))['solved'])")
            TOTAL=$(python -c "import json; print(json.load(open('$SUMMARY_FILE'))['total_problems'])")

            TOTAL_SOLVED[$MODEL]=$((${TOTAL_SOLVED[$MODEL]} + $SOLVED))
            TOTAL_PROBLEMS[$MODEL]=$((${TOTAL_PROBLEMS[$MODEL]} + $TOTAL))

            echo "✓ $DIFFICULTY: Solved $SOLVED/$TOTAL problems"
        else
            echo "✗ Failed to find summary file for $DIFFICULTY"
        fi

        echo -e "\n=== Finished $DIFFICULTY problems for $MODEL: $(date) ==="
        nvidia-smi | grep MiB
    done

    # Model summary
    echo -e "\n============================================================"
    echo "=== Summary for $MODEL ==="
    echo "Total problems solved: ${TOTAL_SOLVED[$MODEL]}/${TOTAL_PROBLEMS[$MODEL]}"
    SOLVE_RATE=$(python -c "print(f'{${TOTAL_SOLVED[$MODEL]} / ${TOTAL_PROBLEMS[$MODEL]} * 100:.1f}')")
    echo "Overall solve rate: $SOLVE_RATE%"
    echo "============================================================"
done

# Generate comprehensive comparison report
echo -e "\n\n============================================================"
echo -e "=== Generating Comprehensive Baseline Report ==="
echo -e "============================================================\n"

cat > generate_baseline_report.py << 'EOF'
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

results_dir = Path(sys.argv[1])
models = sys.argv[2].split(',')

# Collect all results
all_results = []
pass_at_k_data = []

for model in models:
    for difficulty in ["Easy", "Medium", "Hard"]:
        summary_file = results_dir / model / difficulty / "baseline_summary.json"

        if not summary_file.exists():
            print(f"Warning: Summary file not found for {model} - {difficulty}")
            continue

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        # Extract problem-level results
        for result in summary.get("results", []):
            problem_data = {
                "model": model,
                "difficulty": difficulty,
                "problem_id": result.get("problem_id"),
                "status": result.get("status"),
                "num_passed": result.get("num_solutions_passed", 0),
                "num_generated": result.get("num_solutions_generated", 0),
                "processing_time": result.get("processing_time", 0),
                "unique_solutions": result.get("stats", {}).get("unique_solutions", 0)
            }
            all_results.append(problem_data)

            # Extract Pass@k data
            if "code_eval_results" in result and "pass_at_k" in result["code_eval_results"]:
                for k, pass_rate in result["code_eval_results"]["pass_at_k"].items():
                    pass_at_k_data.append({
                        "model": model,
                        "difficulty": difficulty,
                        "k": int(k),
                        "pass_rate": pass_rate * 100
                    })

# Create DataFrames
df = pd.DataFrame(all_results)
pass_k_df = pd.DataFrame(pass_at_k_data)

# Save raw data
df.to_csv(results_dir / "baseline_all_results.csv", index=False)
pass_k_df.to_csv(results_dir / "baseline_pass_at_k.csv", index=False)

# 1. Overall Success Rate by Model and Difficulty
plt.figure(figsize=(12, 8))
success_rates = df.groupby(['model', 'difficulty'])['status'].apply(
    lambda x: (x == 'solved').mean() * 100
).reset_index()
success_rates.columns = ['model', 'difficulty', 'success_rate']

pivot_success = success_rates.pivot(index='model', columns='difficulty', values='success_rate')
ax = pivot_success.plot(kind='bar', width=0.8)
plt.title('Baseline Success Rate by Model and Difficulty (%)', fontsize=14)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.ylim(0, 100)
plt.legend(title='Difficulty')
plt.xticks(rotation=45)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=3)

plt.tight_layout()
plt.savefig(results_dir / "baseline_success_rate_comparison.png", dpi=300)

# 2. Pass@k Metrics Comparison
if not pass_k_df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, difficulty in enumerate(["Easy", "Medium", "Hard"]):
        ax = axes[i]
        diff_data = pass_k_df[pass_k_df['difficulty'] == difficulty]

        if not diff_data.empty:
            pivot_pass_k = diff_data.pivot(index='k', columns='model', values='pass_rate')
            pivot_pass_k.plot(kind='bar', ax=ax)
            ax.set_title(f'Pass@k - {difficulty} Problems', fontsize=12)
            ax.set_xlabel('k', fontsize=10)
            ax.set_ylabel('Pass Rate (%)', fontsize=10)
            ax.set_ylim(0, 100)
            ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(results_dir / "baseline_pass_at_k_by_difficulty.png", dpi=300)

# 3. Overall statistics table
stats_table = []
for model in models:
    model_data = df[df['model'] == model]

    for difficulty in ["Easy", "Medium", "Hard"]:
        diff_data = model_data[model_data['difficulty'] == difficulty]

        if not diff_data.empty:
            stats_table.append({
                "Model": model,
                "Difficulty": difficulty,
                "Total": len(diff_data),
                "Solved": (diff_data['status'] == 'solved').sum(),
                "Success Rate (%)": f"{(diff_data['status'] == 'solved').mean() * 100:.1f}",
                "Avg Time (s)": f"{diff_data['processing_time'].mean():.1f}",
                "Avg Unique Solutions": f"{diff_data['unique_solutions'].mean():.1f}"
            })

stats_df = pd.DataFrame(stats_table)
print("\n=== Baseline Results Summary ===")
print(stats_df.to_string(index=False))

# Save summary table
stats_df.to_csv(results_dir / "baseline_summary_table.csv", index=False)

# 4. Combined Pass@k across all difficulties
plt.figure(figsize=(10, 6))
overall_pass_k = pass_k_df.groupby(['model', 'k'])['pass_rate'].mean().reset_index()
pivot_overall = overall_pass_k.pivot(index='k', columns='model', values='pass_rate')
ax = pivot_overall.plot(kind='bar')
plt.title('Average Pass@k Across All Difficulties', fontsize=14)
plt.xlabel('k', fontsize=12)
plt.ylabel('Pass Rate (%)', fontsize=12)
plt.ylim(0, 100)
plt.legend(title='Model')

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=3)

plt.tight_layout()
plt.savefig(results_dir / "baseline_overall_pass_at_k.png", dpi=300)

print(f"\nGenerated visualizations and reports in {results_dir}")
print("\nFiles created:")
print("- baseline_all_results.csv")
print("- baseline_pass_at_k.csv")
print("- baseline_summary_table.csv")
print("- baseline_success_rate_comparison.png")
print("- baseline_pass_at_k_by_difficulty.png")
print("- baseline_overall_pass_at_k.png")
EOF

python generate_baseline_report.py "$RESULTS" "$(IFS=,; echo "${MODELS[*]}")"

# Final summary
echo -e "\n\n============================================================"
echo "=== FINAL BASELINE EXPERIMENT SUMMARY ==="
echo "============================================================"
echo "Experiment completed at: $(date)"
echo "Results directory: $RESULTS"
echo ""
echo "Models tested: ${MODELS[*]}"
echo "Problems per difficulty:"
echo "  - Easy: ${PROBLEM_COUNTS[Easy]}"
echo "  - Medium: ${PROBLEM_COUNTS[Medium]}"
echo "  - Hard: ${PROBLEM_COUNTS[Hard]}"
echo ""
echo "Overall results:"
for MODEL in "${MODELS[@]}"; do
    if [ -n "${TOTAL_PROBLEMS[$MODEL]}" ] && [ "${TOTAL_PROBLEMS[$MODEL]}" -gt 0 ]; then
        SOLVE_RATE=$(python -c "print(f'{${TOTAL_SOLVED[$MODEL]} / ${TOTAL_PROBLEMS[$MODEL]} * 100:.1f}')")
        echo "  $MODEL: ${TOTAL_SOLVED[$MODEL]}/${TOTAL_PROBLEMS[$MODEL]} solved ($SOLVE_RATE%)"
    fi
done
echo "============================================================"

nvidia-smi
