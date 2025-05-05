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
MODEL="deepseek-r1-distill"            # Using only DeepSeek model
MAX_ITERATIONS=5                        # Maximum iterations per bug
MAX_ISSUES=1                           # Maximum number of issues to process
MEMORY_EFFICIENT=true                  # Use memory-efficient processing

# Set additional arguments based on config
ARGS=()
$MEMORY_EFFICIENT && ARGS+=(--memory-efficient)

#–– Copy config file to experiments directory ––-----------------------------
cp configs/experiments/robust_pipeline.yaml configs/experiments/robust_pipeline_run_${TS}.yaml

#–– Run robust bug fixing pipeline ––-----------------------------------------
OUTDIR=$RESULTS/$MODEL
mkdir -p "$OUTDIR"
echo -e "\n=== Running Robust Bug Fixing Pipeline with $MODEL ==="

# Main command to run the pipeline
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

# Clear cache after model run

echo "=== Job finished: $(date) ==="
nvidia-smi
