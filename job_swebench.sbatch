#!/bin/bash
###############################################################################
#  UBELIX – SWE‑Bench Context-Based Bug Detector Test on 2 × NVIDIA H100
###############################################################################
#SBATCH --job-name=swebench-bug-detector-h100
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
mkdir -p data/{datasets,repositories,cache/embeddings}
mkdir -p src/data/swe-bench-verified
mkdir -p results/bug_detector offload_folder jobs_logs

#–– Timestamp & logging ––----------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
EXP=bug_detector_test
LOGFILE=jobs_logs/${EXP}_${TS}.log
RESULTS=results/bug_detector/${TS}
mkdir -p "$RESULTS"
exec > >(tee -a "$LOGFILE") 2>&1

#–– Dataset ––----------------------------------------------------------------
DATASET=data/datasets/swe_bench_verified.json
if [[ ! -f $DATASET ]]; then
    echo "[+] Downloading SWE‑bench (lite)…"
    pip install -q requests tqdm pyyaml
    python -m src.scripts.download_swe_bench --output "$DATASET" --lite
fi
ln -sf "$(realpath $DATASET)" src/data/swe-bench-verified/swe_bench_verified.json

#–– Extra deps for bug detector ––-------------------------------------------
pip install -q sentence-transformers scikit-learn seaborn matplotlib pandas numpy networkx

#–– Clear GPU memory ––-------------------------------------------------------
python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache")
PY

#–– Test configuration ––-----------------------------------------------------
# Define issues to test
ISSUES="astropy__astropy-12907"
# Max issues to process if not using specific issues
MAX=5
# Use memory-efficient processing
MEMORY_EFFICIENT=true
# Log level
LOG_LEVEL="INFO"
# Whether to validate and visualize
VALIDATE=true
VISUALIZE=true

# Set additional arguments based on config
ARGS=()
$MEMORY_EFFICIENT && ARGS+=(--memory-efficient)
$VALIDATE && ARGS+=(--validate)
$VISUALIZE && ARGS+=(--visualize)

#–– Test bug detector arguments parsing ––-----------------------------------
echo -e "\n=== Testing Bug Detector Arguments Parsing ==="

# Test with default arguments
echo "Testing with default arguments:"
python -m src.scripts.run_context_bug_detector --limit 1

# Test with specific issues
echo -e "\nTesting with specific issues:"
python -m src.scripts.run_context_bug_detector --issues "$ISSUES" --limit 1

# Test with all arguments
echo -e "\nTesting with all arguments:"
python -m src.scripts.run_context_bug_detector \
    --config configs/experiments/bug_detector.yaml \
    --issues "$ISSUES" \
    --limit $MAX \
    --output "$RESULTS" \
    --log-level "$LOG_LEVEL" \
    "${ARGS[@]}"

#–– Run full bug detector ––-------------------------------------------------
#echo -e "\n=== Running Full Bug Detector Test ==="
#
#python -m src.scripts.run_context_bug_detector \
#    --config configs/experiments/bug_detector.yaml \
#    --issues "$ISSUES" \
#    --limit $MAX \
#    --output "$RESULTS/full_test" \
#    --log-level "DEBUG" \
#    "${ARGS[@]}"

# Clear cache after run
python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache after bug detector run")
PY

#–– Final report ––-----------------------------------------------------------
echo -e "\n=== Bug Detector Results Summary ==="
[ -f "$RESULTS/full_test/summary.json" ] && cat "$RESULTS/full_test/summary.json" | python -m json.tool || echo "Summary not found"

echo "=== Job finished: $(date) ==="
nvidia-smi