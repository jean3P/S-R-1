#!/bin/bash
###############################################################################
#  UBELIX – SWE‑Bench Integrated Pipeline with Validation on 2 × NVIDIA H100
###############################################################################
#SBATCH --job-name=swebench-integrated-h100
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
mkdir -p results offload_folder jobs_logs

#–– Timestamp & logging ––----------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
EXP=integrated_pipeline_benchmark
LOGFILE=jobs_logs/${EXP}_${TS}.log
RESULTS=results/${EXP}_${TS}
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

#–– Extra deps for validation pipeline ––-------------------------------------
pip install -q sentence-transformers scikit-learn seaborn matplotlib pandas numpy

#–– Clear GPU memory ––-------------------------------------------------------
python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache")
PY

#–– Benchmark configs ––------------------------------------------------------
MODELS=(deepseek-r1-distill)
MAX=5                                  # Maximum number of issues to process
MEMORY_EFFICIENT=true                  # Use memory-efficient processing

# Set additional arguments based on config
ARGS=()
$MEMORY_EFFICIENT && ARGS+=(--memory-efficient)

#–– Run integrated pipeline ––------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    OUTDIR=$RESULTS/$MODEL
    mkdir -p "$OUTDIR"
    echo -e "\n=== Running Integrated Pipeline with $MODEL ==="

    python -m src.scripts.run_integrated_pipeline \
        --config configs/experiments/integrated_pipeline.yaml \
        --model "$MODEL" \
        --limit $MAX \
        --output "$OUTDIR" \
        --log-level "DEBUG" \
        --disable-quantization \
        --disable-flash-attention \
        "${ARGS[@]}"

    # Clear cache after model run
    python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache after model run")
PY
done

#–– Run comparative analysis ––-----------------------------------------------
echo -e "\n=== Running comparative analysis ==="
python -m src.scripts.compare_results \
    --baseline "$RESULTS/baseline" \
    --integrated "$RESULTS/deepseek-r1-distill" \
    --output "$RESULTS/comparison" \
    --visualize

#–– Final report ––-----------------------------------------------------------
echo -e "\n=== Integrated Pipeline Results Summary ==="
cat "$RESULTS/deepseek-r1-distill/summary.json" | python -m json.tool

echo "=== Job finished: $(date) ==="
nvidia-smi
