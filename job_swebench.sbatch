#!/bin/bash
###############################################################################
#  UBELIX – SWE‑Bench Chain‑of‑Thought Benchmark on 2 × NVIDIA H100
###############################################################################
#SBATCH --job-name=swebench-cot-h100
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

#–– Directories ––------------------------------------------------------------
mkdir -p configs/{models,prompts,experiments,datasets}
mkdir -p data/{datasets,repositories,cache/embeddings}
mkdir -p src/data/swe-bench-verified
mkdir -p results offload_folder jobs_logs

#–– Timestamp & logging ––----------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
EXP=chain_of_thought_benchmark
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

#–– Extra deps for RAG & plots ––--------------------------------------------
pip install -q sentence-transformers scikit-learn seaborn matplotlib pandas numpy

#–– Benchmark loop ––---------------------------------------------------------
MODELS=(deepseek-r1-distill)
MAX=1
ITER=3
USE_LLM_GUIDANCE=true
USE_RAG=true

LLM_ARGS=()
$USE_LLM_GUIDANCE && LLM_ARGS+=(--use-llm-guidance --guidance-iterations 3)
RAG_ARGS=()
$USE_RAG && RAG_ARGS+=(--memory-efficient --max-context-length 100000)

python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache")
PY

for MODEL in "${MODELS[@]}"; do
    OUTDIR=$RESULTS/$MODEL
    mkdir -p "$OUTDIR"
    echo -e "\n=== Running $MODEL ==="
    python -m src.scripts.run_experiment \
        --config configs/experiments/swe_bench_experiment.yaml \
        --model "$MODEL" \
        --reasoning chain_of_thought \
        --limit $MAX \
        --iterations $ITER \
        --output "$OUTDIR" \
        --log-file "$OUTDIR/run.log" \
        --debug \
        --disable-quantization \
        --disable-flash-attention \
        --max-new-tokens 3048 \
        "${RAG_ARGS[@]}" "${LLM_ARGS[@]}"

    python - <<'PY'
import torch
torch.cuda.empty_cache()
print("[+] Cleared CUDA cache after model run")
PY
done

#–– Report & visualisations ––------------------------------------------------
python -m src.scripts.benchmark_runner \
    --config configs/experiments/swe_bench_experiment.yaml \
    --skip-solving \
    --output "$RESULTS/report" \
    --log-file "$RESULTS/report.log"

if [[ -f $RESULTS/report/results.json ]]; then
    python -m src.statistics.benchmark_report \
        --input  "$RESULTS/report/results.json" \
        --output "$RESULTS/visualizations" \
        --format png
fi

echo "=== Job finished: $(date) ==="
nvidia-smi
