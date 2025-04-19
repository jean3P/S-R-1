#!/bin/bash
# Run model comparison script

# Set environment variables
export HF_TOKEN=
export PYTHONPATH="$(pwd)"

# Create output directory
mkdir -p results/model_comparison

# Run the comparison script
python src/scripts/run_model_comparison.py --models "deepseek-r1-distill,qwen2-5-coder,qwq-preview" --disable-quantization

echo "Model comparison completed!"
