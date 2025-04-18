#!/bin/bash
# Run model comparison script

# Set environment variables
export PYTHONPATH="$(pwd)"

# Create output directory
mkdir -p results/model_comparison

# Run the comparison script
python src/scripts/run_model_comparison.py --models "qwen-32b,qwq-32b" --debug

echo "Model comparison completed!"
