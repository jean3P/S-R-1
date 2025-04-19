#!/usr/bin/env python3
# scripts/run_model_comparison.py

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config import Config
from src.models import create_model
from src.utils.logging_utils import setup_logging
from src.utils.file_utils import FileUtils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare model responses")
    parser.add_argument("--models", type=str, default="deepseek-32b,qwen-32b,qwq-32b",
                        help="Comma-separated list of models to compare")
    parser.add_argument("--prompt", type=str, help="Prompt to use")
    parser.add_argument("--prompt-file", type=str, help="File containing the prompt")
    parser.add_argument("--output", type=str, default="results/model_comparison",
                        help="Output directory for results")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--disable-quantization", action="store_true",
                        help="Disable model quantization (helpful if bitsandbytes is not installed)")
    parser.add_argument("--disable-flash-attention", action="store_true",
                        help="Disable flash attention (helpful if flash-attn is not installed)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force using CPU even if CUDA is available")
    return parser.parse_args()


def main():
    """Main function for comparing models."""
    args = parse_args()

    # Load configuration
    config = Config()

    # Set up logging
    if args.debug:
        config["logging"]["log_level"] = "DEBUG"
    setup_logging(config)

    # Check for HF_TOKEN
    if "HF_TOKEN" not in os.environ and "HUGGINGFACE_TOKEN" not in os.environ:
        logging.warning("HF_TOKEN or HUGGINGFACE_TOKEN environment variable not set.")
        logging.warning("You may need it to access some models. Set it with:")
        logging.warning("export HF_TOKEN=your_huggingface_token")

    # Apply command line overrides to config
    if args.cpu_only:
        config["models"]["device"] = "cpu"

    # Get prompt
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read()
    else:
        prompt = "Explain the concept of self-reflection in AI systems and how it can improve problem-solving capabilities."

    # Parse models
    models = args.models.split(",")

    # Create output directory
    output_dir = Path(args.output)
    FileUtils.ensure_directory(output_dir)

    # Run comparison
    results = []

    for model_name in models:
        print(f"\nTesting model: {model_name}")
        try:
            # Modify model config if needed
            if args.disable_quantization:
                model_config = config.get_model_config(model_name)
                if "quantization" in model_config:
                    print(f"Disabling quantization for {model_name}")
                    del model_config["quantization"]

            if args.disable_flash_attention:
                model_config = config.get_model_config(model_name)
                if model_config.get("use_flash_attention", False):
                    print(f"Disabling flash attention for {model_name}")
                    model_config["use_flash_attention"] = False

            # Create model
            model = create_model(model_name, config)

            # Generate response
            start_time = time.time()
            response = model.generate(prompt)
            end_time = time.time()

            # Record results
            results.append({
                "model": model_name,
                "prompt": prompt,
                "response": response,
                "execution_time": end_time - start_time
            })

            print(f"Response generated in {end_time - start_time:.2f} seconds")
            print(f"Response length: {len(response)} characters")

        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            results.append({
                "model": model_name,
                "prompt": prompt,
                "error": str(e)
            })

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comparison_{timestamp}.json"
    FileUtils.write_json(results, output_file)

    # Print summary
    print(f"\nComparison completed. Results saved to {output_file}")
    print("\nSummary:")
    for result in results:
        if "error" in result:
            print(f"  {result['model']}: Error - {result['error']}")
        else:
            print(f"  {result['model']}: {result['execution_time']:.2f} seconds, {len(result['response'])} chars")

    return 0


if __name__ == "__main__":
    sys.exit(main())
