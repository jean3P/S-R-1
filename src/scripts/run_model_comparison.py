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
