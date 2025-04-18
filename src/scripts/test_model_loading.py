#!/usr/bin/env python3
# scripts/test_model_loading.py

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config import Config
from src.models import create_model
from src.utils.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument("--model", type=str, choices=["deepseek-32b", "qwen-32b", "qwq-32b", "qwen-coder-7b"],
                        default="qwen-32b", help="Model to test")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Test prompt")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    """Main function for testing model loading."""
    args = parse_args()
    
    # Load configuration
    config = Config()
    
    # Set up logging
    if args.debug:
        config["logging"]["log_level"] = "DEBUG"
    setup_logging(config)
    
    # Print model configuration
    model_config = config.get_model_config(args.model)
    print(f"\nModel configuration for {args.model}:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Create model
    print(f"\nLoading model {args.model}...")
    try:
        model = create_model(args.model, config)
        print(f"Model loaded successfully: {model.model_name}")
        
        # Test generation
        print(f"\nGenerating response for prompt: '{args.prompt}'")
        response = model.generate(args.prompt)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
