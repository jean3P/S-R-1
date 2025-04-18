# utils/logging_utils.py
import os
import logging
from pathlib import Path
import datetime
from typing import Optional


def setup_logging(config, log_file: Optional[str] = None):
    """
    Set up logging configuration.

    Args:
        config: Configuration object.
        log_file: Optional specific log file name.
    """
    log_dir = Path(config["logging"]["log_dir"])
    log_level_str = config["logging"]["log_level"]
    log_level = getattr(logging, log_level_str.upper())

    # Create log directory if it doesn't exist
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Generate log file name if not provided
    if not log_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"swereflect_{timestamp}.log"

    log_path = log_dir / log_file

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    # Log initial message
    logging.info(f"Logging initialized at {log_path}")
