# src/utils/logging.py

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional


def setup_logging(log_dir: str = "logs", log_level: str = "INFO",
                  log_to_console: bool = True, log_to_file: bool = True,
                  max_file_size_mb: int = 10, max_backup_count: int = 5) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        max_file_size_mb: Maximum log file size in MB
        max_backup_count: Maximum number of backup log files

    Returns:
        Root logger instance configured with appropriate handlers
    """
    # Create log directory if it doesn't exist
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Get numeric logging level
    try:
        numeric_level = getattr(logging, log_level.upper())
    except (AttributeError, TypeError):
        print(f"Invalid log level: {log_level}, using INFO")
        numeric_level = logging.INFO

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=max_backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log the logging setup
    root_logger.info(f"Logging initialized at level {log_level}")
    if log_to_file:
        root_logger.info(f"Logging to file: {log_file}")

    return root_logger


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a component.

    Args:
        name: Name of the component
        log_level: Optional specific log level for this logger

    Returns:
        Logger instance for the component
    """
    logger = logging.getLogger(f"ai_system.{name}")

    # Set specific log level if provided
    if log_level:
        try:
            numeric_level = getattr(logging, log_level.upper())
            logger.setLevel(numeric_level)
        except (AttributeError, TypeError):
            pass  # Keep the parent logger's level

    return logger


def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """
    Log an exception with context.

    Args:
        logger: Logger to use
        exc: Exception to log
        context: Context information
    """
    import traceback

    if context:
        logger.error(f"Exception in {context}: {type(exc).__name__}: {exc}")
    else:
        logger.error(f"Exception: {type(exc).__name__}: {exc}")

    # Log the full traceback at debug level
    logger.debug(f"Traceback:\n{''.join(traceback.format_tb(exc.__traceback__))}")