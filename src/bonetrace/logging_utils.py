"""Logging setup for BoneTrace.

Provides consistent console and file logging formatting across all CLI stages
and submodules.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path | str] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configure logging to console and optionally to a file.

    Args:
        level: Logging level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to write log output. Parent directories are created if needed.
        logger_name: Name of the logger to configure. Defaults to the root logger.

    Returns:
        The configured logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    # Close and remove existing handlers to avoid duplicates and Windows file lock leaks
    if logger.hasHandlers():
        for handler in list(logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger child within the 'bonetrace' namespace."""
    return logging.getLogger(f"bonetrace.{name}")
