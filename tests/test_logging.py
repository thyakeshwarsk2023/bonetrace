"""Tests for logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile

from bonetrace.logging_utils import get_logger, setup_logging


def test_setup_logging_console():
    logger = setup_logging(level="DEBUG", logger_name="test_console_logger")
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_setup_logging_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "sub" / "test.log"
        logger = setup_logging(level="INFO", log_file=log_file, logger_name="test_file_logger")

        try:
            logger.info("Test log message for BoneTrace")

            assert log_file.exists(), "Log file should be created"
            content = log_file.read_text(encoding="utf-8")
            assert "Test log message for BoneTrace" in content
            assert "[INFO]" in content
        finally:
            # On Windows, open FileHandler locks the file; explicitly close handlers
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


def test_get_logger():
    logger = get_logger("custom_module")
    assert logger.name == "bonetrace.custom_module"
