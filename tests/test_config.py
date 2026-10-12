"""Tests for configuration loading and path resolution."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from bonetrace.config import AppConfig, find_project_root, load_config


def test_find_project_root():
    root = find_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_load_default_config():
    root = find_project_root()
    config_file = root / "configs" / "default.yaml"
    assert config_file.exists(), "Default config file must exist"

    cfg = load_config(config_path=config_file)
    assert isinstance(cfg, AppConfig)
    assert cfg.project.name == "bonetrace"
    assert cfg.project.seed == 42
    assert cfg.paths.raw_data_dir == (root / "data" / "raw").resolve()
    assert cfg.paths.output_dir == (root / "outputs").resolve()
    assert cfg.logging.level == "INFO"


def test_config_overrides():
    root = find_project_root()
    custom_raw = "data/custom_raw"
    custom_output = "custom_outputs"

    cfg = load_config(
        overrides={
            "raw_data_dir": custom_raw,
            "output_dir": custom_output,
            "log_level": "DEBUG",
        }
    )
    assert cfg.paths.raw_data_dir == (root / custom_raw).resolve()
    assert cfg.paths.output_dir == (root / custom_output).resolve()
    assert cfg.logging.level == "DEBUG"


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("BONETRACE_RAW_DATA_DIR", "env_data/raw")
    monkeypatch.setenv("BONETRACE_OUTPUT_DIR", "env_outputs")
    monkeypatch.setenv("BONETRACE_LOG_LEVEL", "WARNING")

    root = find_project_root()
    cfg = load_config()

    assert cfg.paths.raw_data_dir == (root / "env_data" / "raw").resolve()
    assert cfg.paths.output_dir == (root / "env_outputs").resolve()
    assert cfg.logging.level == "WARNING"


def test_ensure_output_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = load_config(
            overrides={
                "output_dir": str(tmp_path / "test_outputs"),
            }
        )
        cfg.paths.ensure_output_directories()

        assert cfg.paths.output_dir.exists()
        assert cfg.paths.checkpoints_dir.exists()
        assert cfg.paths.logs_dir.exists()
        assert cfg.paths.explanations_dir.exists()
        assert cfg.paths.evaluation_dir.exists()
