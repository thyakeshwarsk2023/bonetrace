"""Tests for CLI entry points and stage dispatching."""

from __future__ import annotations

import pytest

from bonetrace.cli import build_parser, main


def test_build_parser_subcommands():
    parser = build_parser()
    for subcmd in ["audit", "preprocess", "train", "evaluate"]:
        parsed = parser.parse_args([subcmd, "--log-level", "DEBUG"])
        assert parsed.command == subcmd
        assert parsed.log_level == "DEBUG"


def test_cli_no_args(capsys):
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.err.lower() or "usage:" in captured.out.lower()


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_audit_stage_succeeds():
    # Audit stage is fully implemented and exits with code 0
    exit_code = main(["audit"])
    assert exit_code == 0


def test_cli_preprocess_stage_succeeds():
    # Preprocess stage is fully implemented and exits with code 0
    exit_code = main(["preprocess"])
    assert exit_code == 0


def test_cli_train_dry_run_succeeds():
    # Train stage dry-run initializes successfully and exits with code 0
    exit_code = main(["train", "--dry-run", "--model-type", "baseline"])
    assert exit_code == 0


def test_cli_train_region_supervised_dry_run_succeeds():
    # Region-supervised dry run initializes successfully and exits with code 0
    exit_code = main(["train", "--dry-run", "--model-type", "region_supervised"])
    assert exit_code == 0


def test_cli_evaluate_fails_when_checkpoints_missing(tmp_path):
    # Evaluation must fail cleanly when trained checkpoints are missing
    exit_code = main(["evaluate", "--output-dir", str(tmp_path / "outputs")])
    assert exit_code == 2


def test_cli_unknown_command_returns_error(capsys):
    with pytest.raises(SystemExit):
        main(["unknown_stage"])

