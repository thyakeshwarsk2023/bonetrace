"""Command Line Interface (CLI) entry point for BoneTrace.

Provides subcommands for audit, preprocessing, training, and evaluation pipelines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from bonetrace import __version__
from bonetrace.config import AppConfig, load_config
from bonetrace.logging_utils import get_logger, setup_logging


logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser and subcommands."""
    parser = argparse.ArgumentParser(
        prog="bonetrace",
        description="BoneTrace: Region Supervision vs Explanation Reliability",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Shared parent parser for common pipeline arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-c", "--config",
        dest="config_path",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    common_parser.add_argument(
        "--raw-data-dir",
        dest="raw_data_dir",
        type=str,
        default=None,
        help="Override raw data directory path",
    )
    common_parser.add_argument(
        "--processed-data-dir",
        dest="processed_data_dir",
        type=str,
        default=None,
        help="Override processed data directory path",
    )
    common_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Override base output directory path",
    )
    common_parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override logging verbosity level",
    )
    common_parser.add_argument(
        "--device",
        dest="device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Computation device override (cuda, cpu)",
    )
    common_parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Random seed override for deterministic execution",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Pipeline Commands",
        description="Available pipeline execution stages",
        metavar="<command>",
    )

    # Subcommand: audit
    subparsers.add_parser(
        "audit",
        parents=[common_parser],
        help="Audit raw dataset directory structure, annotations, and file integrity",
        description="Audit raw FracAtlas files and verify directory layout before preprocessing.",
    )

    # Subcommand: preprocess
    subparsers.add_parser(
        "preprocess",
        parents=[common_parser],
        help="Preprocess raw images and prepare train/val/test splits",
        description="Normalize images, generate split manifests, and prepare bounding box data.",
    )

    # Subcommand: train
    train_parser = subparsers.add_parser(
        "train",
        parents=[common_parser],
        help="Train baseline or region-supervised bone fracture models",
        description="Execute model training with checkpointing and loss tracking.",
    )
    train_parser.add_argument(
        "--model-type",
        dest="model_type",
        choices=["baseline", "region_supervised"],
        default=None,
        help="Select model variant ('baseline' or 'region_supervised')",
    )
    train_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Run fast initialization check without full epoch iterations",
    )
    train_parser.add_argument(
        "--epochs",
        dest="num_epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    # Subcommand: evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        parents=[common_parser],
        help="Evaluate classification metrics and explanation reliability",
        description="Run evaluation benchmarks and generate explanation reliability metrics.",
    )
    eval_parser.add_argument(
        "--model-type",
        dest="model_type",
        choices=["baseline", "region_supervised"],
        default=None,
        help="Optional model filter for evaluation",
    )
    eval_parser.add_argument(
        "--allow-untrained",
        dest="allow_untrained",
        action="store_true",
        default=False,
        help="Allow evaluating randomly initialized weights if checkpoints do not exist (default: False; evaluation fails if checkpoints are missing)",
    )

    # Subcommand: explain
    explain_parser = subparsers.add_parser(
        "explain",
        parents=[common_parser],
        help="Generate Grad-CAM explanation heatmaps and diagnostic overlays",
        description="Extract and visualize Grad-CAM post-hoc explanations for test samples.",
    )
    explain_parser.add_argument(
        "--model-type",
        dest="model_type",
        choices=["baseline", "region_supervised"],
        default=None,
        help="Select model variant ('baseline' or 'region_supervised')",
    )
    explain_parser.add_argument(
        "--num-samples",
        dest="num_samples",
        type=int,
        default=20,
        help="Number of explanation samples to extract (default: 20)",
    )

    return parser


def run_stage(command: str, config: AppConfig) -> None:
    """Dispatch to the corresponding pipeline stage implementation."""
    if command == "audit":
        from bonetrace.data.audit import run_audit
        run_audit(config)
    elif command == "preprocess":
        from bonetrace.data.preprocess import run_preprocessing
        run_preprocessing(config)
    elif command == "train":
        from bonetrace.training.trainer import run_training
        run_training(config)
    elif command == "evaluate":
        from bonetrace.evaluation.metrics import run_evaluation
        raw_eval = config.raw_config.get("evaluation", {}) if hasattr(config, "raw_config") else {}
        allow_untrained = bool(raw_eval.get("allow_untrained", False))
        run_evaluation(config, allow_untrained=allow_untrained)
    elif command == "explain":
        from bonetrace.explainability.gradcam import compute_explanations
        compute_explanations(config)
    else:
        raise ValueError(f"Unknown command: '{command}'")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns exit code (0 for success, non-zero for error)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help(sys.stderr)
        return 0

    # Build overrides from CLI args
    overrides = {
        "raw_data_dir": args.raw_data_dir,
        "processed_data_dir": args.processed_data_dir,
        "output_dir": args.output_dir,
        "log_level": args.log_level,
        "device": args.device,
        "model_type": getattr(args, "model_type", None),
        "dry_run": getattr(args, "dry_run", False),
        "num_epochs": getattr(args, "num_epochs", None),
        "num_samples": getattr(args, "num_samples", None),
        "allow_untrained": getattr(args, "allow_untrained", False),
        "seed": getattr(args, "seed", None),
    }

    try:
        config = load_config(config_path=args.config_path, overrides=overrides)
        config.paths.ensure_output_directories()
        setup_logging(
            level=config.logging.level,
            log_file=config.get_log_file_path(),
        )

        logger.info("BoneTrace v%s - Executing stage: %s", __version__, args.command)
        logger.info("Project root: %s", config.project_root)
        logger.info("Raw data dir: %s", config.paths.raw_data_dir)
        logger.info("Output dir:   %s", config.paths.output_dir)

        run_stage(args.command, config)
        return 0
    except NotImplementedError as err:
        logger.warning("Stage '%s' is not yet implemented: %s", args.command, err)
        return 1
    except Exception as err:
        logger.error("Error executing stage '%s': %s", args.command, err, exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
