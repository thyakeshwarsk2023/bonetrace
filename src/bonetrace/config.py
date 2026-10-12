"""Configuration management for BoneTrace.

Handles loading, validating, and resolving paths from YAML configuration files
with support for environment variable and CLI argument overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def find_project_root(anchor_file: str = "pyproject.toml") -> Path:
    """Find the project root by searching upwards for an anchor file.

    Falls back to current working directory if anchor is not found.
    """
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / anchor_file).exists():
            return parent
    return Path.cwd()


@dataclass
class ProjectConfig:
    name: str = "bonetrace"
    seed: int = 42
    device: str = "cuda"


@dataclass
class PathsConfig:
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs")
    checkpoints_dir: Path = Path("outputs/checkpoints")
    logs_dir: Path = Path("outputs/logs")
    explanations_dir: Path = Path("outputs/explanations")
    evaluation_dir: Path = Path("outputs/evaluation")

    def resolve(self, base_dir: Path) -> PathsConfig:
        """Resolve all paths relative to base_dir if they are relative."""
        return PathsConfig(
            raw_data_dir=(base_dir / self.raw_data_dir).resolve() if not self.raw_data_dir.is_absolute() else self.raw_data_dir,
            processed_data_dir=(base_dir / self.processed_data_dir).resolve() if not self.processed_data_dir.is_absolute() else self.processed_data_dir,
            output_dir=(base_dir / self.output_dir).resolve() if not self.output_dir.is_absolute() else self.output_dir,
            checkpoints_dir=(base_dir / self.checkpoints_dir).resolve() if not self.checkpoints_dir.is_absolute() else self.checkpoints_dir,
            logs_dir=(base_dir / self.logs_dir).resolve() if not self.logs_dir.is_absolute() else self.logs_dir,
            explanations_dir=(base_dir / self.explanations_dir).resolve() if not self.explanations_dir.is_absolute() else self.explanations_dir,
            evaluation_dir=(base_dir / self.evaluation_dir).resolve() if not self.evaluation_dir.is_absolute() else self.evaluation_dir,
        )

    def ensure_output_directories(self) -> None:
        """Ensure all output directories exist on disk."""
        for path in [
            self.output_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.explanations_dir,
            self.evaluation_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = "bonetrace.log"


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    raw_config: Dict[str, Any] = field(default_factory=dict)
    project_root: Path = field(default_factory=find_project_root)

    def get_log_file_path(self) -> Optional[Path]:
        """Return the full resolved path to the log file if file logging is enabled."""
        if not self.logging.log_to_file:
            return None
        return self.paths.logs_dir / self.logging.log_filename


def load_config(
    config_path: Optional[Path | str] = None,
    project_root: Optional[Path | str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Load configuration from a YAML file, environment variables, and explicit overrides.

    Args:
        config_path: Path to YAML config file. If None, checks BONETRACE_CONFIG or defaults to
            '<project_root>/configs/default.yaml'.
        project_root: Root directory of the repository. Defaults to auto-detected root.
        overrides: Optional dictionary of path/setting overrides.

    Returns:
        Populated AppConfig instance with resolved paths.
    """
    root = Path(project_root).resolve() if project_root else find_project_root()

    if config_path is None:
        env_config = os.environ.get("BONETRACE_CONFIG")
        if env_config:
            config_file = Path(env_config)
        else:
            config_file = root / "configs" / "default.yaml"
    else:
        config_file = Path(config_path)

    if not config_file.is_absolute():
        config_file = (root / config_file).resolve()

    raw_data: Dict[str, Any] = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

    # Extract sections
    project_dict = raw_data.get("project", {})
    paths_dict = raw_data.get("paths", {})
    logging_dict = raw_data.get("logging", {})

    # Apply environment variable overrides
    if "BONETRACE_RAW_DATA_DIR" in os.environ:
        paths_dict["raw_data_dir"] = os.environ["BONETRACE_RAW_DATA_DIR"]
    if "BONETRACE_OUTPUT_DIR" in os.environ:
        paths_dict["output_dir"] = os.environ["BONETRACE_OUTPUT_DIR"]
    if "BONETRACE_LOG_LEVEL" in os.environ:
        logging_dict["level"] = os.environ["BONETRACE_LOG_LEVEL"]

    # Apply explicit overrides
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                if k in ("raw_data_dir", "processed_data_dir", "output_dir"):
                    paths_dict[k] = v
                elif k in ("log_level", "level"):
                    logging_dict["level"] = v
                elif k in ("seed", "device"):
                    project_dict[k] = v
                elif k in ("model_type", "batch_size", "dry_run", "num_epochs", "learning_rate"):
                    if "training" not in raw_data:
                        raw_data["training"] = {}
                    raw_data["training"][k] = v
                elif k in ("allow_untrained",):
                    if "evaluation" not in raw_data:
                        raw_data["evaluation"] = {}
                    raw_data["evaluation"][k] = v

    # Build objects
    project_cfg = ProjectConfig(
        name=project_dict.get("name", "bonetrace"),
        seed=int(project_dict.get("seed", 42)),
        device=str(project_dict.get("device", "cuda")),
    )

    base_output_dir = Path(paths_dict.get("output_dir", "outputs"))
    paths_cfg = PathsConfig(
        raw_data_dir=Path(paths_dict.get("raw_data_dir", "data/raw")),
        processed_data_dir=Path(paths_dict.get("processed_data_dir", "data/processed")),
        output_dir=base_output_dir,
        checkpoints_dir=Path(paths_dict.get("checkpoints_dir", base_output_dir / "checkpoints")),
        logs_dir=Path(paths_dict.get("logs_dir", base_output_dir / "logs")),
        explanations_dir=Path(paths_dict.get("explanations_dir", base_output_dir / "explanations")),
        evaluation_dir=Path(paths_dict.get("evaluation_dir", base_output_dir / "evaluation")),
    ).resolve(root)

    logging_cfg = LoggingConfig(
        level=str(logging_dict.get("level", "INFO")).upper(),
        log_to_file=bool(logging_dict.get("log_to_file", True)),
        log_filename=str(logging_dict.get("log_filename", "bonetrace.log")),
    )

    return AppConfig(
        project=project_cfg,
        paths=paths_cfg,
        logging=logging_cfg,
        raw_config=raw_data,
        project_root=root,
    )
