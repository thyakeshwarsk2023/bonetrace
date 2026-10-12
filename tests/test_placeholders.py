"""Tests verifying package metadata and module exports."""

from __future__ import annotations

import bonetrace
from bonetrace.config import load_config
from bonetrace.data.dataset import FracAtlasDataset, export_exclusion_report
from bonetrace.evaluation.metrics import compute_classification_metrics, pointing_game_hit
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.models.factory import build_model
from bonetrace.training.trainer import Trainer


def test_package_metadata():
    assert bonetrace.__version__ == "0.1.0"


def test_modules_exported_and_callable():
    cfg = load_config()
    assert cfg is not None
    assert callable(FracAtlasDataset)
    assert callable(build_model)
    assert callable(Trainer)
    assert callable(GradCAM)
    assert callable(compute_classification_metrics)
    assert callable(pointing_game_hit)
    assert callable(export_exclusion_report)
