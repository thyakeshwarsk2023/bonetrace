"""Evaluation metrics and benchmarking for BoneTrace."""

from bonetrace.evaluation.metrics import (
    compute_classification_metrics,
    compute_expected_calibration_error,
    compute_explanation_stability,
    energy_inside_boxes,
    heatmap_box_iou,
    pointing_game_hit,
    run_evaluation,
)

__all__ = [
    "compute_classification_metrics",
    "compute_expected_calibration_error",
    "compute_explanation_stability",
    "energy_inside_boxes",
    "heatmap_box_iou",
    "pointing_game_hit",
    "run_evaluation",
]
