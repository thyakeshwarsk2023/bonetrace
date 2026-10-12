"""Unit tests for classification, localization, and stability metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bonetrace.config import load_config
from bonetrace.evaluation.metrics import (
    compute_attribution_occlusion_faithfulness,
    compute_classification_metrics,
    compute_expected_calibration_error,
    compute_explanation_stability,
    energy_inside_boxes,
    heatmap_box_iou,
    pointing_game_hit,
    run_evaluation,
)
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.models.baseline import BaselineClassifier


def test_compute_classification_metrics_perfect_classifier():
    """Verify classification metrics on separable predictions."""
    y_true = [1, 0, 1, 0, 1, 0]
    y_prob = [0.95, 0.05, 0.90, 0.10, 0.85, 0.15]

    metrics = compute_classification_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["tp"] == 3.0
    assert metrics["tn"] == 3.0
    assert metrics["fp"] == 0.0
    assert metrics["fn"] == 0.0


def test_compute_classification_metrics_single_class_edge_case():
    """Verify robust fallback when only one class is present in true labels."""
    y_true = [0, 0, 0, 0]
    y_prob = [0.1, 0.2, 0.3, 0.4]

    metrics = compute_classification_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == 0.5  # fallback
    assert metrics["accuracy"] == 1.0
    assert metrics["tn"] == 4.0


def test_pointing_game_hit():
    """Verify Pointing Game correctly classifies peak hits and misses."""
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[50, 50] = 1.0  # peak at (y=50, x=50)

    # Box encompassing peak: [xmin, ymin, xmax, ymax]
    hit_boxes = [[40.0, 40.0, 60.0, 60.0]]
    assert pointing_game_hit(heatmap, hit_boxes) == 1.0

    # Box missing peak
    miss_boxes = [[10.0, 10.0, 30.0, 30.0]]
    assert pointing_game_hit(heatmap, miss_boxes) == 0.0

    # Empty boxes
    assert pointing_game_hit(heatmap, []) == 0.0


def test_energy_inside_boxes():
    """Verify E_in calculates energy proportion correctly."""
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[50, 50] = 1.0
    boxes = [[40.0, 40.0, 60.0, 60.0]]

    e_in = energy_inside_boxes(heatmap, boxes)
    assert np.isclose(e_in, 1.0, atol=1e-5)

    # All energy outside box
    outside_boxes = [[10.0, 10.0, 20.0, 20.0]]
    assert energy_inside_boxes(heatmap, outside_boxes) == 0.0

    # Empty boxes
    assert energy_inside_boxes(heatmap, []) == 0.0


def test_heatmap_box_iou():
    """Verify thresholded heatmap Intersection over Union against bounding box."""
    heatmap = np.zeros((100, 100), dtype=np.float32)
    # 20x20 activation
    heatmap[40:60, 40:60] = 1.0
    # Matching 20x20 box
    boxes = [[40.0, 40.0, 60.0, 60.0]]

    iou = heatmap_box_iou(heatmap, boxes, threshold_ratio=0.5)
    assert np.isclose(iou, 1.0, atol=1e-3)

    # Non-overlapping box
    other_boxes = [[0.0, 0.0, 10.0, 10.0]]
    assert heatmap_box_iou(heatmap, other_boxes) == 0.0


def test_compute_expected_calibration_error():
    """Verify calibration calculation."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])
    ece = compute_expected_calibration_error(y_true, y_prob)
    assert 0.0 <= ece <= 1.0


def test_compute_explanation_stability():
    """Verify explanation stability calculates correlation under small noise."""
    model = BaselineClassifier(pretrained=False)
    gradcam = GradCAM(model=model, target_size=(512, 512))

    x = torch.randn(3, 512, 512)
    stability = compute_explanation_stability(gradcam, x, noise_std=0.01)

    assert -1.0 <= stability <= 1.0
    # Small noise on smooth conv layers typically produces positive stability
    assert stability > 0.0
    gradcam.remove_hooks()


def test_compute_attribution_occlusion_faithfulness():
    """Verify attribution-guided region removal against equal-area control removal."""
    model = BaselineClassifier(pretrained=False)
    gradcam = GradCAM(model=model, target_size=(512, 512))

    x = torch.randn(3, 512, 512)
    faith = compute_attribution_occlusion_faithfulness(
        model=model,
        gradcam=gradcam,
        image_tensor=x,
        top_k_ratio=0.10,
    )

    assert "orig_logit" in faith
    assert "delta_logit_attributed" in faith
    assert "delta_logit_control" in faith
    assert "faithfulness_delta" in faith
    assert "is_faithful" in faith
    assert np.isclose(faith["faithfulness_delta"], faith["delta_logit_attributed"] - faith["delta_logit_control"], atol=1e-5)
    assert faith["is_faithful"] in (0.0, 1.0)
    gradcam.remove_hooks()


def test_run_evaluation_fails_when_checkpoints_missing(tmp_path):
    """Verify run_evaluation raises FileNotFoundError when checkpoints are absent and allow_untrained=False."""
    cfg = load_config(overrides={
        "output_dir": str(tmp_path / "outputs"),
    })
    # Points to non-existent checkpoints in tmp_path
    with pytest.raises(FileNotFoundError, match="trained model checkpoint"):
        run_evaluation(cfg, allow_untrained=False)

