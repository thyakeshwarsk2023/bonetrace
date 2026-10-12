"""Unit tests for Grad-CAM post-hoc explanation extraction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bonetrace.explainability.gradcam import (
    GradCAM,
    render_heatmap_overlay,
)
from bonetrace.models.baseline import BaselineClassifier
from bonetrace.models.region_guided import RegionSupervisedModel


def test_gradcam_baseline_heatmap_shape_and_range():
    """Verify GradCAM generates 512x512 normalized heatmap for BaselineClassifier."""
    model = BaselineClassifier(pretrained=False)
    gradcam = GradCAM(model=model, target_size=(512, 512))

    x = torch.randn(1, 3, 512, 512)
    heatmap = gradcam.generate(x, class_idx=0)

    assert heatmap.shape == (512, 512)
    assert 0.0 <= heatmap.min() <= 1.0
    assert 0.0 <= heatmap.max() <= 1.0
    gradcam.remove_hooks()
    assert len(gradcam.hooks) == 0


def test_gradcam_region_supervised_heatmap_generation():
    """Verify GradCAM generates 512x512 normalized heatmap for RegionSupervisedModel."""
    model = RegionSupervisedModel(pretrained=False, target_size=(512, 512))
    gradcam = GradCAM(model=model, target_size=(512, 512))

    x = torch.randn(1, 3, 512, 512)
    heatmap = gradcam.generate(x, class_idx=0)

    assert heatmap.shape == (512, 512)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    gradcam.remove_hooks()


def test_render_heatmap_overlay():
    """Verify overlay blending returns HxWx3 uint8 image."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    heatmap = np.zeros((512, 512), dtype=np.float32)
    heatmap[200:300, 200:300] = 1.0

    overlay = render_heatmap_overlay(img, heatmap, alpha=0.5)
    assert overlay.shape == (512, 512, 3)
    assert overlay.dtype == np.uint8
    # Blended region should differ from neutral background
    assert not np.array_equal(overlay[250, 250], [128, 128, 128])
