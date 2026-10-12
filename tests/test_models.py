"""Unit tests for BoneTrace model architectures and factory."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bonetrace.config import AppConfig, ProjectConfig
from bonetrace.models.baseline import BaselineClassifier
from bonetrace.models.factory import build_model
from bonetrace.models.region_guided import RegionSupervisedModel


def test_baseline_classifier_forward_shape():
    """Verify BaselineClassifier produces binary logits of shape (B, 1)."""
    model = BaselineClassifier(pretrained=False, num_classes=1)
    model.eval()

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape == (2, 1)
    assert model.model_type == "baseline"


def test_region_supervised_forward_shapes():
    """Verify RegionSupervisedModel produces classification logits and spatial mask logits."""
    model = RegionSupervisedModel(pretrained=False, num_classes=1, target_size=(512, 512))
    model.eval()

    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, dict)
    assert "logits" in out
    assert "mask_logits" in out
    assert out["logits"].shape == (2, 1)
    assert out["mask_logits"].shape == (2, 1, 512, 512)
    assert model.model_type == "region_supervised"


def test_models_share_equivalent_cam_target_layer():
    """Verify both baseline and region-supervised models return equivalent layer4 target layers."""
    m_base = BaselineClassifier(pretrained=False)
    m_reg = RegionSupervisedModel(pretrained=False)

    target_b = m_base.get_cam_target_layer()
    target_r = m_reg.get_cam_target_layer()

    # Both must be the last BasicBlock in layer4
    assert type(target_b) == type(target_r)
    assert isinstance(target_b, nn.Module)
    assert hasattr(target_b, "conv2")


def test_build_model_factory_types():
    """Verify build_model instantiates correct model variants."""
    m_base = build_model(model_type="baseline")
    assert isinstance(m_base, BaselineClassifier)

    m_reg = build_model(model_type="region_supervised")
    assert isinstance(m_reg, RegionSupervisedModel)


def test_build_model_from_config():
    """Verify build_model correctly consumes AppConfig."""
    cfg = AppConfig(
        project=ProjectConfig(seed=123),
        raw_config={"training": {"model_type": "region_supervised"}},
    )
    model = build_model(config=cfg)
    assert isinstance(model, RegionSupervisedModel)


def test_build_model_invalid_type_raises():
    """Verify build_model raises ValueError for unrecognized architecture types."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_model(model_type="invalid_architecture")


def test_deterministic_initialization():
    """Verify identical random seeds produce identical initialized weights."""
    m1 = build_model(model_type="baseline")
    m2 = build_model(model_type="baseline")

    w1 = m1.backbone.conv1.weight.detach()
    w2 = m2.backbone.conv1.weight.detach()
    assert torch.equal(w1, w2)
