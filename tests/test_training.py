"""Unit tests for BoneTrace training infrastructure and loss computation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from bonetrace.config import AppConfig, PathsConfig, ProjectConfig
from bonetrace.training.trainer import Trainer, set_seed


@pytest.fixture
def synthetic_trainer_env(tmp_path):
    """Create a minimal test configuration and paths for Trainer."""
    paths = PathsConfig(
        output_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "outputs/checkpoints",
        logs_dir=tmp_path / "outputs/logs",
    )
    cfg = AppConfig(
        project=ProjectConfig(seed=42, device="cpu"),
        paths=paths,
        raw_config={"training": {"batch_size": 2, "learning_rate": 1e-4, "aux_loss_weight": 1.0}},
    )
    return cfg


def test_set_seed_reproducibility():
    """Verify deterministic seed configuration across PyTorch."""
    set_seed(999)
    t1 = torch.randn(5)
    set_seed(999)
    t2 = torch.randn(5)
    assert torch.equal(t1, t2)


def test_trainer_loss_computation_baseline(synthetic_trainer_env):
    """Verify classification loss calculation for baseline model."""
    trainer = Trainer(synthetic_trainer_env, model_type="baseline")

    outputs = {"logits": torch.tensor([[0.5], [-0.5]])}
    batch = {
        "label": torch.tensor([[1.0], [0.0]]),
        "mask": torch.zeros((2, 1, 512, 512)),
        "has_annotation": torch.tensor([True, False]),
    }

    total_loss, loss_dict = trainer.compute_loss(outputs, batch)
    assert total_loss.item() > 0.0
    assert loss_dict["cls_loss"] == total_loss.item()
    assert loss_dict["aux_loss"] == 0.0


def test_trainer_loss_computation_region_supervised(synthetic_trainer_env):
    """Verify combined classification and auxiliary localization loss for region model."""
    trainer = Trainer(synthetic_trainer_env, model_type="region_supervised")

    outputs = {
        "logits": torch.tensor([[0.5], [-0.5]]),
        "mask_logits": torch.zeros((2, 1, 512, 512)),
    }
    batch = {
        "label": torch.tensor([[1.0], [0.0]]),
        "mask": torch.ones((2, 1, 512, 512)) * 0.5,
        "has_annotation": torch.tensor([True, False]),
    }

    total_loss, loss_dict = trainer.compute_loss(outputs, batch)
    assert total_loss.item() > 0.0
    assert loss_dict["cls_loss"] > 0.0
    assert loss_dict["aux_loss"] > 0.0
    import numpy as np
    assert np.isclose(loss_dict["total_loss"], loss_dict["cls_loss"] + loss_dict["aux_loss"])


def test_train_epoch_and_validate_synthetic_batch(synthetic_trainer_env):
    """Verify train_epoch and validate complete a forward/backward cycle."""
    trainer = Trainer(synthetic_trainer_env, model_type="baseline")

    batch = {
        "image": torch.randn(2, 3, 512, 512),
        "label": torch.tensor([[1.0], [0.0]]),
        "mask": torch.zeros((2, 1, 512, 512)),
        "has_annotation": torch.tensor([True, False]),
    }

    class DummyDataLoader:
        def __iter__(self):
            return iter([batch])
        def __len__(self):
            return 1

    train_metrics = trainer.train_epoch(DummyDataLoader())
    assert "loss" in train_metrics
    assert "cls_loss" in train_metrics

    val_metrics = trainer.validate(DummyDataLoader())
    assert "val_loss" in val_metrics
    assert "roc_auc" in val_metrics
    assert "f1" in val_metrics


def test_checkpoint_save_and_load(synthetic_trainer_env):
    """Verify model state and metrics can be saved and restored accurately."""
    trainer = Trainer(synthetic_trainer_env, model_type="baseline")

    metrics = {"roc_auc": 0.88, "f1": 0.75}
    saved_path = trainer.save_checkpoint(epoch=3, metrics=metrics, is_best=True)

    assert saved_path.exists()

    # Create new trainer and load
    new_trainer = Trainer(synthetic_trainer_env, model_type="baseline")
    checkpoint = new_trainer.load_checkpoint(saved_path)

    assert checkpoint["epoch"] == 3
    assert checkpoint["metrics"]["roc_auc"] == 0.88
    assert checkpoint["model_type"] == "baseline"
