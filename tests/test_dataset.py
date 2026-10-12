"""Unit tests for FracAtlas PyTorch Dataset pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import pytest
import torch

from bonetrace.data.dataset import (
    DISPUTED_CLONE_CANONICAL_IDS,
    FracAtlasDataset,
    export_exclusion_report,
    fracatlas_collate_fn,
    load_image_safely,
    rasterize_boxes_to_mask,
    safe_truncated_image_loading,
)


@pytest.fixture
def synthetic_image_dir(tmp_path):
    """Create a temporary directory with synthetic test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)

    # 1. Grayscale image
    img_gray = Image.new("L", (100, 200), color=128)
    gray_path = img_dir / "test_gray.jpg"
    img_gray.save(gray_path)

    # 2. RGB image
    img_rgb = Image.new("RGB", (300, 150), color=(200, 100, 50))
    rgb_path = img_dir / "test_rgb.jpg"
    img_rgb.save(rgb_path)

    return {
        "gray_path": str(gray_path),
        "rgb_path": str(rgb_path),
        "dir": tmp_path,
    }


@pytest.fixture
def synthetic_manifest_df(synthetic_image_dir):
    """Create a synthetic DataFrame matching canonical FracAtlas schema."""
    rows = [
        {
            "canonical_id": "CANON_0001",
            "resolved_primary_path": synthetic_image_dir["gray_path"],
            "fractured_label": 0,
            "split": "train",
            "letterbox_target_boxes": "[]",
            "has_coco_ann": False,
        },
        {
            "canonical_id": "CANON_0002",
            "resolved_primary_path": synthetic_image_dir["rgb_path"],
            "fractured_label": 1,
            "split": "train",
            "letterbox_target_boxes": "[[100, 150, 250, 300]]",
            "has_coco_ann": True,
        },
        {
            "canonical_id": "CANON_3339",  # Disputed clone
            "resolved_primary_path": synthetic_image_dir["gray_path"],
            "fractured_label": 0,
            "split": "train",
            "letterbox_target_boxes": "[]",
            "has_coco_ann": False,
        },
        {
            "canonical_id": "CANON_0382",  # Retained candidate
            "resolved_primary_path": synthetic_image_dir["rgb_path"],
            "fractured_label": 0,
            "split": "train",
            "letterbox_target_boxes": "[]",
            "has_coco_ann": False,
        },
    ]
    return pd.DataFrame(rows)


def test_safe_truncated_loading_restores_global_state():
    """Verify that safe_truncated_image_loading context manager cleanly resets Pillow state."""
    initial_state = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    with safe_truncated_image_loading():
        assert ImageFile.LOAD_TRUNCATED_IMAGES is True

    assert ImageFile.LOAD_TRUNCATED_IMAGES is False
    ImageFile.LOAD_TRUNCATED_IMAGES = initial_state


def test_load_image_safely_rgb_conversion(synthetic_image_dir):
    """Verify that images in 'L' or other modes are reliably converted to 'RGB'."""
    gray_img = load_image_safely(synthetic_image_dir["gray_path"])
    assert gray_img.mode == "RGB"

    rgb_img = load_image_safely(synthetic_image_dir["rgb_path"])
    assert rgb_img.mode == "RGB"


def test_rasterize_boxes_to_mask():
    """Verify bounding box rasterization onto binary mask tensor."""
    # Empty boxes return all zeros
    empty_mask = rasterize_boxes_to_mask([], canvas_size=(100, 100))
    assert empty_mask.shape == (1, 100, 100)
    assert empty_mask.sum().item() == 0.0

    # Box [20, 30, 40, 50]: width 20, height 20 -> area 400
    boxes = [[20.0, 30.0, 40.0, 50.0]]
    mask = rasterize_boxes_to_mask(boxes, canvas_size=(100, 100))
    assert mask.shape == (1, 100, 100)
    assert mask.max().item() == 1.0
    assert mask[0, 35, 25].item() == 1.0  # inside box
    assert mask[0, 5, 5].item() == 0.0    # outside box


def test_disputed_clone_exclusion(synthetic_manifest_df):
    """Verify that disputed clone group CANON_3339 is excluded by default, but kept when False."""
    ds_excluded = FracAtlasDataset(manifest=synthetic_manifest_df, exclude_conflicts=True)
    cids_excluded = [r["canonical_id"] for r in ds_excluded.records]
    assert "CANON_3339" not in cids_excluded
    assert "CANON_0382" in cids_excluded  # Retained candidate is preserved!
    assert len(ds_excluded) == 3

    # Sensitivity cohort keeps all records
    ds_all = FracAtlasDataset(manifest=synthetic_manifest_df, exclude_conflicts=False)
    cids_all = [r["canonical_id"] for r in ds_all.records]
    assert "CANON_3339" in cids_all
    assert len(ds_all) == 4


def test_dataset_item_structure_and_shapes(synthetic_manifest_df):
    """Verify __getitem__ returns correct tensor types, keys, and shapes."""
    ds = FracAtlasDataset(manifest=synthetic_manifest_df, exclude_conflicts=True, target_size=(512, 512))
    # Negative sample
    item_neg = ds[0]
    assert item_neg["image"].shape == (3, 512, 512)
    assert item_neg["label"].item() == 0.0
    assert item_neg["mask"].shape == (1, 512, 512)
    assert item_neg["mask"].sum().item() == 0.0
    assert item_neg["has_annotation"].item() is False
    assert item_neg["boxes"].shape == (0, 4)

    # Positive sample
    item_pos = ds[1]
    assert item_pos["image"].shape == (3, 512, 512)
    assert item_pos["label"].item() == 1.0
    assert item_pos["mask"].shape == (1, 512, 512)
    assert item_pos["mask"].sum().item() > 0.0
    assert item_pos["has_annotation"].item() is True
    assert item_pos["boxes"].shape == (1, 4)


def test_class_counts_and_pos_weight(synthetic_manifest_df):
    """Verify positive class weight calculation for BCEWithLogitsLoss."""
    ds = FracAtlasDataset(manifest=synthetic_manifest_df, exclude_conflicts=True)
    counts = ds.get_class_counts()
    # 2 negative, 1 positive in ds (after CANON_3339 exclusion)
    assert counts[0] == 2
    assert counts[1] == 1
    assert ds.get_pos_weight() == 2.0


def test_missing_required_columns_raises():
    """Verify ValueError is raised if essential manifest columns are absent."""
    invalid_df = pd.DataFrame({"random_col": [1, 2]})
    with pytest.raises(ValueError, match="missing required columns"):
        FracAtlasDataset(manifest=invalid_df)


def test_invalid_label_values_raises(synthetic_image_dir):
    """Verify ValueError is raised if fractured_label contains non-binary values."""
    bad_df = pd.DataFrame({
        "canonical_id": ["CANON_0001"],
        "resolved_primary_path": [synthetic_image_dir["gray_path"]],
        "fractured_label": [2],  # Non-binary
    })
    with pytest.raises(ValueError, match="non-binary"):
        FracAtlasDataset(manifest=bad_df)


def test_collate_fn_handles_variable_boxes(synthetic_manifest_df):
    """Verify that fracatlas_collate_fn batches variable bounding box counts without stacking errors."""
    ds = FracAtlasDataset(manifest=synthetic_manifest_df, exclude_conflicts=True)
    batch = [ds[0], ds[1]]  # ds[0] has 0 boxes, ds[1] has 1 box
    collated = fracatlas_collate_fn(batch)

    assert collated["image"].shape == (2, 3, 512, 512)
    assert collated["label"].shape == (2, 1)
    assert collated["mask"].shape == (2, 1, 512, 512)
    assert collated["has_annotation"].shape == (2,)
    assert len(collated["boxes"]) == 2
    assert collated["boxes"][0].shape == (0, 4)
    assert collated["boxes"][1].shape == (1, 4)


def test_export_exclusion_report(tmp_path):
    """Verify generation of auditable JSON and Markdown exclusion reports."""
    dummy_manifest = pd.DataFrame([
        {"canonical_id": "CANON_0206", "representative_image_id": "IMG1.jpg", "all_filenames": "IMG1.jpg", "split": "test", "fractured_label": 1},
        {"canonical_id": "CANON_0001", "representative_image_id": "IMG2.jpg", "all_filenames": "IMG2.jpg", "split": "train", "fractured_label": 0},
    ])
    csv_file = tmp_path / "test_manifest.csv"
    dummy_manifest.to_csv(csv_file, index=False)

    report = export_exclusion_report(csv_file, tmp_path)
    assert report["total_canonical_records_before"] == 2
    assert report["total_canonical_records_after"] == 1
    assert (tmp_path / "exclusions_report.json").exists()
    assert (tmp_path / "exclusions_report.md").exists()


def test_frozen_split_manifests_exclusion_counts():
    """Verify that the primary benchmark protocol yields 2,842 / 612 / 610 across real split CSVs."""
    proc_dir = Path("data/processed")
    if not (proc_dir / "train.csv").is_file():
        pytest.skip("Processed split CSVs not available")

    train_ds = FracAtlasDataset(proc_dir / "train.csv", split="train", exclude_conflicts=True)
    val_ds = FracAtlasDataset(proc_dir / "val.csv", split="val", exclude_conflicts=True)
    test_ds = FracAtlasDataset(proc_dir / "test.csv", split="test", exclude_conflicts=True)

    # 1. Exact primary benchmark counts
    assert len(train_ds) == 2842
    assert len(val_ds) == 612
    assert len(test_ds) == 610

    # 2. Verify exact canonical IDs excluded per split
    assert [r["canonical_id"] for r in train_ds.excluded_records] == ["CANON_3339"]
    assert [r["canonical_id"] for r in val_ds.excluded_records] == ["CANON_1668"]
    assert sorted([r["canonical_id"] for r in test_ds.excluded_records]) == ["CANON_0206", "CANON_0678"]

    # 3. Retained distinct-radiograph pairs are preserved
    train_cids = {r["canonical_id"] for r in train_ds.records}
    assert "CANON_0382" in train_cids
    assert "CANON_3487" in train_cids
    assert "CANON_1242" in train_cids
    assert "CANON_3205" in train_cids
