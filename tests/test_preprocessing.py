"""Unit tests for Phase 2: Canonical Preprocessing, Transforms, and Leakage-Aware Splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bonetrace.data.preprocess import (
    LetterboxTransform,
    TransformParams,
    UnionFind,
    create_leakage_aware_splits,
)


# ==============================================================================
# Test LetterboxTransform: Images, Boxes, Masks & Invertibility
# ==============================================================================

def test_letterbox_aspect_ratio_and_padding():
    """Verify aspect-ratio preservation and neutral padding margins."""
    transform = LetterboxTransform(target_size=(512, 512))

    # Landscape image: 800 x 400 (aspect 2:1)
    # Target 512x512 -> scale = min(512/800, 512/400) = 0.64
    # scaled_w = 512, scaled_h = 256
    # pad_left = 0, pad_right = 0, pad_top = 128, pad_bottom = 128
    params_landscape = transform.get_params(orig_w=800, orig_h=400)
    assert params_landscape.scale == pytest.approx(0.64)
    assert params_landscape.scaled_w == 512
    assert params_landscape.scaled_h == 256
    assert params_landscape.pad_left == 0
    assert params_landscape.pad_right == 0
    assert params_landscape.pad_top == 128
    assert params_landscape.pad_bottom == 128

    # Portrait image: 300 x 600 (aspect 1:2)
    # Target 512x512 -> scale = min(512/300, 512/600) = 512/600 = 0.85333...
    # scaled_w = 256, scaled_h = 512
    params_portrait = transform.get_params(orig_w=300, orig_h=600)
    assert params_portrait.scaled_w == 256
    assert params_portrait.scaled_h == 512
    assert params_portrait.pad_top == 0
    assert params_portrait.pad_bottom == 0
    assert params_portrait.pad_left == 128
    assert params_portrait.pad_right == 128

    # Invalid zero/negative dimensions
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        transform.get_params(orig_w=0, orig_h=500)


def test_letterbox_image_and_mask_transform():
    """Verify PIL image and mask transformation to target canvas size."""
    transform = LetterboxTransform(target_size=(512, 512), fill_color=(0, 0, 0))

    # Create dummy RGB image (1000x500)
    img = Image.new("RGB", (1000, 500), color=(128, 128, 128))
    canvas, params = transform.apply_image(img)
    assert canvas.size == (512, 512)
    assert canvas.mode == "RGB"

    # Create binary mask (mode 'L')
    mask = Image.new("L", (1000, 500), color=0)
    # Fill a subregion with 255
    for y in range(100, 400):
        for x in range(200, 800):
            mask.putpixel((x, y), 255)

    transformed_mask = transform.apply_mask(mask, params)
    assert transformed_mask.size == (512, 512)
    assert transformed_mask.mode == "L"

    # The corners of canvas should be padded with 0
    assert transformed_mask.getpixel((0, 0)) == 0
    assert transformed_mask.getpixel((511, 511)) == 0


def test_letterbox_box_forward_and_inversion():
    """Verify bounding box forward transformation, normalization, and exact inversion."""
    transform = LetterboxTransform(target_size=(512, 512))
    orig_w, orig_h = 1200, 800
    params = transform.get_params(orig_w, orig_h)

    # Box [xmin, ymin, xmax, ymax]
    boxes = [
        (100.0, 150.0, 500.0, 600.0),
        (0.0, 0.0, 1200.0, 800.0),  # Full image bounds
    ]

    # Forward transform
    transformed_boxes = transform.apply_boxes(boxes, params)
    assert len(transformed_boxes) == 2

    # Invert back to original coordinates
    inverted_boxes = transform.invert_boxes(transformed_boxes, params)
    assert len(inverted_boxes) == 2

    for orig, inv in zip(boxes, inverted_boxes):
        assert inv[0] == pytest.approx(orig[0], abs=1e-4)
        assert inv[1] == pytest.approx(orig[1], abs=1e-4)
        assert inv[2] == pytest.approx(orig[2], abs=1e-4)
        assert inv[3] == pytest.approx(orig[3], abs=1e-4)

    # Normalized boxes should strictly lie within [0.0, 1.0]
    norm_boxes = transform.apply_boxes_normalized(boxes, params)
    for nb in norm_boxes:
        assert 0.0 <= nb[0] <= 1.0
        assert 0.0 <= nb[1] <= 1.0
        assert 0.0 <= nb[2] <= 1.0
        assert 0.0 <= nb[3] <= 1.0
        assert nb[0] <= nb[2]
        assert nb[1] <= nb[3]


# ==============================================================================
# Test UnionFind Disjoint Set Isolation
# ==============================================================================

def test_union_find_grouping():
    """Verify connected components and transitivity in duplicate grouping."""
    uf = UnionFind()
    uf.union("A", "B")
    uf.union("B", "C")
    uf.union("D", "E")

    assert uf.find("A") == uf.find("C")
    assert uf.find("A") == uf.find("B")
    assert uf.find("D") == uf.find("E")
    assert uf.find("A") != uf.find("D")

    groups = uf.get_groups()
    # A, B, C share a root
    root_abc = uf.find("A")
    assert set(groups[root_abc]) == {"A", "B", "C"}


# ==============================================================================
# Test Leakage-Aware Splitting
# ==============================================================================

@pytest.fixture
def synthetic_manifest_df() -> pd.DataFrame:
    """Create a realistic synthetic dataset manifest with duplicate groups and labels."""
    np.random.seed(123)
    n_samples = 400

    canonical_ids = [f"IMG_{i:04d}" for i in range(n_samples)]
    labels = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

    # Assign duplicate groups: first 60 samples belong to 20 clusters of size 3
    group_ids = []
    for i in range(n_samples):
        if i < 60:
            group_ids.append(f"grp_{i // 3:03d}")
        else:
            group_ids.append(f"grp_{i:03d}")

    return pd.DataFrame({
        "canonical_id": canonical_ids,
        "fractured_label": labels,
        "future_split_group_id": group_ids,
    })


def test_split_disjointness_and_group_isolation(synthetic_manifest_df):
    """Verify zero overlap between splits and strict duplicate group isolation."""
    df_splits, summary = create_leakage_aware_splits(synthetic_manifest_df, seed=42)

    assert set(df_splits["split"].unique()) == {"train", "val", "test"}
    assert summary["leakage_check_passed"] is True

    # Image-level disjointness
    train_ids = set(df_splits[df_splits["split"] == "train"]["canonical_id"])
    val_ids = set(df_splits[df_splits["split"] == "val"]["canonical_id"])
    test_ids = set(df_splits[df_splits["split"] == "test"]["canonical_id"])

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(synthetic_manifest_df)

    # Group-level isolation (zero duplicate leakage)
    train_groups = set(df_splits[df_splits["split"] == "train"]["future_split_group_id"])
    val_groups = set(df_splits[df_splits["split"] == "val"]["future_split_group_id"])
    test_groups = set(df_splits[df_splits["split"] == "test"]["future_split_group_id"])

    assert len(train_groups & val_groups) == 0
    assert len(train_groups & test_groups) == 0
    assert len(val_groups & test_groups) == 0


def test_split_reproducibility(synthetic_manifest_df):
    """Verify that split generation is strictly deterministic across runs with the same seed."""
    df1, _ = create_leakage_aware_splits(synthetic_manifest_df, seed=42)
    df2, _ = create_leakage_aware_splits(synthetic_manifest_df, seed=42)

    pd.testing.assert_series_equal(df1["split"], df2["split"])

    # Different seed produces different assignments
    df3, _ = create_leakage_aware_splits(synthetic_manifest_df, seed=99)
    assert not (df1["split"] == df3["split"]).all()


def test_split_label_balance(synthetic_manifest_df):
    """Verify that class balance (positive fracture ratio) is preserved across splits."""
    df_splits, summary = create_leakage_aware_splits(synthetic_manifest_df, seed=42)

    overall_pos_pct = synthetic_manifest_df["fractured_label"].mean() * 100
    train_pos_pct = summary["train_positive_pct"]
    val_pos_pct = summary["val_positive_pct"]
    test_pos_pct = summary["test_positive_pct"]

    # Stratified grouping should keep class ratios within ~3 percentage points
    assert abs(train_pos_pct - overall_pos_pct) < 3.0
    assert abs(val_pos_pct - overall_pos_pct) < 4.0
    assert abs(test_pos_pct - overall_pos_pct) < 4.0


# ==============================================================================
# Phase 2.5: Synthetic Geometry Alignment & Truncated JPEG Tests
# ==============================================================================

def test_synthetic_geometry_letterbox_alignment():
    """Verify that letterbox transform preserves geometric alignment across image, box, and mask."""
    transform = LetterboxTransform(target_size=(512, 512), fill_color=(0, 0, 0))

    # Synthetic portrait image: 400 x 800
    orig_w, orig_h = 400, 800
    img = Image.new("RGB", (orig_w, orig_h), color=(50, 50, 50))
    mask = Image.new("L", (orig_w, orig_h), color=0)

    # Known object at [xmin, ymin, xmax, ymax]
    box = (80.0, 160.0, 320.0, 640.0)
    for y in range(int(box[1]), int(box[3])):
        for x in range(int(box[0]), int(box[2])):
            img.putpixel((x, y), (255, 100, 100))
            mask.putpixel((x, y), 255)

    # Apply transform
    trans_img, params = transform.apply_image(img)
    trans_boxes = transform.apply_boxes([box], params)
    trans_mask = transform.apply_mask(mask, params)

    assert trans_img.size == (512, 512)
    assert trans_mask.size == (512, 512)
    assert len(trans_boxes) == 1

    tbox = trans_boxes[0]
    # Check that mask non-zero pixels are strictly within the transformed box bounds
    mask_np = np.asarray(trans_mask)
    nonzero_ys, nonzero_xs = np.where(mask_np > 0)

    assert len(nonzero_xs) > 0
    assert nonzero_xs.min() >= int(np.floor(tbox[0])) - 1
    assert nonzero_xs.max() <= int(np.ceil(tbox[2])) + 1
    assert nonzero_ys.min() >= int(np.floor(tbox[1])) - 1
    assert nonzero_ys.max() <= int(np.ceil(tbox[3])) + 1

    # Invert box back to original
    inv_boxes = transform.invert_boxes(trans_boxes, params)
    assert inv_boxes[0][0] == pytest.approx(box[0], abs=1e-4)
    assert inv_boxes[0][1] == pytest.approx(box[1], abs=1e-4)
    assert inv_boxes[0][2] == pytest.approx(box[2], abs=1e-4)
    assert inv_boxes[0][3] == pytest.approx(box[3], abs=1e-4)


def test_truncated_jpeg_recovery(tmp_path):
    """Verify that truncated JPEGs raise in strict mode but recover with LOAD_TRUNCATED_IMAGES=True."""
    from PIL import ImageFile

    test_img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    valid_jpg_path = tmp_path / "valid.jpg"
    test_img.save(valid_jpg_path, format="JPEG")

    # Read bytes and chop off the last 20 bytes (simulating missing EOF marker)
    with open(valid_jpg_path, "rb") as f:
        jpg_bytes = f.read()

    trunc_bytes = jpg_bytes[:-20]
    trunc_jpg_path = tmp_path / "truncated.jpg"
    with open(trunc_jpg_path, "wb") as f:
        f.write(trunc_bytes)

    # In strict mode, opening and decoding should raise OSError
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with pytest.raises(OSError, match="image file is truncated"):
        with Image.open(trunc_jpg_path) as im:
            im.load()

    # In truncated mode, decoding should succeed without error
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(trunc_jpg_path) as im:
        im.load()
        assert im.size == (100, 100)
        arr = np.asarray(im)
        assert arr.shape == (100, 100, 3)

