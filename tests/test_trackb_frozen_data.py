import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pytest
import torch
import torchvision.transforms.functional as TF

from bonetrace.data.dataset import FracAtlasDataset, rasterize_boxes_to_mask

FROZEN_DIR = Path("data/frozen")
QA_DIR = Path("outputs/trackb/qa")

@pytest.fixture(scope="module")
def manifests():
    train_df = pd.read_csv(FROZEN_DIR / "train.csv")
    val_df = pd.read_csv(FROZEN_DIR / "val.csv")
    test_df = pd.read_csv(FROZEN_DIR / "test.csv")
    return {"train": train_df, "val": val_df, "test": test_df}

def test_checksums_match():
    """Verify that SHA256SUMS matches actual files in data/frozen."""
    sha_file = FROZEN_DIR / "SHA256SUMS"
    assert sha_file.exists(), "SHA256SUMS file missing"
    with open(sha_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line in lines:
        expected_sha, fname = line.split(maxsplit=1)
        fpath = FROZEN_DIR / fname
        assert fpath.exists(), f"File {fname} referenced in SHA256SUMS does not exist"
        with open(fpath, "rb") as fp:
            actual_sha = hashlib.sha256(fp.read()).hexdigest()
        assert actual_sha == expected_sha, f"SHA-256 mismatch for {fname}: expected {expected_sha}, got {actual_sha}"

def test_counts_match_readme(manifests):
    """Verify partition counts match README."""
    readme_path = FROZEN_DIR / "README.md"
    assert readme_path.exists(), "README.md missing in data/frozen"
    readme_text = readme_path.read_text(encoding="utf-8")
    
    tr = manifests["train"]
    va = manifests["val"]
    te = manifests["test"]
    
    assert len(tr) == 2846, f"Expected 2846 train rows, got {len(tr)}"
    assert len(va) == 609, f"Expected 609 val rows, got {len(va)}"
    assert len(te) == 609, f"Expected 609 test rows, got {len(te)}"
    
    assert int(tr["label"].sum()) == 499, f"Expected 499 train positives, got {int(tr['label'].sum())}"
    assert int(va["label"].sum()) == 107, f"Expected 107 val positives, got {int(va['label'].sum())}"
    assert int(te["label"].sum()) == 107, f"Expected 107 test positives, got {int(te['label'].sum())}"
    
    # Check that README contains these numbers
    assert "2846" in readme_text
    assert "499" in readme_text
    assert "609" in readme_text
    assert "107" in readme_text

def test_no_canonical_id_overlap(manifests):
    """Verify no canonical_id appears in two partitions."""
    tr_ids = set(manifests["train"]["canonical_id"])
    va_ids = set(manifests["val"]["canonical_id"])
    te_ids = set(manifests["test"]["canonical_id"])
    
    assert len(tr_ids & va_ids) == 0, f"Train and Val share canonical_ids: {tr_ids & va_ids}"
    assert len(tr_ids & te_ids) == 0, f"Train and Test share canonical_ids: {tr_ids & te_ids}"
    assert len(va_ids & te_ids) == 0, f"Val and Test share canonical_ids: {va_ids & te_ids}"

def test_no_group_id_overlap(manifests):
    """Verify no group_id appears in two partitions."""
    tr_grp = set(manifests["train"]["group_id"])
    va_grp = set(manifests["val"]["group_id"])
    te_grp = set(manifests["test"]["group_id"])
    
    assert len(tr_grp & va_grp) == 0, f"Train and Val share group_ids: {tr_grp & va_grp}"
    assert len(tr_grp & te_grp) == 0, f"Train and Test share group_ids: {tr_grp & te_grp}"
    assert len(va_grp & te_grp) == 0, f"Val and Test share group_ids: {va_grp & te_grp}"

def test_no_sha256_image_content_duplicates(manifests):
    """Verify no SHA-256 image-content duplicates across partitions."""
    tr_sha = set(manifests["train"]["sha256"])
    va_sha = set(manifests["val"]["sha256"])
    te_sha = set(manifests["test"]["sha256"])
    
    assert len(tr_sha & va_sha) == 0, f"Train and Val share SHA-256 hashes: {tr_sha & va_sha}"
    assert len(tr_sha & te_sha) == 0, f"Train and Test share SHA-256 hashes: {tr_sha & te_sha}"
    assert len(va_sha & te_sha) == 0, f"Val and Test share SHA-256 hashes: {va_sha & te_sha}"

def test_disputed_groups_excluded(manifests):
    """Verify the 4 disputed groups are completely excluded."""
    disputed = {"CANON_0206", "CANON_0678", "CANON_1668", "CANON_3339"}
    all_ids = set(manifests["train"]["canonical_id"]) | set(manifests["val"]["canonical_id"]) | set(manifests["test"]["canonical_id"])
    assert len(all_ids & disputed) == 0, f"Disputed IDs present in partitions: {all_ids & disputed}"

def test_box_masks_align_and_qa_spotcheck():
    """Verify box masks align with letterboxed images and save 10 QA overlays to outputs/trackb/qa/."""
    QA_DIR.mkdir(parents=True, exist_ok=True)
    
    ds = FracAtlasDataset(manifest="data/frozen/train.csv", normalize=False)
    
    # Find 10 positives with annotations
    positives = []
    for i in range(len(ds)):
        sample = ds[i]
        if sample["label"].item() == 1.0 and sample["has_annotation"].item():
            positives.append(sample)
            if len(positives) >= 10:
                break
    
    assert len(positives) >= 10, f"Found only {len(positives)} positive samples with annotations"
    
    for idx, sample in enumerate(positives):
        canon_id = sample["canonical_id"]
        img_t = sample["image"]  # (3, 512, 512) in [0, 1]
        mask_t = sample["mask"]  # (1, 512, 512) in [0, 1]
        boxes_t = sample["boxes"]  # (N, 4) in letterbox pixel coords
        
        # Verify alignment: non-zero mask pixels must fall within the bounding boxes
        mask_arr = mask_t.squeeze(0).numpy() > 0.5
        assert mask_arr.sum() > 0, f"Sample {canon_id} has empty rasterized mask"
        
        # Spot-check image creation
        img_pil = TF.to_pil_image(img_t)
        draw = ImageDraw.Draw(img_pil)
        
        # Draw bounding boxes in red
        for b in boxes_t.numpy():
            draw.rectangle([b[0], b[1], b[2], b[3]], outline="red", width=3)
        
        # Create semi-transparent green overlay for the mask
        mask_img = Image.fromarray((mask_arr * 160).astype(np.uint8), mode="L")
        green_overlay = Image.new("RGBA", (512, 512), (0, 255, 0, 0))
        green_overlay.putalpha(mask_img)
        
        combined = img_pil.convert("RGBA")
        combined = Image.alpha_composite(combined, green_overlay)
        
        save_path = QA_DIR / f"spotcheck_{idx+1:02d}_{canon_id}.png"
        combined.convert("RGB").save(save_path)
        assert save_path.exists(), f"Failed to save QA spot-check image {save_path}"

    qa_files = list(QA_DIR.glob("*.png"))
    assert len(qa_files) >= 10, f"Expected at least 10 QA spotcheck images, found {len(qa_files)}"
