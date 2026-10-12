"""Unit tests for Phase 1.5 reconciliation and annotation verification logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
from PIL import Image

from bonetrace.config import AppConfig, PathsConfig
from bonetrace.data.reconcile import (
    DatasetReconciler,
    calculate_box_iou,
    coco_bbox_to_xyxy,
    compare_polygon_vertices,
    investigate_published_vs_audit_counts,
    polygon_to_xy_pairs,
    run_reconciliation,
    voc_bndbox_to_xyxy,
    yolo_to_xyxy,
)


def create_dummy_jpeg(path: Path, size: Tuple[int, int] = (100, 100), color: Tuple[int, int, int] = (128, 128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG")


# ==============================================================================
# Coordinate Normalization & Math Unit Tests
# ==============================================================================

def test_coco_bbox_to_xyxy():
    # [x, y, w, h] -> [xmin, ymin, xmax, ymax]
    box = coco_bbox_to_xyxy([10.5, 20.0, 30.0, 40.5])
    assert box == (10.5, 20.0, 40.5, 60.5)

    with pytest.raises(ValueError):
        coco_bbox_to_xyxy([10, 20, 30])  # Length != 4


def test_voc_bndbox_to_xyxy():
    box = voc_bndbox_to_xyxy(10, 20, 40, 60)
    assert box == (10.0, 20.0, 40.0, 60.0)


def test_yolo_to_xyxy():
    # Image 100x200: center (0.5, 0.5), width 0.2, height 0.4
    # abs center: (50, 100), abs w: 20, abs h: 80
    # xmin: 40, ymin: 60, xmax: 60, ymax: 140
    box = yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 100, 200)
    assert box == (40.0, 60.0, 60.0, 140.0)

    with pytest.raises(ValueError):
        yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 0, 200)


def test_calculate_box_iou():
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (0.0, 0.0, 10.0, 10.0)
    # Identical
    assert calculate_box_iou(b1, b2) == 1.0

    # Non-overlapping
    b3 = (20.0, 20.0, 30.0, 30.0)
    assert calculate_box_iou(b1, b3) == 0.0

    # 50% overlap: intersection 5x10=50, union 100+100-50=150 -> 1/3
    b4 = (5.0, 0.0, 15.0, 10.0)
    assert pytest.approx(calculate_box_iou(b1, b4), 0.001) == 50.0 / 150.0


def test_polygon_to_xy_pairs():
    flat = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    pairs = polygon_to_xy_pairs(flat)
    assert pairs == [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]


def test_compare_polygon_vertices():
    # Identical open
    p1 = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
    p2 = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
    assert compare_polygon_vertices(p1, p2)["match"] is True

    # One closed (last point repeats first), one open
    p2_closed = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (10.0, 20.0)]
    assert compare_polygon_vertices(p1, p2_closed)["match"] is True

    # Shifted point
    p3 = [(10.0, 20.0), (35.0, 40.0), (50.0, 60.0)]
    assert compare_polygon_vertices(p1, p3)["match"] is False


# ==============================================================================
# Manifest, Duplicate Grouping & Discrepancy Unit Tests
# ==============================================================================

@pytest.fixture
def mock_recon_dataset(tmp_path: Path) -> Tuple[Path, Path]:
    raw_dir = tmp_path / "raw" / "FracAtlas"
    out_dir = tmp_path / "outputs" / "stage1"

    frac_dir = raw_dir / "images" / "Fractured"
    non_frac_dir = raw_dir / "images" / "Non_fractured"
    frac_dir.mkdir(parents=True, exist_ok=True)
    non_frac_dir.mkdir(parents=True, exist_ok=True)

    # Image 1 (Positive)
    create_dummy_jpeg(frac_dir / "IMG0000001.jpg", size=(100, 100), color=(150, 0, 0))
    # Image 2 (Clone of Image 1, different name)
    create_dummy_jpeg(frac_dir / "IMG0000002.jpg", size=(100, 100), color=(150, 0, 0))
    # Image 3 (Negative)
    create_dummy_jpeg(non_frac_dir / "IMG0000003.jpg", size=(100, 100), color=(0, 0, 150))
    # Image 4 (Discrepancy: real 80x120, but annotation assumed 120x80)
    create_dummy_jpeg(frac_dir / "IMG0000004.jpg", size=(80, 120), color=(50, 50, 50))

    # dataset.csv
    csv_rows = [
        {"image_id": "IMG0000001.jpg", "hand": 1, "leg": 0, "hip": 0, "shoulder": 0, "mixed": 0, "hardware": 0, "multiscan": 0, "fractured": 1, "fracture_count": 1, "frontal": 1, "lateral": 0, "oblique": 0},
        {"image_id": "IMG0000002.jpg", "hand": 1, "leg": 0, "hip": 0, "shoulder": 0, "mixed": 0, "hardware": 0, "multiscan": 0, "fractured": 1, "fracture_count": 1, "frontal": 1, "lateral": 0, "oblique": 0},
        {"image_id": "IMG0000003.jpg", "hand": 0, "leg": 1, "hip": 0, "shoulder": 0, "mixed": 0, "hardware": 0, "multiscan": 0, "fractured": 0, "fracture_count": 0, "frontal": 0, "lateral": 1, "oblique": 0},
        {"image_id": "IMG0000004.jpg", "hand": 0, "leg": 0, "hip": 1, "shoulder": 0, "mixed": 0, "hardware": 0, "multiscan": 0, "fractured": 1, "fracture_count": 1, "frontal": 1, "lateral": 0, "oblique": 0},
    ]
    pd.DataFrame(csv_rows).to_csv(raw_dir / "dataset.csv", index=False)

    # COCO JSON
    coco_data = {
        "images": [
            {"id": 1, "file_name": "IMG0000001.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "IMG0000002.jpg", "width": 100, "height": 100},
            {"id": 4, "file_name": "IMG0000004.jpg", "width": 120, "height": 80},  # Swapped metadata!
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]], "area": 400.0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20], "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]], "area": 400.0},
            {"id": 4, "image_id": 4, "category_id": 1, "bbox": [40, 20, 20, 20], "segmentation": [[40, 20, 60, 20, 60, 40, 40, 40]], "area": 400.0},
        ],
        "categories": [{"id": 1, "name": "fractured"}],
    }
    coco_dir = raw_dir / "Annotations" / "COCO JSON"
    coco_dir.mkdir(parents=True, exist_ok=True)
    with open(coco_dir / "COCO_fracture_masks.json", "w") as f:
        json.dump(coco_data, f)

    # VGG JSON
    vgg_data = {
        "IMG0000001.jpg": {"regions": {"0": {"shape_attributes": {"name": "polygon", "all_points_x": [10, 30, 30, 10, 10], "all_points_y": [10, 10, 30, 30, 10]}}}},
        "IMG0000002.jpg": {"regions": {"0": {"shape_attributes": {"name": "polygon", "all_points_x": [10, 30, 30, 10, 10], "all_points_y": [10, 10, 30, 30, 10]}}}},
        "IMG0000004.jpg": {"regions": {"0": {"shape_attributes": {"name": "polygon", "all_points_x": [40, 60, 60, 40, 40], "all_points_y": [20, 20, 40, 40, 20]}}}},
    }
    vgg_dir = raw_dir / "Annotations" / "VGG JSON"
    vgg_dir.mkdir(parents=True, exist_ok=True)
    with open(vgg_dir / "VGG_fracture_masks.json", "w") as f:
        json.dump(vgg_data, f)

    # VOC XMLs
    voc_dir = raw_dir / "Annotations" / "PASCAL VOC"
    voc_dir.mkdir(parents=True, exist_ok=True)
    (voc_dir / "IMG0000001.xml").write_text("<annotation><filename>IMG0000001.jpg</filename><size><width>100</width><height>100</height></size><object><name>fracture</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object></annotation>", encoding="utf-8")
    (voc_dir / "IMG0000002.xml").write_text("<annotation><filename>IMG0000002.jpg</filename><size><width>100</width><height>100</height></size><object><name>fracture</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object></annotation>", encoding="utf-8")
    (voc_dir / "IMG0000003.xml").write_text("<annotation><filename>IMG0000003.jpg</filename><size><width>100</width><height>100</height></size></annotation>", encoding="utf-8")
    # IMG0000004 with swapped XML size tag
    (voc_dir / "IMG0000004.xml").write_text("<annotation><filename>IMG0000004.jpg</filename><size><width>120</width><height>80</height></size><object><name>fracture</name><bndbox><xmin>40</xmin><ymin>20</ymin><xmax>60</xmax><ymax>40</ymax></bndbox></object></annotation>", encoding="utf-8")

    # YOLO TXTs
    yolo_dir = raw_dir / "Annotations" / "YOLO"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels.txt").write_text("fractured\n", encoding="utf-8")
    # IMG0000001: center (20, 20) -> norm 0.2, 0.2
    (yolo_dir / "IMG0000001.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    (yolo_dir / "IMG0000002.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    (yolo_dir / "IMG0000003.txt").write_text("", encoding="utf-8")
    # IMG0000004: normalized with swapped dims (120, 80): center (50/120=0.4167, 30/80=0.375)
    (yolo_dir / "IMG0000004.txt").write_text("0 0.416667 0.375 0.166667 0.25\n", encoding="utf-8")

    return raw_dir, out_dir


def test_duplicate_grouping_prevents_split_leakage(mock_recon_dataset: Tuple[Path, Path]):
    raw_dir, out_dir = mock_recon_dataset
    reconciler = DatasetReconciler(raw_data_dir=raw_dir, output_stage1_dir=out_dir)
    records, dup_groups = reconciler._build_canonical_manifest()

    # IMG0000001 and IMG0000002 have identical bytes -> must be collapsed into 1 canonical record
    # Total unique contents: 3 (IMG1/2 clone, IMG3, IMG4)
    assert len(records) == 3

    clone_record = next(r for r in records if "IMG0000001.jpg" in r.all_image_ids)
    assert clone_record.num_copies == 2
    assert clone_record.is_duplicate_content is True
    assert set(clone_record.all_image_ids) == {"IMG0000001.jpg", "IMG0000002.jpg"}
    assert clone_record.duplicate_group_id is not None
    # Verify future_split_group_id is set to the duplicate group key
    assert clone_record.future_split_group_id == clone_record.duplicate_group_id


def test_discrepancy_detection_yolo_shift(mock_recon_dataset: Tuple[Path, Path]):
    raw_dir, out_dir = mock_recon_dataset
    reconciler = DatasetReconciler(raw_data_dir=raw_dir, output_stage1_dir=out_dir)
    comparisons, metrics = reconciler._compare_annotations()

    # Check IMG0000004 (has swapped dims 120x80 in XML vs real 80x120)
    c4 = next(c for c in comparisons if c["image_id"] == "IMG0000004.jpg")
    assert c4["dimension_transposition_detected"] is True
    # Due to normalization with swapped dims, YOLO de-normalized with real dims shifts
    assert c4["yolo_coordinate_shift_detected"] is True
    assert c4["mean_coco_yolo_iou"] < 0.5


def test_dataset_reconciler_full_pipeline(mock_recon_dataset: Tuple[Path, Path]):
    raw_dir, out_dir = mock_recon_dataset
    reconciler = DatasetReconciler(raw_data_dir=raw_dir, output_stage1_dir=out_dir)
    summary = reconciler.run()

    assert (out_dir / "canonical_manifest.json").exists()
    assert (out_dir / "canonical_manifest.csv").exists()
    assert (out_dir / "annotation_comparison.csv").exists()
    assert (out_dir / "reconciliation_summary.json").exists()
    assert (out_dir / "RECONCILIATION_REPORT.md").exists()
    assert (out_dir / "annotation_review").exists()

    df_man = pd.read_csv(out_dir / "canonical_manifest.csv")
    assert len(df_man) == 3
    assert "future_split_group_id" in df_man.columns


def test_run_reconciliation_pipeline_entrypoint(mock_recon_dataset: Tuple[Path, Path]):
    raw_dir, out_dir = mock_recon_dataset
    cfg = AppConfig(
        paths=PathsConfig(
            raw_data_dir=raw_dir,
            output_dir=out_dir.parent,
        )
    )
    res = run_reconciliation(cfg)
    assert "manifest_summary" in res
    assert res["manifest_summary"]["total_unique_image_contents"] == 3
