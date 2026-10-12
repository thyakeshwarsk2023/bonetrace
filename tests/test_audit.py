"""Unit tests for Phase 1 scientific dataset auditing logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
from PIL import Image

from bonetrace.config import AppConfig, PathsConfig
from bonetrace.data.audit import (
    DatasetAuditor,
    compute_file_hashes,
    polygon_area,
    resolve_raw_dataset_dir,
    run_audit,
)


def create_dummy_jpeg(path: Path, size: Tuple[int, int] = (64, 64), color: Tuple[int, int, int] = (100, 100, 100)) -> None:
    """Helper to create a minimal valid JPEG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG")


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a fully-formed mock dataset structure with annotations and images."""
    raw_dir = tmp_path / "raw" / "FracAtlas"
    out_dir = tmp_path / "outputs" / "stage1"

    # Images dirs
    frac_dir = raw_dir / "images" / "Fractured"
    non_frac_dir = raw_dir / "images" / "Non_fractured"
    frac_dir.mkdir(parents=True, exist_ok=True)
    non_frac_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create images
    # img1: Fractured
    create_dummy_jpeg(frac_dir / "IMG0000001.jpg", size=(100, 100), color=(200, 50, 50))
    # img2: Fractured (duplicate of img1 data under different filename)
    create_dummy_jpeg(frac_dir / "IMG0000002.jpg", size=(100, 100), color=(200, 50, 50))
    # img3: Non-fractured
    create_dummy_jpeg(non_frac_dir / "IMG0000003.jpg", size=(100, 100), color=(50, 50, 200))
    # img4: Cross-folder duplicate (exists in both folders)
    create_dummy_jpeg(frac_dir / "IMG0000004.jpg", size=(80, 80), color=(80, 80, 80))
    create_dummy_jpeg(non_frac_dir / "IMG0000004.jpg", size=(80, 80), color=(80, 80, 80))

    # 2. Create dataset.csv
    csv_data = [
        {"image_id": "IMG0000001.jpg", "hand": 1, "leg": 0, "hip": 0, "shoulder": 0, "mixed": 0, "hardware": 0, "multiscan": 0, "fractured": 1, "fracture_count": 1, "frontal": 1, "lateral": 0, "oblique": 0},
        {"image_id": "IMG0000002.jpg", "hand": 1, "leg": 0, "hip": 0, "shoulder": 0, "mixed": 0, "hardware": 0, "multiscan": 0, "fractured": 1, "fracture_count": 1, "frontal": 1, "lateral": 0, "oblique": 0},
        {"image_id": "IMG0000003.jpg", "hand": 0, "leg": 1, "hip": 0, "shoulder": 0, "mixed": 0, "hardware": 0, "multiscan": 0, "fractured": 0, "fracture_count": 0, "frontal": 0, "lateral": 1, "oblique": 0},
        {"image_id": "IMG0000004.jpg", "hand": 0, "leg": 0, "hip": 1, "shoulder": 0, "mixed": 0, "hardware": 1, "multiscan": 0, "fractured": 1, "fracture_count": 1, "frontal": 1, "lateral": 0, "oblique": 0},
    ]
    pd.DataFrame(csv_data).to_csv(raw_dir / "dataset.csv", index=False)

    # 3. Create COCO JSON
    coco_data = {
        "images": [
            {"id": 1, "file_name": "IMG0000001.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "IMG0000002.jpg", "width": 100, "height": 100},
            {"id": 4, "file_name": "IMG0000004.jpg", "width": 80, "height": 80},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]], "area": 400.0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20], "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]], "area": 400.0},
            {"id": 3, "image_id": 4, "category_id": 1, "bbox": [5, 5, 15, 15], "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]], "area": 225.0},
        ],
        "categories": [{"id": 1, "name": "fractured"}],
    }
    coco_dir = raw_dir / "Annotations" / "COCO JSON"
    coco_dir.mkdir(parents=True, exist_ok=True)
    with open(coco_dir / "COCO_fracture_masks.json", "w") as f:
        json.dump(coco_data, f)

    # 4. Create VGG JSON
    vgg_data = {
        "IMG0000001.jpg": {
            "filename": "IMG0000001.jpg",
            "regions": {"0": {"shape_attributes": {"name": "polygon", "all_points_x": [10, 30, 30, 10], "all_points_y": [10, 10, 30, 30]}, "region_attributes": {"label": "fractured"}}},
        },
        "IMG0000002.jpg": {
            "filename": "IMG0000002.jpg",
            "regions": {"0": {"shape_attributes": {"name": "polygon", "all_points_x": [10, 30, 30, 10], "all_points_y": [10, 10, 30, 30]}, "region_attributes": {"label": "fractured"}}},
        },
        "IMG0000004.jpg": {
            "filename": "IMG0000004.jpg",
            "regions": {"0": {"shape_attributes": {"name": "polygon", "all_points_x": [5, 20, 20, 5], "all_points_y": [5, 5, 20, 20]}, "region_attributes": {"label": "fractured"}}},
        },
    }
    vgg_dir = raw_dir / "Annotations" / "VGG JSON"
    vgg_dir.mkdir(parents=True, exist_ok=True)
    with open(vgg_dir / "VGG_fracture_masks.json", "w") as f:
        json.dump(vgg_data, f)

    # 5. Create PASCAL VOC XMLs
    voc_dir = raw_dir / "Annotations" / "PASCAL VOC"
    voc_dir.mkdir(parents=True, exist_ok=True)
    for img_id, (w, h, has_obj) in [("IMG0000001.jpg", (100, 100, True)), ("IMG0000002.jpg", (100, 100, True)), ("IMG0000003.jpg", (100, 100, False)), ("IMG0000004.jpg", (80, 80, True))]:
        xml_stem = Path(img_id).stem
        obj_xml = "<object><name>fracture</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object>" if has_obj else ""
        xml_content = f"<annotation><filename>{img_id}</filename><size><width>{w}</width><height>{h}</height></size>{obj_xml}</annotation>"
        (voc_dir / f"{xml_stem}.xml").write_text(xml_content, encoding="utf-8")

    # 6. Create YOLO TXTs
    yolo_dir = raw_dir / "Annotations" / "YOLO"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels.txt").write_text("fractured\n", encoding="utf-8")
    (yolo_dir / "IMG0000001.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    (yolo_dir / "IMG0000002.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    (yolo_dir / "IMG0000003.txt").write_text("", encoding="utf-8")
    (yolo_dir / "IMG0000004.txt").write_text("0 0.15 0.15 0.15 0.15\n", encoding="utf-8")

    # 7. Utilities
    (raw_dir / "Utilities").mkdir(parents=True, exist_ok=True)

    return raw_dir, out_dir


def test_polygon_area_formula():
    """Verify Shoelace formula on known geometric shapes."""
    # Unit square
    assert polygon_area([0, 1, 1, 0], [0, 0, 1, 1]) == 1.0
    # Degenerate
    assert polygon_area([0, 1], [0, 1]) == 0.0
    # Triangle
    assert polygon_area([0, 2, 0], [0, 0, 2]) == 2.0


def test_compute_file_hashes(tmp_path: Path):
    """Verify SHA-256 and MD5 hashing."""
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"BoneTrace Phase 1 Audit")
    sha256, md5, size = compute_file_hashes(test_file)
    assert len(sha256) == 64
    assert len(md5) == 32
    assert size == len(b"BoneTrace Phase 1 Audit")


def test_resolve_raw_dataset_dir(tmp_path: Path):
    """Verify raw directory resolution for both flat and nested hierarchies."""
    # Flat
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "dataset.csv").touch()
    assert resolve_raw_dataset_dir(flat) == flat

    # Nested FracAtlas
    nested_root = tmp_path / "nested"
    nested_sub = nested_root / "FracAtlas"
    nested_sub.mkdir(parents=True)
    (nested_sub / "dataset.csv").touch()
    assert resolve_raw_dataset_dir(nested_root) == nested_sub


def test_dataset_auditor_full_run(mock_dataset: Tuple[Path, Path]):
    """Execute complete audit on mock dataset and verify exported outputs."""
    raw_dir, out_dir = mock_dataset
    auditor = DatasetAuditor(raw_data_dir=raw_dir, output_stage1_dir=out_dir)
    summary = auditor.run()

    # Verify summary structure
    assert summary["csv_metadata"]["row_count"] == 4
    assert summary["images_summary"]["unique_filenames_on_disk"] == 4
    assert summary["images_summary"]["total_files_on_disk"] == 5  # IMG0000004 in both folders
    assert summary["images_summary"]["corrupted_images_count"] == 0
    assert summary["annotations"]["coco"]["annotations_count"] == 3
    assert summary["annotations"]["pascal_voc"]["total_bounding_boxes"] == 3
    assert summary["annotations"]["yolo"]["total_bounding_boxes"] == 3

    # Check cross-folder overlap detected
    assert "IMG0000004.jpg" in summary["images_summary"]["overlapping_filenames_between_folders"]

    # Check duplicate hash detection
    # IMG0000001 and IMG0000002 share identical bytes
    assert summary["duplicate_analysis"]["num_duplicate_hash_groups"] >= 2

    # Check verified findings and uncompleted checks exist
    assert len(summary["verified_findings"]) > 0
    assert any(uc["id"] == "UC-01" for uc in summary["uncompleted_checks"])

    # Verify file exports
    assert (out_dir / "audit_summary.json").exists()
    assert (out_dir / "image_audit_records.csv").exists()
    assert (out_dir / "duplicate_images.csv").exists()
    assert (out_dir / "annotation_discrepancies.csv").exists()
    assert (out_dir / "class_distribution.csv").exists()
    assert (out_dir / "AUDIT_REPORT.md").exists()

    # Verify records CSV content
    df_recs = pd.read_csv(out_dir / "image_audit_records.csv")
    assert len(df_recs) == 4
    assert set(df_recs["image_id"]) == {"IMG0000001.jpg", "IMG0000002.jpg", "IMG0000003.jpg", "IMG0000004.jpg"}


def test_auditor_detects_label_conflict(tmp_path: Path):
    """Verify that logical contradictions in CSV labels are caught."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    out_dir = tmp_path / "out"

    # CSV with fractured=0 but fracture_count=2
    bad_csv = [
        {"image_id": "IMG1.jpg", "hand": 1, "leg": 0, "hip": 0, "shoulder": 0, "mixed": 0, "hardware": 0, "multiscan": 0, "fractured": 0, "fracture_count": 2, "frontal": 1, "lateral": 0, "oblique": 0}
    ]
    pd.DataFrame(bad_csv).to_csv(raw_dir / "dataset.csv", index=False)

    auditor = DatasetAuditor(raw_data_dir=raw_dir, output_stage1_dir=out_dir)
    csv_res = auditor._audit_csv_metadata()
    assert len(csv_res["label_conflicts"]) > 0
    assert any("fractured==0 but fracture_count > 0" in c for c in csv_res["label_conflicts"])


def test_auditor_detects_corrupted_image(mock_dataset: Tuple[Path, Path]):
    """Verify that unreadable/corrupted images are recorded without crashing."""
    raw_dir, out_dir = mock_dataset
    # Corrupt an image
    bad_img = raw_dir / "images" / "Non_fractured" / "IMG0000003.jpg"
    bad_img.write_bytes(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED")

    auditor = DatasetAuditor(raw_data_dir=raw_dir, output_stage1_dir=out_dir)
    img_res = auditor._audit_disk_images()
    assert img_res["corrupted_images_count"] == 1
    assert img_res["corrupted_images"][0]["filename"] == "IMG0000003.jpg"


def test_auditor_read_only_guarantee(mock_dataset: Tuple[Path, Path]):
    """Verify that raw files are not modified during auditing."""
    raw_dir, out_dir = mock_dataset
    csv_file = raw_dir / "dataset.csv"
    orig_bytes = csv_file.read_bytes()
    orig_mtime = csv_file.stat().st_mtime

    auditor = DatasetAuditor(raw_data_dir=raw_dir, output_stage1_dir=out_dir)
    auditor.run()

    assert csv_file.read_bytes() == orig_bytes
    assert csv_file.stat().st_mtime == orig_mtime


def test_run_audit_pipeline_entrypoint(mock_dataset: Tuple[Path, Path]):
    """Verify the top-level run_audit function with AppConfig."""
    raw_dir, out_dir = mock_dataset
    cfg = AppConfig(
        paths=PathsConfig(
            raw_data_dir=raw_dir,
            output_dir=out_dir.parent,
        )
    )
    res = run_audit(cfg)
    assert "audit_metadata" in res
    assert (out_dir / "audit_summary.json").exists()
