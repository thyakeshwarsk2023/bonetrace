"""PyTorch Dataset for the FracAtlas musculoskeletal radiograph dataset.

Implements FracAtlasDataset with support for:
- Frozen split manifests (train, val, test) and content-addressed canonical manifests.
- RGB standardisation and aspect-ratio preserving letterboxing (512x512).
- Ground truth fracture bounding boxes and binary region-supervision target masks.
- Safe local handling of truncated JPEGs without leaking global Pillow state.
- Transparent, auditable exclusion of confirmed duplicate clone groups with
  conflicting fracture labels (CANON_0206, CANON_0678, CANON_1668, CANON_3339)
  while preserving distinct candidate radiographs (RETAIN_SEPARATE_CANDIDATE).
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from bonetrace.data.preprocess import LetterboxTransform

logger = logging.getLogger("bonetrace.data.dataset")

# The four confirmed duplicate clone groups exhibiting conflicting binary fracture labels.
# CANON_0206: IMG0002886.jpg (fractured=1) vs IMG0004243.jpg (fractured=0)
# CANON_0678: IMG0002952.jpg (fractured=0) vs IMG0004262.jpg (fractured=1)
# CANON_1668: IMG0003034.jpg (fractured=1) vs IMG0004314.jpg (fractured=0)
# CANON_3339: IMG0004280.jpg (fractured=0) vs IMG0002998.jpg (fractured=1)
DISPUTED_CLONE_CANONICAL_IDS: Tuple[str, ...] = (
    "CANON_0206",
    "CANON_0678",
    "CANON_1668",
    "CANON_3339",
)

# Standard ImageNet normalization parameters
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@contextlib.contextmanager
def safe_truncated_image_loading():
    """Context manager ensuring LOAD_TRUNCATED_IMAGES is restored after decoding."""
    previous_state = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        yield
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_state


def load_image_safely(file_path: Union[str, Path]) -> Image.Image:
    """Safely decode an image file to RGB mode without mutating global Pillow state."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    with safe_truncated_image_loading():
        with Image.open(path) as img:
            return img.convert("RGB")


def rasterize_boxes_to_mask(
    boxes: Sequence[Sequence[float]],
    canvas_size: Tuple[int, int] = (512, 512),
    mode: str = "hard",
    sigma_scale: float = 4.0,
) -> torch.Tensor:
    """Rasterize bounding boxes [xmin, ymin, xmax, ymax] onto a region-supervision mask tensor.

    Supports:
      - 'hard': Standard binary box mask with 1.0 inside and 0.0 outside (legacy default).
      - 'gaussian': Normalized 2D elliptical Gaussian distance transform centered on the lesion fissure,
        concentrating supervisory signal at the central fracture line and smoothly attenuating
        false penalties on healthy surrounding tissue near box borders.
      - 'distance_transform': Normalized Euclidean distance transform from box borders to medial axis.

    Args:
        boxes: List or sequence of bounding boxes in canvas pixel coordinates.
        canvas_size: (width, height) of the canvas.
        mode: Rasterization strategy ('hard', 'gaussian', 'distance_transform').
        sigma_scale: Scale factor for Gaussian standard deviation relative to box dimensions (default: 4.0).

    Returns:
        torch.Tensor of shape (1, height, width) with float values in [0.0, 1.0].
    """
    target_w, target_h = canvas_size
    if not boxes:
        return torch.zeros((1, target_h, target_w), dtype=torch.float32)

    if mode == "hard":
        mask_img = Image.new("L", (target_w, target_h), 0)
        draw = ImageDraw.Draw(mask_img)

        for box in boxes:
            if len(box) != 4:
                continue
            xmin, ymin, xmax, ymax = box
            xmin_c = max(0.0, min(float(xmin), float(target_w)))
            ymin_c = max(0.0, min(float(ymin), float(target_h)))
            xmax_c = max(0.0, min(float(xmax), float(target_w)))
            ymax_c = max(0.0, min(float(ymax), float(target_h)))
            if xmax_c > xmin_c and ymax_c > ymin_c:
                draw.rectangle([xmin_c, ymin_c, xmax_c, ymax_c], fill=255)

        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
        return torch.from_numpy(mask_arr).unsqueeze(0)

    # Soft boundary refinement modes ('gaussian' or 'distance_transform')
    mask_arr = np.zeros((target_h, target_w), dtype=np.float32)
    for box in boxes:
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = [float(v) for v in box]
        xmin_c = max(0.0, min(xmin, float(target_w)))
        ymin_c = max(0.0, min(ymin, float(target_h)))
        xmax_c = max(0.0, min(xmax, float(target_w)))
        ymax_c = max(0.0, min(ymax, float(target_h)))
        bw = xmax_c - xmin_c
        bh = ymax_c - ymin_c
        if bw <= 0.0 or bh <= 0.0:
            continue

        x_indices = np.arange(int(np.floor(xmin_c)), int(np.ceil(xmax_c)))
        y_indices = np.arange(int(np.floor(ymin_c)), int(np.ceil(ymax_c)))
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue
        xx, yy = np.meshgrid(x_indices, y_indices)

        if mode == "gaussian":
            cx = (xmin_c + xmax_c) / 2.0
            cy = (ymin_c + ymax_c) / 2.0
            sigma_x = max(1.0, bw / sigma_scale)
            sigma_y = max(1.0, bh / sigma_scale)
            g = np.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
            mask_arr[yy, xx] = np.maximum(mask_arr[yy, xx], g.astype(np.float32))

        elif mode == "distance_transform":
            dx = np.minimum(xx - xmin_c, xmax_c - xx)
            dy = np.minimum(yy - ymin_c, ymax_c - yy)
            d = np.minimum(dx, dy)
            max_d = max(1.0, min(bw, bh) / 2.0)
            norm_d = np.clip(d / max_d, 0.0, 1.0)
            mask_arr[yy, xx] = np.maximum(mask_arr[yy, xx], norm_d.astype(np.float32))

        else:
            raise ValueError(
                f"Unsupported mask rasterization mode: '{mode}'. Expected 'hard', 'gaussian', or 'distance_transform'."
            )

    return torch.from_numpy(mask_arr).unsqueeze(0)


class FracAtlasDataset(Dataset):
    """PyTorch Dataset for FracAtlas musculoskeletal radiographs.

    Provides standardized letterboxed images, ground truth binary fracture labels,
    canonical IDs, bounding boxes, and rasterized binary region-supervision masks.
    """

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        split: Optional[str] = None,
        target_size: Tuple[int, int] = (512, 512),
        exclude_conflicts: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        mask_mode: str = "hard",
    ) -> None:
        """Initialize FracAtlasDataset.

        Args:
            manifest: Path to split CSV (e.g. data/processed/train.csv) or DataFrame.
            split: Optional split filter ('train', 'val', 'test').
            target_size: Target (width, height) for letterbox transformation (default: 512x512).
            exclude_conflicts: Whether to exclude the 4 verified disputed clone groups.
            transform: Optional tensor-level transformation applied after tensor conversion.
            normalize: Whether to apply ImageNet standardization (mean/std).
            mask_mode: Strategy for rasterizing bounding boxes ('hard', 'gaussian', 'distance_transform').
        """
        super().__init__()
        self.target_size = target_size
        self.exclude_conflicts = exclude_conflicts
        self.transform = transform
        self.normalize = normalize
        self.mask_mode = mask_mode
        self.letterbox = LetterboxTransform(target_size=target_size)

        # 1. Load dataframe
        if isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
        else:
            manifest_path = Path(manifest)
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")
            df = pd.read_csv(manifest_path)

        # 2. Filter by split if requested and column exists
        if split is not None and "split" in df.columns:
            valid_splits = {"train", "val", "test"}
            if split not in valid_splits:
                raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}")
            df = df[df["split"] == split].copy()

        # 3. Validate required columns
        required_cols = ["canonical_id", "resolved_primary_path", "fractured_label"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Manifest missing required columns: {missing_cols}")

        # 4. Exclude disputed label-conflict clone groups if configured
        self.excluded_records: List[Dict[str, Any]] = []
        if self.exclude_conflicts:
            mask_disputed = df["canonical_id"].isin(DISPUTED_CLONE_CANONICAL_IDS)
            if mask_disputed.any():
                self.excluded_records = df[mask_disputed].to_dict(orient="records")
                df = df[~mask_disputed].copy()
                logger.info(
                    "Excluded %d disputed clone records: %s",
                    len(self.excluded_records),
                    [r["canonical_id"] for r in self.excluded_records],
                )

        # 5. Validate label values
        invalid_labels = df[~df["fractured_label"].isin([0, 1])]
        if len(invalid_labels) > 0:
            raise ValueError(
                f"Manifest contains {len(invalid_labels)} rows with non-binary fractured_label values."
            )

        self.df = df.reset_index(drop=True)
        self.records: List[Dict[str, Any]] = self.df.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        canon_id = str(record["canonical_id"])
        image_path = record["resolved_primary_path"]
        label_val = float(record["fractured_label"])

        # Decode image safely
        img_raw = load_image_safely(image_path)

        # Apply aspect-ratio preserving letterbox transform
        img_letterboxed, _ = self.letterbox.apply_image(img_raw)

        # Convert PIL Image to PyTorch FloatTensor in [0.0, 1.0] range: (3, H, W)
        img_tensor = TF.to_tensor(img_letterboxed)

        # Parse ground-truth bounding boxes
        boxes_list: List[List[float]] = []
        if "letterbox_target_boxes" in record and pd.notna(record["letterbox_target_boxes"]):
            raw_boxes = record["letterbox_target_boxes"]
            if isinstance(raw_boxes, str):
                try:
                    boxes_list = json.loads(raw_boxes)
                except Exception:
                    boxes_list = []
            elif isinstance(raw_boxes, list):
                boxes_list = raw_boxes

        has_annotation = bool(len(boxes_list) > 0 and label_val > 0.5)

        # Generate region target mask (1, H, W) via configured rasterization strategy
        mask_tensor = rasterize_boxes_to_mask(
            boxes_list, canvas_size=self.target_size, mode=self.mask_mode
        )

        # Bounding boxes tensor (N, 4)
        if boxes_list:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)

        # Apply user data transforms if provided
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        elif self.normalize:
            img_tensor = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        label_tensor = torch.tensor([label_val], dtype=torch.float32)

        return {
            "image": img_tensor,
            "label": label_tensor,
            "canonical_id": canon_id,
            "boxes": boxes_tensor,
            "mask": mask_tensor,
            "has_annotation": torch.tensor(has_annotation, dtype=torch.bool),
        }

    def get_class_counts(self) -> Dict[int, int]:
        """Return binary class counts {0: count_neg, 1: count_pos}."""
        counts = self.df["fractured_label"].value_counts().to_dict()
        return {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0))}

    def get_pos_weight(self) -> float:
        """Calculate recommended positive class weight for BCEWithLogitsLoss."""
        counts = self.get_class_counts()
        n_neg = counts.get(0, 0)
        n_pos = counts.get(1, 0)
        if n_pos == 0:
            return 1.0
        return float(n_neg / n_pos)


def fracatlas_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function handling variable-length bounding box tensors per sample."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    has_anns = torch.stack([b["has_annotation"] for b in batch], dim=0)
    canonical_ids = [b["canonical_id"] for b in batch]
    boxes = [b["boxes"] for b in batch]

    return {
        "image": images,
        "label": labels,
        "mask": masks,
        "has_annotation": has_anns,
        "canonical_id": canonical_ids,
        "boxes": boxes,
    }


def export_exclusion_report(
    canonical_manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Generate an auditable JSON and Markdown exclusion report for disputed clone groups."""
    cpath = Path(canonical_manifest_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cpath)
    total_canonical_count = len(df)

    disputed_df = df[df["canonical_id"].isin(DISPUTED_CLONE_CANONICAL_IDS)].copy()

    # Pre-exclusion vs post-exclusion counts per split
    split_counts_pre = df["split"].value_counts().to_dict()
    df_clean = df[~df["canonical_id"].isin(DISPUTED_CLONE_CANONICAL_IDS)]
    split_counts_post = df_clean["split"].value_counts().to_dict()

    retained_candidates_ids = ["CANON_0382", "CANON_3487", "CANON_1242", "CANON_3205"]
    retained_df = df[df["canonical_id"].isin(retained_candidates_ids)]

    report = {
        "exclusion_policy": "DISPUTED_LABEL_CLONES_PRIMARY_EXCLUSION",
        "rationale": (
            "Four duplicate groups contain exact/recompression pixel clones with conflicting "
            "binary fracture labels across dataset folders. To prevent contaminated supervision, "
            "these four groups are excluded from the primary benchmark while preserving the original files."
        ),
        "total_canonical_records_before": total_canonical_count,
        "total_canonical_records_after": len(df_clean),
        "excluded_disputed_count": len(disputed_df),
        "split_counts_before": {str(k): int(v) for k, v in split_counts_pre.items()},
        "split_counts_after": {str(k): int(v) for k, v in split_counts_post.items()},
        "excluded_groups": disputed_df[[
            "canonical_id",
            "representative_image_id",
            "all_filenames",
            "split",
            "fractured_label",
        ]].to_dict(orient="records"),
        "retained_candidates_policy": {
            "status": "RETAINED_SEPARATE",
            "rationale": (
                "The pairs CANON_0382/CANON_3487 and CANON_1242/CANON_3205 represent distinct "
                "radiograph acquisitions with differing pixel values (MSE > 200). They are retained "
                "as separate valid radiographs."
            ),
            "retained_records": retained_df[[
                "canonical_id",
                "representative_image_id",
                "split",
                "fractured_label",
            ]].to_dict(orient="records"),
        },
    }

    json_path = out_dir / "exclusions_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = out_dir / "exclusions_report.md"
    md_content = f"""# BoneTrace: Disputed Clone Groups Exclusion Report

**Policy:** `DISPUTED_LABEL_CLONES_PRIMARY_EXCLUSION`  
**Audited Cohort:** FracAtlas Canonical Manifest (`{cpath.name}`)  

## 1. Summary of Exclusions

| Split | Pre-Exclusion Count | Excluded Count | Post-Exclusion Count |
| :--- | :--- | :--- | :--- |
| **Train** | {split_counts_pre.get('train', 0)} | 1 (`CANON_3339`) | {split_counts_post.get('train', 0)} |
| **Validation** | {split_counts_pre.get('val', 0)} | 1 (`CANON_1668`) | {split_counts_post.get('val', 0)} |
| **Test** | {split_counts_pre.get('test', 0)} | 2 (`CANON_0206`, `CANON_0678`) | {split_counts_post.get('test', 0)} |
| **Total** | **{total_canonical_count}** | **4** | **{len(df_clean)}** |

## 2. Excluded Disputed Clone Records

The following 4 records exhibit verified conflicting labels between identical/recompressed pixel clones:

1. **CANON_0206** (Test split): `IMG0002886.jpg` (Fractured) vs `IMG0004243.jpg` (Non-fractured), MSE = 0.0
2. **CANON_0678** (Test split): `IMG0002952.jpg` (Non-fractured) vs `IMG0004262.jpg` (Fractured), MSE = 0.0767
3. **CANON_1668** (Val split): `IMG0003034.jpg` (Fractured) vs `IMG0004314.jpg` (Non-fractured), MSE = 0.0
4. **CANON_3339** (Train split): `IMG0004280.jpg` (Non-fractured) vs `IMG0002998.jpg` (Fractured), MSE = 0.0

## 3. Retained Candidates (`RETAIN_SEPARATE_CANDIDATE`)

The following candidate pairs were evaluated and verified as distinct radiographs:
- **`CANON_0382` (`IMG0004290.jpg`, label 0) & `CANON_3487` (`IMG0002979.jpg`, label 1)**: Distinct exposures (MSE = 907.4). Retained separate in training set.
- **`CANON_1242` (`IMG0002738.jpg`, label 1) & `CANON_3205` (`IMG0004121.jpg`, label 0)**: Distinct radiographs (MSE = 262.3). Retained separate in training set.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return report
