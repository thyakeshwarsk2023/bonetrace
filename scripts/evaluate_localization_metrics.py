"""scripts/evaluate_localization_metrics.py

Granular Grad-CAM Localization Metrics Benchmark for BoneTrace:
- Pointing Game Accuracy (%)
- Attribution Energy Inside Annotated Fracture Region (%)
- Heatmap-to-Annotation IoU (tau = 0.3, 0.5)

Evaluated strictly on the held-out fracture-positive test cohort (N = 100)
with 0.0% letterbox padding overlap (strictly within the valid anatomical FOV).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from bonetrace.config import load_config
from bonetrace.data.dataset import FracAtlasDataset
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.models.factory import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LocalizationEvaluation")


def evaluate_sample_localization(
    cam: np.ndarray,
    gt_mask: np.ndarray,
    fov_mask: np.ndarray,
    boxes: Union[np.ndarray, List[List[float]], torch.Tensor],
) -> Dict[str, float]:
    """Computes Pointing Game, Energy inside mask, and IoU at tau=0.3, 0.5."""
    # Ensure 2D (512, 512) shapes
    if cam.ndim == 3:
        cam = np.squeeze(cam)
    if gt_mask.ndim == 3:
        gt_mask = np.squeeze(gt_mask)
    if fov_mask.ndim == 3:
        fov_mask = np.squeeze(fov_mask)

    # Constrain CAM strictly to FOV (zero out letterbox padding)
    cam_fov = (cam * fov_mask).astype(np.float32)

    # 1. Pointing Game: Peak activation coordinate (y, x) within valid FOV
    peak_idx = int(np.argmax(cam_fov))
    peak_y, peak_x = np.unravel_index(peak_idx, cam_fov.shape)

    # Convert boxes to numpy array of shape (N, 4)
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    elif isinstance(boxes, (list, tuple)):
        boxes_np = np.array(boxes, dtype=np.float32)
    elif isinstance(boxes, np.ndarray):
        boxes_np = boxes
    else:
        boxes_np = np.empty((0, 4), dtype=np.float32)

    if boxes_np.ndim == 1 and len(boxes_np) == 4:
        boxes_np = boxes_np[np.newaxis, :]

    pointing_hit = 0.0
    # Check rasterized ground truth mask first
    if 0 <= peak_y < gt_mask.shape[0] and 0 <= peak_x < gt_mask.shape[1] and gt_mask[peak_y, peak_x] > 0.5:
        pointing_hit = 1.0
    elif len(boxes_np) > 0:
        # Check against bounding box bounds [x1, y1, x2, y2]
        for b in boxes_np:
            if len(b) >= 4:
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                if x1 <= float(peak_x) <= x2 and y1 <= float(peak_y) <= y2:
                    pointing_hit = 1.0
                    break

    # 2. Attribution Energy Inside Annotated Region (%)
    total_energy_fov = float(np.sum(cam_fov))
    if total_energy_fov > 1e-8:
        energy_inside = float(np.sum(cam_fov * (gt_mask > 0.5)) / total_energy_fov * 100.0)
    else:
        energy_inside = 0.0

    # 3. Heatmap-to-Box IoU at relative thresholds tau in {0.3, 0.5}
    max_val = float(np.max(cam_fov))
    binary_gt = (gt_mask > 0.5).astype(bool)

    ious: Dict[str, float] = {}
    for tau in [0.3, 0.5]:
        if max_val > 1e-8:
            threshold = tau * max_val
            binary_cam = (cam_fov >= threshold)
        else:
            binary_cam = np.zeros_like(cam_fov, dtype=bool)

        intersection = np.logical_and(binary_cam, binary_gt).sum()
        union = np.logical_or(binary_cam, binary_gt).sum()

        iou = float(intersection / union) if union > 0 else 0.0
        ious[f"iou_{int(tau * 100)}"] = iou

    return {
        "pointing_hit": pointing_hit,
        "energy_inside_pct": energy_inside,
        "iou_30": ious["iou_30"],
        "iou_50": ious["iou_50"],
    }


def run_evaluation(device_str: str = "cuda", baseline_weights: str | None = None, region_weights: str | None = None):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("Running Step 3 Localization Evaluation on device: %s", device)

    cfg = load_config()
    test_csv = cfg.paths.processed_data_dir / "test.csv"
    chk_baseline = Path(baseline_weights) if baseline_weights else (Path(cfg.paths.checkpoints_dir) / "baseline" / "best_model.pt")
    chk_region = Path(region_weights) if region_weights else (Path(cfg.paths.checkpoints_dir) / "region_supervised" / "best_model.pt")

    if not chk_baseline.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {chk_baseline}")
    if not chk_region.exists():
        raise FileNotFoundError(f"Region-supervised checkpoint not found: {chk_region}")

    # Load frozen held-out test split (excluding disputed canonical conflicts)
    dataset = FracAtlasDataset(manifest=test_csv, split="test", exclude_conflicts=True)
    n_total = len(dataset)
    logger.info("Test dataset loaded: %d total samples", n_total)

    # Filter strictly for fracture-positive samples with valid annotations (using dataset.records for instant metadata access)
    positive_indices = [
        i for i, r in enumerate(dataset.records)
        if int(r.get("fractured_label", 0)) == 1 and pd.notna(r.get("letterbox_target_boxes")) and r.get("letterbox_target_boxes") not in ["", "[]"]
    ]
    logger.info("Identified %d fracture-positive annotated test records.", len(positive_indices))

    # Initialize models and Grad-CAM extractors
    models: Dict[str, nn.Module] = {}
    gradcams: Dict[str, GradCAM] = {}
    for mtype, ckpt_path in [("baseline", chk_baseline), ("region_supervised", chk_region)]:
        m = build_model(config=cfg, model_type=mtype)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device)
        m.eval()
        models[mtype] = m
        gradcams[mtype] = GradCAM(m)

    results_baseline: List[Dict[str, float]] = []
    results_region: List[Dict[str, float]] = []

    for idx in tqdm(positive_indices, desc="Evaluating Localization Metrics", ncols=75):
        sample = dataset[idx]
        img_tensor = sample["image"].to(device)
        gt_mask = sample["mask"].squeeze().cpu().numpy()
        boxes = sample["boxes"]

        # Anatomical FOV mask (exclude black letterbox padding where channel 0 < -2.0)
        pad_mask = (img_tensor[0] < -2.0).cpu().numpy()
        fov_mask = (~pad_mask).astype(np.float32)

        # 1. Baseline Model
        cam_base = gradcams["baseline"].generate(img_tensor, class_idx=0)
        m_base = evaluate_sample_localization(cam_base, gt_mask, fov_mask, boxes)
        results_baseline.append(m_base)

        # 2. Region-Supervised Model
        cam_reg = gradcams["region_supervised"].generate(img_tensor, class_idx=0)
        m_reg = evaluate_sample_localization(cam_reg, gt_mask, fov_mask, boxes)
        results_region.append(m_reg)

    # Clean Grad-CAM hooks
    for g in gradcams.values():
        g.remove_hooks()

    # Aggregate summaries
    def aggregate(res_list: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        keys = ["pointing_hit", "energy_inside_pct", "iou_30", "iou_50"]
        out = {}
        for k in keys:
            vals = [r[k] for r in res_list]
            if k == "pointing_hit":
                out[k] = (float(np.mean(vals) * 100.0), 0.0)
            else:
                out[k] = (float(np.mean(vals)), float(np.std(vals)))
        return out

    agg_base = aggregate(results_baseline)
    agg_reg = aggregate(results_region)

    hit_diff = agg_reg["pointing_hit"][0] - agg_base["pointing_hit"][0]
    hit_rel = (hit_diff / max(agg_base["pointing_hit"][0], 1e-4)) * 100.0

    energy_diff = agg_reg["energy_inside_pct"][0] - agg_base["energy_inside_pct"][0]
    energy_rel = (energy_diff / max(agg_base["energy_inside_pct"][0], 1e-4)) * 100.0

    iou30_diff = agg_reg["iou_30"][0] - agg_base["iou_30"][0]
    iou30_rel = (iou30_diff / max(agg_base["iou_30"][0], 1e-4)) * 100.0

    iou50_diff = agg_reg["iou_50"][0] - agg_base["iou_50"][0]
    iou50_rel = (iou50_diff / max(agg_base["iou_50"][0], 1e-4)) * 100.0

    # Build Markdown Report
    report_md = fr"""# BoneTrace: Granular Localization Metrics Breakdown

**Cohort:** Held-Out Fracture-Positive Test Cohort ($N = {len(positive_indices)}$)  
**Target Layer:** `backbone.layer4[-1]`  
**Padding Exclusion:** 0.0% letterbox overlap (strictly constrained to anatomical FOV)

---

### Comparative Localization Summary

| Metric | Baseline (Model A) | Region-Supervised (Model B) | Absolute Shift | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Pointing Game Hit Rate** | {agg_base['pointing_hit'][0]:.2f}% | **{agg_reg['pointing_hit'][0]:.2f}%** | +{hit_diff:.2f}% | +{hit_rel:.1f}% |
| **Energy Inside Fracture BBox** | {agg_base['energy_inside_pct'][0]:.2f} ± {agg_base['energy_inside_pct'][1]:.2f}% | **{agg_reg['energy_inside_pct'][0]:.2f} ± {agg_reg['energy_inside_pct'][1]:.2f}%** | +{energy_diff:.2f}% | +{energy_rel:.1f}% |
| **Heatmap-to-Box IoU ($\tau = 0.3$)** | {agg_base['iou_30'][0]:.4f} ± {agg_base['iou_30'][1]:.4f} | **{agg_reg['iou_30'][0]:.4f} ± {agg_reg['iou_30'][1]:.4f}** | +{iou30_diff:.4f} | +{iou30_rel:.1f}% |
| **Heatmap-to-Box IoU ($\tau = 0.5$)** | {agg_base['iou_50'][0]:.4f} ± {agg_base['iou_50'][1]:.4f} | **{agg_reg['iou_50'][0]:.4f} ± {agg_reg['iou_50'][1]:.4f}** | +{iou50_diff:.4f} | +{iou50_rel:.1f}% |

### Interpretation
- **Pointing Game Hit Rate:** Fraction of cases where the single highest activation peak falls directly within the annotated fracture area.
- **Energy Inside Region ($E_{{in}}$):** Percentage of total post-hoc attribution concentrated on the pathology rather than non-pathological bone or surrounding tissue.
- **Heatmap-to-Box IoU ($\tau \in {{0.3, 0.5}}$):** Spatial overlap between the thresholded explanation and the ground-truth bounding box.
"""

    out_md_path = Path(cfg.paths.output_dir) / "evaluation" / "LOCALIZATION_REPORT.md"
    out_json_path = Path(cfg.paths.output_dir) / "evaluation" / "localization_metrics.json"
    out_md_path.parent.mkdir(parents=True, exist_ok=True)

    out_md_path.write_text(report_md, encoding="utf-8")

    json_data = {
        "n_samples": len(positive_indices),
        "baseline": {k: {"mean": agg_base[k][0], "std": agg_base[k][1]} for k in agg_base},
        "region_supervised": {k: {"mean": agg_reg[k][0], "std": agg_reg[k][1]} for k in agg_reg},
        "absolute_shifts": {
            "pointing_hit": hit_diff,
            "energy_inside_pct": energy_diff,
            "iou_30": iou30_diff,
            "iou_50": iou50_diff,
        },
    }
    out_json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    logger.info("Localization evaluation completed successfully.")
    logger.info("Markdown report saved to: %s", out_md_path)
    logger.info("JSON metrics saved to: %s", out_json_path)

    # Clean terminal table
    print("\n" + "=" * 80)
    print(" BINETRACE: GRANULAR LOCALIZATION METRICS (N = 100 FRACTURE POSITIVE)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Baseline':<18} | {'Region-Supervised':<18} | {'Shift':<10}")
    print("-" * 88)
    print(f"{'Pointing Game Hit Rate':<35} | {agg_base['pointing_hit'][0]:>16.2f}% | {agg_reg['pointing_hit'][0]:>16.2f}% | {hit_diff:>+9.2f}%")
    print(f"{'Energy Inside BBox (%)':<35} | {agg_base['energy_inside_pct'][0]:>16.2f}% | {agg_reg['energy_inside_pct'][0]:>16.2f}% | {energy_diff:>+9.2f}%")
    print(f"{'Heatmap-to-Box IoU (tau=0.3)':<35} | {agg_base['iou_30'][0]:>18.4f} | {agg_reg['iou_30'][0]:>18.4f} | {iou30_diff:>+10.4f}")
    print(f"{'Heatmap-to-Box IoU (tau=0.5)':<35} | {agg_base['iou_50'][0]:>18.4f} | {agg_reg['iou_50'][0]:>18.4f} | {iou50_diff:>+10.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Grad-CAM Localization Metrics")
    parser.add_argument("--device", type=str, default="cuda", help="Target computing device (cuda or cpu)")
    parser.add_argument("--baseline-weights", type=str, default=None, help="Custom path to baseline best_model.pt checkpoint")
    parser.add_argument("--region-weights", type=str, default=None, help="Custom path to region-supervised best_model.pt checkpoint")
    args = parser.parse_args()
    run_evaluation(args.device, args.baseline_weights, args.region_weights)