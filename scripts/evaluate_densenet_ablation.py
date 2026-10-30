#!/usr/bin/env python
"""
scripts/evaluate_densenet_ablation.py

Comprehensive Evaluation of DenseNet-121 Backbone Ablation Study for BoneTrace.
Strictly adheres to the audited Track B evaluation protocol:
1. Axis 1 — Classification Discrimination:
   - Test ROC-AUC, PR-AUC, Sensitivity, Specificity, PPV, NPV at p >= 0.50
   - Multi-seed reporting (Mean +- SD across seeds 42, 123, 2026)
2. Axis 2 — FOV-Constrained Perturbation Faithfulness & Area-Matched Lesion Dependence:
   - Restricted strictly to Omega_FOV (Channel 0 >= -2.0, 0.0% padding overlap)
   - Top-10% attributed drop (dy_attr), control drop (dy_ctrl), Causal Advantage (CA = dy_attr - dy_ctrl)
   - Evaluated on full test (N=609) and fracture-positive cohorts (N=107 / N=100)
   - Area-matched lesion dependence: masking GT box vs masking equal-area top-attributed pixels outside box
3. Axis 3 — Spatial Localization (N=107 / N=100 fracture-positive test cases):
   - Pointing Game Hit Rate (%) with Wilson 95% CIs
   - Energy inside ground-truth bounding box (%)
   - Bounding box IoU at tau = 0.3 and tau = 0.5
   - Evaluated for denseblock4 Grad-CAM (16x16), denseblock3 Grad-CAM (32x32), and Direct Head sigma(M_aux)
4. Axis 4 — Probability Calibration:
   - 10-bin equal-width ECE, 15-bin equal-mass (quantile) ECE, and Brier score
   - Temperature scaling: scalar T fitted on validation set (data/frozen/val.csv) and applied to test set
5. Artifacts & Outputs:
   - outputs/evaluation/densenet121_ablation_metrics.json
   - outputs/tables/T_ablation_densenet.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bonetrace.data.dataset import FracAtlasDataset, fracatlas_collate_fn
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.models.baseline import BaselineClassifier
from bonetrace.models.region_guided import RegionSupervisedModel

# Protocol Paths
FROZEN_DIR = REPO_ROOT / "data" / "frozen"
CHECKPOINTS_DIR = REPO_ROOT / "outputs" / "checkpoints"
EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"


def is_letterbox_padding(image_tensor: torch.Tensor) -> np.ndarray:
    """Identify black letterbox padding (Channel 0 < -2.0)."""
    return (image_tensor[0] < -2.0).cpu().numpy()


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculate Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return float(p), float(max(0.0, center - margin)), float(min(1.0, center + margin))


def compute_pointing_hit(peak_y: int, peak_x: int, boxes: list) -> float:
    """Check if argmax peak falls inside any bounding box."""
    for b in boxes:
        if len(b) == 4:
            x1, y1, x2, y2 = b
            if x1 <= peak_x <= x2 and y1 <= peak_y <= y2:
                return 1.0
    return 0.0


def compute_energy_inside(cam: np.ndarray, boxes: list) -> float:
    """Calculate percentage of heatmap energy inside ground-truth bounding boxes."""
    total_energy = float(cam.sum())
    if total_energy <= 0.0:
        return 0.0
    box_mask = np.zeros_like(cam, dtype=bool)
    for b in boxes:
        if len(b) == 4:
            x1, y1, x2, y2 = [int(round(c)) for c in b]
            box_mask[max(0, y1):min(512, y2), max(0, x1):min(512, x2)] = True
    inside_energy = float(cam[box_mask].sum())
    return float((inside_energy / total_energy) * 100.0)


def compute_iou_at_threshold(cam: np.ndarray, boxes: list, tau: float) -> float:
    """Calculate spatial IoU between binarized heatmap and ground-truth bounding boxes."""
    c_min, c_max = cam.min(), cam.max()
    if c_max > c_min:
        norm_cam = (cam - c_min) / (c_max - c_min)
    else:
        norm_cam = np.zeros_like(cam)
    cam_binary = (norm_cam >= tau)

    box_mask = np.zeros_like(cam, dtype=bool)
    for b in boxes:
        if len(b) == 4:
            x1, y1, x2, y2 = [int(round(c)) for c in b]
            box_mask[max(0, y1):min(512, y2), max(0, x1):min(512, x2)] = True

    intersection = np.sum(cam_binary & box_mask)
    union = np.sum(cam_binary | box_mask)
    return float(intersection / union) if union > 0 else 0.0


def compute_peak_to_box_distance(peak_y: int, peak_x: int, boxes: list) -> float:
    """Euclidean distance in pixels from CAM argmax to nearest box edge."""
    min_dist = float("inf")
    for b in boxes:
        if len(b) == 4:
            x1, y1, x2, y2 = b
            dx = max(0.0, x1 - peak_x, peak_x - x2)
            dy = max(0.0, y1 - peak_y, peak_y - y2)
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
    return float(min_dist if min_dist != float("inf") else 0.0)


def compute_equal_width_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Compute 10-bin equal-width Expected Calibration Error (ECE) and MCE."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high) if i < n_bins - 1 else (y_prob >= low) & (y_prob <= high)
        n_in_bin = int(np.sum(mask))

        if n_in_bin > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            gap = abs(acc - conf)
            ece += (n_in_bin / n) * gap
            if gap > mce:
                mce = gap

    return {"ece": float(ece), "mce": float(mce)}


def compute_equal_mass_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    """Compute 15-bin equal-mass (quantile) Expected Calibration Error (ECE)."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.percentile(y_prob, quantiles * 100))
    actual_bins = len(bin_edges) - 1
    if actual_bins < 1:
        return {"ece": 0.0, "mce": 0.0}

    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for i in range(actual_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high) if i < actual_bins - 1 else (y_prob >= low) & (y_prob <= high)
        n_in_bin = int(np.sum(mask))

        if n_in_bin > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            gap = abs(acc - conf)
            ece += (n_in_bin / n) * gap
            if gap > mce:
                mce = gap

    return {"ece": float(ece), "mce": float(mce)}


def fit_temperature_scaling(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fit optimal temperature scalar T > 0 on validation logits using NLL loss."""
    z = torch.from_numpy(val_logits.copy()).float()
    y = torch.from_numpy(val_labels.copy()).float()

    def nll_eval(T_val):
        T = float(T_val)
        if T <= 1e-4:
            return 1e6
        loss = F.binary_cross_entropy_with_logits(z / T, y)
        return float(loss.item())

    res = minimize_scalar(nll_eval, bounds=(0.05, 5.0), method="bounded")
    return float(res.x)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute discrimination metrics at decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    p, r, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(r, p))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def load_densenet_model(model_type: str, seed: int, device: torch.device) -> nn.Module:
    """Instantiate and load checkpoint weights for DenseNet-121 model."""
    ckpt_path = CHECKPOINTS_DIR / f"densenet121_{model_type}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if model_type == "region":
        model = RegionSupervisedModel(
            backbone="densenet121",
            pretrained=False,
            target_size=(512, 512),
            memory_efficient=False,
        )
    else:
        model = BaselineClassifier(
            backbone="densenet121",
            pretrained=False,
            memory_efficient=False,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_densenet_run(
    model: nn.Module,
    model_type: str,
    seed: int,
    test_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a single DenseNet checkpoint across all 4 benchmark axes."""
    print(f"\n--- Evaluating DenseNet-121 [{model_type.upper()} | Seed {seed}] ---", flush=True)

    # 1. Validation Predictions (for Temperature Scaling)
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for sample in val_samples:
            img = sample["image"].unsqueeze(0).to(device)
            out = model(img)
            logit = (out["logits"] if isinstance(out, dict) else out)[0, 0].item()
            val_logits.append(logit)
            val_labels.append(int(sample["label"].item()))
    val_logits = np.array(val_logits, dtype=np.float32)
    val_labels = np.array(val_labels, dtype=np.int64)

    # Fit Temperature T
    optimal_T = fit_temperature_scaling(val_logits, val_labels)

    # 2. Test Set Predictions
    test_logits = []
    test_labels = []
    test_cids = []
    test_boxes = []

    with torch.no_grad():
        for sample in test_samples:
            img = sample["image"].unsqueeze(0).to(device)
            out = model(img)
            logit = (out["logits"] if isinstance(out, dict) else out)[0, 0].item()
            test_logits.append(logit)
            test_labels.append(int(sample["label"].item()))
            test_cids.append(sample["canonical_id"])
            b = sample["boxes"].cpu().numpy() if isinstance(sample["boxes"], torch.Tensor) else np.array(sample["boxes"])
            test_boxes.append(b)

    test_logits = np.array(test_logits, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int64)
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))

    # Axis 1: Classification Discrimination
    cls_metrics = compute_classification_metrics(test_labels, test_probs, threshold=0.5)

    # Axis 4: Calibration
    cal_10 = compute_equal_width_calibration(test_labels, test_probs, n_bins=10)
    cal_15 = compute_equal_mass_calibration(test_labels, test_probs, n_bins=15)
    brier = float(brier_score_loss(test_labels, test_probs))

    # Post-Temperature Scaling Metrics
    test_probs_scaled = 1.0 / (1.0 + np.exp(-test_logits / optimal_T))
    cal_10_scaled = compute_equal_width_calibration(test_labels, test_probs_scaled, n_bins=10)
    brier_scaled = float(brier_score_loss(test_labels, test_probs_scaled))

    cal_metrics = {
        "ece_10": cal_10["ece"],
        "mce_10": cal_10["mce"],
        "ece_15_quantile": cal_15["ece"],
        "brier": brier,
        "temperature_T": optimal_T,
        "post_scaling_ece_10": cal_10_scaled["ece"],
        "post_scaling_brier": brier_scaled,
    }

    # Setup Grad-CAM extractors
    cam_db4 = GradCAM(model, target_layer=model.backbone.features.denseblock4, target_size=(512, 512))
    cam_db3 = GradCAM(model, target_layer=model.backbone.features.denseblock3, target_size=(512, 512))

    # Axis 2: FOV-Constrained Perturbation & Lesion Dependence
    perturbation_records = []
    lesion_dep_records = []

    # Axis 3: Spatial Localization records (for fracture positives)
    loc_records_db4 = []
    loc_records_db3 = []
    loc_records_aux = []

    print("Computing FOV-constrained perturbations and spatial localizations...", flush=True)

    for idx in tqdm(range(len(test_samples)), desc=f"{model_type[:3]}_s{seed} Eval", ncols=75):
        sample = test_samples[idx]
        img_t = sample["image"].to(device)
        y_true = test_labels[idx]
        boxes = test_boxes[idx]
        orig_logit = float(test_logits[idx])

        # Generate denseblock4 Grad-CAM for perturbation
        cam_4 = cam_db4.generate(img_t, class_idx=0)

        # FOV constraint (Channel 0 >= -2.0)
        pad_mask = is_letterbox_padding(img_t)
        fov_mask = ~pad_mask
        n_fov = int(fov_mask.sum())
        n_mask = max(1, int(round(n_fov * 0.10)))

        # Attribution Top-10% in FOV
        cam_attr = cam_4.copy()
        cam_attr[pad_mask] = -np.inf
        idx_attr = np.argsort(-cam_attr.flatten())[:n_mask]
        m_attr = np.zeros_like(cam_4, dtype=bool)
        m_attr[np.unravel_index(idx_attr, cam_4.shape)] = True

        # Control Bottom-10% in FOV
        cam_ctrl = cam_4.copy()
        cam_ctrl[pad_mask] = np.inf
        idx_ctrl = np.argsort(cam_ctrl.flatten())[:n_mask]
        m_ctrl = np.zeros_like(cam_4, dtype=bool)
        m_ctrl[np.unravel_index(idx_ctrl, cam_4.shape)] = True

        # Forward perturbed
        img_attr = img_t.clone()
        img_attr[:, torch.from_numpy(m_attr).to(device)] = 0.0

        img_ctrl = img_t.clone()
        img_ctrl[:, torch.from_numpy(m_ctrl).to(device)] = 0.0

        with torch.no_grad():
            out_attr = model(img_attr.unsqueeze(0))
            l_attr = float((out_attr["logits"] if isinstance(out_attr, dict) else out_attr)[0, 0].item())
            out_ctrl = model(img_ctrl.unsqueeze(0))
            l_ctrl = float((out_ctrl["logits"] if isinstance(out_ctrl, dict) else out_ctrl)[0, 0].item())

        dy_attr = orig_logit - l_attr
        dy_ctrl = orig_logit - l_ctrl
        ca = dy_attr - dy_ctrl

        perturbation_records.append({
            "idx": idx,
            "y_true": y_true,
            "dy_attr": dy_attr,
            "dy_ctrl": dy_ctrl,
            "causal_advantage": ca,
        })

        # Fracture-Positive Specific: Lesion Dependence & Localization
        if y_true == 1 and len(boxes) > 0:
            # --- Area-Matched Lesion Dependence ---
            m_gt_box = np.zeros_like(cam_4, dtype=bool)
            for b in boxes:
                if len(b) == 4:
                    x1, y1, x2, y2 = [int(round(coord)) for coord in b]
                    m_gt_box[max(0, y1):min(512, y2), max(0, x1):min(512, x2)] = True
            n_box_px = int(m_gt_box.sum())

            if n_box_px > 0:
                # 1. Mask GT Box
                img_box = img_t.clone()
                img_box[:, torch.from_numpy(m_gt_box).to(device)] = 0.0
                with torch.no_grad():
                    out_box = model(img_box.unsqueeze(0))
                    l_box = float((out_box["logits"] if isinstance(out_box, dict) else out_box)[0, 0].item())
                drop_gt_box = orig_logit - l_box

                # 2. Mask Area-Matched Top-Attributed Outside Box
                cam_outside = cam_4.copy()
                cam_outside[pad_mask | m_gt_box] = -np.inf
                idx_outside = np.argsort(-cam_outside.flatten())[:n_box_px]
                m_outside = np.zeros_like(cam_4, dtype=bool)
                m_outside[np.unravel_index(idx_outside, cam_4.shape)] = True

                img_outside = img_t.clone()
                img_outside[:, torch.from_numpy(m_outside).to(device)] = 0.0
                with torch.no_grad():
                    out_outside = model(img_outside.unsqueeze(0))
                    l_outside = float((out_outside["logits"] if isinstance(out_outside, dict) else out_outside)[0, 0].item())
                drop_outside = orig_logit - l_outside

                # 3. Mask Top-10% Attributed Minus Box
                m_attr_minus = m_attr & (~m_gt_box)
                img_minus = img_t.clone()
                img_minus[:, torch.from_numpy(m_attr_minus).to(device)] = 0.0
                with torch.no_grad():
                    out_minus = model(img_minus.unsqueeze(0))
                    l_minus = float((out_minus["logits"] if isinstance(out_minus, dict) else out_minus)[0, 0].item())
                drop_minus = orig_logit - l_minus

                lesion_dep_records.append({
                    "idx": idx,
                    "drop_gt_box": drop_gt_box,
                    "drop_outside_area_matched": drop_outside,
                    "drop_attr_minus_box": drop_minus,
                    "box_pixels": n_box_px,
                })

            # --- Spatial Localization: denseblock4 Grad-CAM ---
            peak_y4, peak_x4 = np.unravel_index(np.argmax(cam_4), cam_4.shape)
            hit4 = compute_pointing_hit(int(peak_y4), int(peak_x4), boxes)
            e_in4 = compute_energy_inside(cam_4, boxes)
            iou4_3 = compute_iou_at_threshold(cam_4, boxes, tau=0.3)
            iou4_5 = compute_iou_at_threshold(cam_4, boxes, tau=0.5)
            disp4 = compute_peak_to_box_distance(int(peak_y4), int(peak_x4), boxes)
            loc_records_db4.append({
                "hit": hit4,
                "energy_in_box": e_in4,
                "iou_03": iou4_3,
                "iou_05": iou4_5,
                "displacement_px": disp4,
            })

            # --- Spatial Localization: denseblock3 Grad-CAM ---
            cam_3 = cam_db3.generate(img_t, class_idx=0)
            peak_y3, peak_x3 = np.unravel_index(np.argmax(cam_3), cam_3.shape)
            hit3 = compute_pointing_hit(int(peak_y3), int(peak_x3), boxes)
            e_in3 = compute_energy_inside(cam_3, boxes)
            iou3_3 = compute_iou_at_threshold(cam_3, boxes, tau=0.3)
            iou3_5 = compute_iou_at_threshold(cam_3, boxes, tau=0.5)
            disp3 = compute_peak_to_box_distance(int(peak_y3), int(peak_x3), boxes)
            loc_records_db3.append({
                "hit": hit3,
                "energy_in_box": e_in3,
                "iou_03": iou3_3,
                "iou_05": iou3_5,
                "displacement_px": disp3,
            })

            # --- Spatial Localization: Direct Auxiliary Head sigma(M_aux) ---
            if model_type == "region":
                with torch.no_grad():
                    fwd_out = model(img_t.unsqueeze(0))
                    raw_mask = torch.sigmoid(fwd_out["mask_logits"]).squeeze().cpu().numpy()
                m_min, m_max = raw_mask.min(), raw_mask.max()
                norm_aux = (raw_mask - m_min) / (m_max - m_min + 1e-8) if m_max > m_min else np.zeros_like(raw_mask)
                peak_ya, peak_xa = np.unravel_index(np.argmax(norm_aux), norm_aux.shape)
                hita = compute_pointing_hit(int(peak_ya), int(peak_xa), boxes)
                e_ina = compute_energy_inside(norm_aux, boxes)
                ioua_3 = compute_iou_at_threshold(norm_aux, boxes, tau=0.3)
                ioua_5 = compute_iou_at_threshold(norm_aux, boxes, tau=0.5)
                dispa = compute_peak_to_box_distance(int(peak_ya), int(peak_xa), boxes)
                loc_records_aux.append({
                    "hit": hita,
                    "energy_in_box": e_ina,
                    "iou_03": ioua_3,
                    "iou_05": ioua_5,
                    "displacement_px": dispa,
                })

    cam_db4.remove_hooks()
    cam_db3.remove_hooks()

    # Aggregate Perturbation Metrics
    df_p = pd.DataFrame(perturbation_records)
    df_pos_p = df_p[df_p["y_true"] == 1]
    df_neg_p = df_p[df_p["y_true"] == 0]

    # Full Test (N=609)
    ca_full_mean = float(df_p["causal_advantage"].mean())
    dy_attr_full = float(df_p["dy_attr"].mean())
    dy_ctrl_full = float(df_p["dy_ctrl"].mean())

    # Fracture-Positive (N=107)
    ca_pos_mean = float(df_pos_p["causal_advantage"].mean())
    dy_attr_pos = float(df_pos_p["dy_attr"].mean())
    dy_ctrl_pos = float(df_pos_p["dy_ctrl"].mean())

    # Fracture-Positive Subset (First 100)
    df_pos_100 = df_pos_p.iloc[:100]
    ca_pos_100_mean = float(df_pos_100["causal_advantage"].mean())

    # Non-Fracture (N=502)
    ca_neg_mean = float(df_neg_p["causal_advantage"].mean())

    # Aggregate Lesion Dependence
    df_ld = pd.DataFrame(lesion_dep_records)
    drop_gt_box_mean = float(df_ld["drop_gt_box"].mean())
    drop_outside_mean = float(df_ld["drop_outside_area_matched"].mean())
    drop_minus_box_mean = float(df_ld["drop_attr_minus_box"].mean())

    # Aggregate Localization Metrics
    def summarize_loc(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {}
        hits = sum(r["hit"] for r in records)
        n = len(records)
        p, low, high = wilson_score_interval(int(hits), n)
        return {
            "n_samples": n,
            "hit_rate_pct": p * 100.0,
            "wilson_ci_95": [low * 100.0, high * 100.0],
            "energy_in_box_pct": float(np.mean([r["energy_in_box"] for r in records])),
            "iou_03": float(np.mean([r["iou_03"] for r in records])),
            "iou_05": float(np.mean([r["iou_05"] for r in records])),
            "median_displacement_px": float(np.median([r["displacement_px"] for r in records])),
        }

    loc_summary_db4 = summarize_loc(loc_records_db4)
    loc_summary_db3 = summarize_loc(loc_records_db3)
    loc_summary_aux = summarize_loc(loc_records_aux) if model_type == "region" else None

    # First 100 positives localization subset for exact N=100 comparison
    loc_summary_db4_100 = summarize_loc(loc_records_db4[:100])
    loc_summary_db3_100 = summarize_loc(loc_records_db3[:100])
    loc_summary_aux_100 = summarize_loc(loc_records_aux[:100]) if model_type == "region" else None

    return {
        "model_type": model_type,
        "seed": seed,
        "classification": cls_metrics,
        "calibration": cal_metrics,
        "perturbation": {
            "full_test_ca": ca_full_mean,
            "full_test_dy_attr": dy_attr_full,
            "full_test_dy_ctrl": dy_ctrl_full,
            "pos_ca": ca_pos_mean,
            "pos_dy_attr": dy_attr_pos,
            "pos_dy_ctrl": dy_ctrl_pos,
            "pos_100_ca": ca_pos_100_mean,
            "neg_ca": ca_neg_mean,
            "lesion_dependence": {
                "drop_gt_box": drop_gt_box_mean,
                "drop_outside_area_matched": drop_outside_mean,
                "drop_attr_minus_box": drop_minus_box_mean,
            },
        },
        "localization": {
            "denseblock4": loc_summary_db4,
            "denseblock3": loc_summary_db3,
            "aux_head": loc_summary_aux,
            "denseblock4_n100": loc_summary_db4_100,
            "denseblock3_n100": loc_summary_db3_100,
            "aux_head_n100": loc_summary_aux_100,
        },
    }


def aggregate_across_seeds(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and sample standard deviation across 3 seeds."""
    seeds = [r["seed"] for r in runs]
    agg: Dict[str, Any] = {"seeds": seeds}

    # Helper to compute mean +- std
    def get_stats(vals: List[float]) -> Tuple[float, float]:
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return m, s

    # 1. Classification
    cls_keys = ["roc_auc", "pr_auc", "sensitivity", "specificity", "ppv", "npv", "f1", "fp"]
    agg["classification"] = {}
    for k in cls_keys:
        vals = [r["classification"][k] for r in runs]
        m, s = get_stats(vals)
        agg["classification"][k] = {"mean": m, "std": s, "raw": vals}

    # 2. Calibration
    cal_keys = ["ece_10", "mce_10", "ece_15_quantile", "brier", "temperature_T", "post_scaling_ece_10", "post_scaling_brier"]
    agg["calibration"] = {}
    for k in cal_keys:
        vals = [r["calibration"][k] for r in runs]
        m, s = get_stats(vals)
        agg["calibration"][k] = {"mean": m, "std": s, "raw": vals}

    # 3. Perturbation
    pert_keys = ["full_test_ca", "full_test_dy_attr", "full_test_dy_ctrl", "pos_ca", "pos_dy_attr", "pos_dy_ctrl", "pos_100_ca", "neg_ca"]
    agg["perturbation"] = {}
    for k in pert_keys:
        vals = [r["perturbation"][k] for r in runs]
        m, s = get_stats(vals)
        agg["perturbation"][k] = {"mean": m, "std": s, "raw": vals}

    # Lesion dependence
    ld_keys = ["drop_gt_box", "drop_outside_area_matched", "drop_attr_minus_box"]
    agg["perturbation"]["lesion_dependence"] = {}
    for k in ld_keys:
        vals = [r["perturbation"]["lesion_dependence"][k] for r in runs]
        m, s = get_stats(vals)
        agg["perturbation"]["lesion_dependence"][k] = {"mean": m, "std": s, "raw": vals}

    # 4. Localization
    for branch in ["denseblock4", "denseblock3", "aux_head"]:
        if runs[0]["localization"][branch] is None:
            agg[f"loc_{branch}"] = None
            continue
        branch_metrics = ["hit_rate_pct", "energy_in_box_pct", "iou_03", "iou_05", "median_displacement_px"]
        agg[f"loc_{branch}"] = {}
        for k in branch_metrics:
            vals = [r["localization"][branch][k] for r in runs if r["localization"][branch] is not None]
            m, s = get_stats(vals)
            agg[f"loc_{branch}"][k] = {"mean": m, "std": s, "raw": vals}

    return agg


def generate_markdown_comparison_table(
    agg_base: Dict[str, Any],
    agg_reg: Dict[str, Any],
    raw_results: Dict[str, Any],
) -> str:
    """Construct publication-grade markdown comparison table."""
    def fmt_pct(m_dict: Dict[str, Any], key: str, mul100: bool = True) -> str:
        d = m_dict[key]
        mult = 100.0 if mul100 else 1.0
        return f"{d['mean'] * mult:.2f}% ± {d['std'] * mult:.2f}%"

    def fmt_num(m_dict: Dict[str, Any], key: str, decimals: int = 4) -> str:
        d = m_dict[key]
        return f"{d['mean']:.{decimals}f} ± {d['std']:.{decimals}f}"

    lines = []
    lines.append("# Table T_ablation_densenet: DenseNet-121 Controlled Backbone Ablation Study")
    lines.append("")
    lines.append("**Dataset:** Frozen FracAtlas Test Split ($N = 609$, 107 Positives, 502 Negatives).  ")
    lines.append("**Backbone:** DenseNet-121 (`torchvision.models.densenet121`) trained from scratch under identical matched protocol.  ")
    lines.append("**Seeds:** Evaluated across multi-seed matched runs (`seed42`, `seed123`, `seed2026`). Reported as $\\text{Mean} \\pm \\text{SD}$.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Axis 1 — Classification Discrimination Benchmark (Decision Threshold $p \\ge 0.50$)")
    lines.append("")
    lines.append("| Metric | DenseNet-121 Baseline (Arm A) | DenseNet-121 Region-Sup (Arm B) | Paired Gain (B - A) |")
    lines.append("| :--- | :---: | :---: | :---: |")

    # Classification rows
    c_b = agg_base["classification"]
    c_r = agg_reg["classification"]
    lines.append(f"| **ROC-AUC** | {fmt_num(c_b, 'roc_auc')} | {fmt_num(c_r, 'roc_auc')} | {c_r['roc_auc']['mean'] - c_b['roc_auc']['mean']:+.4f} |")
    lines.append(f"| **PR-AUC** | {fmt_num(c_b, 'pr_auc')} | {fmt_num(c_r, 'pr_auc')} | {c_r['pr_auc']['mean'] - c_b['pr_auc']['mean']:+.4f} |")
    lines.append(f"| **Sensitivity (Recall)** | {fmt_pct(c_b, 'sensitivity')} | {fmt_pct(c_r, 'sensitivity')} | {(c_r['sensitivity']['mean'] - c_b['sensitivity']['mean'])*100:+.2f}% |")
    lines.append(f"| **Specificity** | {fmt_pct(c_b, 'specificity')} | {fmt_pct(c_r, 'specificity')} | {(c_r['specificity']['mean'] - c_b['specificity']['mean'])*100:+.2f}% |")
    lines.append(f"| **Positive Predictive Value (PPV)** | {fmt_pct(c_b, 'ppv')} | {fmt_pct(c_r, 'ppv')} | {(c_r['ppv']['mean'] - c_b['ppv']['mean'])*100:+.2f}% |")
    lines.append(f"| **Negative Predictive Value (NPV)** | {fmt_pct(c_b, 'npv')} | {fmt_pct(c_r, 'npv')} | {(c_r['npv']['mean'] - c_b['npv']['mean'])*100:+.2f}% |")
    lines.append(f"| **F1 Score** | {fmt_num(c_b, 'f1')} | {fmt_num(c_r, 'f1')} | {c_r['f1']['mean'] - c_b['f1']['mean']:+.4f} |")
    lines.append(f"| **False Positive Count (out of 502)** | {fmt_num(c_b, 'fp', 1)} | {fmt_num(c_r, 'fp', 1)} | {c_r['fp']['mean'] - c_b['fp']['mean']:+.1f} ({(c_r['fp']['mean'] - c_b['fp']['mean'])/max(1e-5, c_b['fp']['mean'])*100:+.1f}%) |")
    lines.append("")

    # Perturbation rows
    lines.append("## 2. Axis 2 — FOV-Constrained Perturbation Faithfulness & Lesion Dependence")
    lines.append("")
    lines.append("Restricted strictly to anatomical Field of View $\\Omega_{\\text{FOV}}$ (Channel 0 $\\ge -2.0$, 0.0% padding overlap).")
    lines.append("")
    lines.append("| Perturbation Setting & Cohort | Baseline (Arm A) | Region-Sup (Arm B) | Difference (B - A) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    p_b = agg_base["perturbation"]
    p_r = agg_reg["perturbation"]
    lbl_attr = r"$\Delta y_{\text{attr}}$"
    lbl_ctrl = r"$\Delta y_{\text{ctrl}}$"
    lbl_ca = r"$\text{CA}$"
    lbl_box = r"$\Delta y_{\text{box}}$"

    lines.append(f"| **Full Test (N=609): Top-10% Drop ({lbl_attr})** | {fmt_num(p_b, 'full_test_dy_attr')} | {fmt_num(p_r, 'full_test_dy_attr')} | {p_r['full_test_dy_attr']['mean'] - p_b['full_test_dy_attr']['mean']:+.4f} |")
    lines.append(f"| **Full Test (N=609): Bottom-10% Drop ({lbl_ctrl})** | {fmt_num(p_b, 'full_test_dy_ctrl')} | {fmt_num(p_r, 'full_test_dy_ctrl')} | {p_r['full_test_dy_ctrl']['mean'] - p_b['full_test_dy_ctrl']['mean']:+.4f} |")
    lines.append(f"| **Full Test (N=609): Causal Advantage ({lbl_ca})** | {fmt_num(p_b, 'full_test_ca')} | {fmt_num(p_r, 'full_test_ca')} | {p_r['full_test_ca']['mean'] - p_b['full_test_ca']['mean']:+.4f} |")
    lines.append(f"| **Fracture-Positive (N=107): Causal Advantage ({lbl_ca})** | {fmt_num(p_b, 'pos_ca')} | {fmt_num(p_r, 'pos_ca')} | {p_r['pos_ca']['mean'] - p_b['pos_ca']['mean']:+.4f} |")
    lines.append(f"| **Fracture-Positive (N=100 subset): Causal Advantage** | {fmt_num(p_b, 'pos_100_ca')} | {fmt_num(p_r, 'pos_100_ca')} | {p_r['pos_100_ca']['mean'] - p_b['pos_100_ca']['mean']:+.4f} |")
    lines.append(f"| **Non-Fracture Controls (N=502): Causal Advantage** | {fmt_num(p_b, 'neg_ca')} | {fmt_num(p_r, 'neg_ca')} | {p_r['neg_ca']['mean'] - p_b['neg_ca']['mean']:+.4f} |")

    # Lesion dependence rows
    ld_b = p_b["lesion_dependence"]
    ld_r = p_r["lesion_dependence"]
    lines.append(f"| **Lesion Mask Drop: GT Fracture Box ({lbl_box})** | {fmt_num(ld_b, 'drop_gt_box')} | {fmt_num(ld_r, 'drop_gt_box')} | {ld_r['drop_gt_box']['mean'] - ld_b['drop_gt_box']['mean']:+.4f} |")
    lines.append(f"| **Area-Matched Drop: Top-Attributed Outside Box** | {fmt_num(ld_b, 'drop_outside_area_matched')} | {fmt_num(ld_r, 'drop_outside_area_matched')} | {ld_r['drop_outside_area_matched']['mean'] - ld_b['drop_outside_area_matched']['mean']:+.4f} |")
    lines.append(f"| **Drop: Top-10% Attributed Minus GT Box** | {fmt_num(ld_b, 'drop_attr_minus_box')} | {fmt_num(ld_r, 'drop_attr_minus_box')} | {ld_r['drop_attr_minus_box']['mean'] - ld_b['drop_attr_minus_box']['mean']:+.4f} |")
    lines.append("")

    # Localization rows
    lines.append("## 3. Axis 3 — Spatial Localization Performance (Fracture-Positive Test Cases, $N=107$)")
    lines.append("")
    lines.append("| Explainer / Target Layer | Stride / Res | Pointing Game Hit Rate (%) | Energy Inside Box (%) | IoU ($\\tau=0.3$) | IoU ($\\tau=0.5$) | Median Peak Disp |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    lbl_aux = r"$\sigma(M_{\text{aux}})$"
    # denseblock4 baseline
    l4_b = agg_base["loc_denseblock4"]
    lines.append(f"| Baseline Grad-CAM (`denseblock4`) | 32 px ($16\\times 16$) | {fmt_pct(l4_b, 'hit_rate_pct', False)} | {fmt_pct(l4_b, 'energy_in_box_pct', False)} | {fmt_num(l4_b, 'iou_03')} | {fmt_num(l4_b, 'iou_05')} | {l4_b['median_displacement_px']['mean']:.1f} px |")
    # denseblock4 region
    l4_r = agg_reg["loc_denseblock4"]
    lines.append(f"| Region-Sup Grad-CAM (`denseblock4`) | 32 px ($16\\times 16$) | {fmt_pct(l4_r, 'hit_rate_pct', False)} | {fmt_pct(l4_r, 'energy_in_box_pct', False)} | {fmt_num(l4_r, 'iou_03')} | {fmt_num(l4_r, 'iou_05')} | {l4_r['median_displacement_px']['mean']:.1f} px |")
    # denseblock3 baseline
    l3_b = agg_base["loc_denseblock3"]
    lines.append(f"| Baseline Grad-CAM (`denseblock3`) | 16 px ($32\\times 32$) | {fmt_pct(l3_b, 'hit_rate_pct', False)} | {fmt_pct(l3_b, 'energy_in_box_pct', False)} | {fmt_num(l3_b, 'iou_03')} | {fmt_num(l3_b, 'iou_05')} | {l3_b['median_displacement_px']['mean']:.1f} px |")
    # denseblock3 region
    l3_r = agg_reg["loc_denseblock3"]
    lines.append(f"| Region-Sup Grad-CAM (`denseblock3`) | 16 px ($32\\times 32$) | {fmt_pct(l3_r, 'hit_rate_pct', False)} | {fmt_pct(l3_r, 'energy_in_box_pct', False)} | {fmt_num(l3_r, 'iou_03')} | {fmt_num(l3_r, 'iou_05')} | {l3_r['median_displacement_px']['mean']:.1f} px |")
    # aux head region
    if agg_reg["loc_aux_head"] is not None:
        la_r = agg_reg["loc_aux_head"]
        lines.append(f"| **Direct Aux Head {lbl_aux} (Arm B)** | 1 px ($512\\times 512$) | **{fmt_pct(la_r, 'hit_rate_pct', False)}** | {fmt_pct(la_r, 'energy_in_box_pct', False)} | **{fmt_num(la_r, 'iou_03')}** | **{fmt_num(la_r, 'iou_05')}** | **{la_r['median_displacement_px']['mean']:.1f} px** |")
    lines.append("")

    # Calibration rows
    lines.append("## 4. Axis 4 — Probability Calibration Benchmark (Test Split $N=609$)")
    lines.append("")
    lines.append("| Calibration Metric | Baseline (Arm A) | Region-Sup (Arm B) | Difference (B - A) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    k_b = agg_base["calibration"]
    k_r = agg_reg["calibration"]
    lines.append(f"| **Unscaled 10-Bin ECE** | {fmt_num(k_b, 'ece_10')} | {fmt_num(k_r, 'ece_10')} | {k_r['ece_10']['mean'] - k_b['ece_10']['mean']:+.4f} |")
    lines.append(f"| **Quantile 15-Bin Equal-Mass ECE** | {fmt_num(k_b, 'ece_15_quantile')} | {fmt_num(k_r, 'ece_15_quantile')} | {k_r['ece_15_quantile']['mean'] - k_b['ece_15_quantile']['mean']:+.4f} |")
    lines.append(f"| **Maximum Calibration Error (MCE)** | {fmt_num(k_b, 'mce_10')} | {fmt_num(k_r, 'mce_10')} | {k_r['mce_10']['mean'] - k_b['mce_10']['mean']:+.4f} |")
    lines.append(f"| **Brier Score** | {fmt_num(k_b, 'brier')} | {fmt_num(k_r, 'brier')} | {k_r['brier']['mean'] - k_b['brier']['mean']:+.4f} |")
    lines.append(f"| **Validation Temperature Scalar $T$** | {fmt_num(k_b, 'temperature_T')} | {fmt_num(k_r, 'temperature_T')} | {k_r['temperature_T']['mean'] - k_b['temperature_T']['mean']:+.4f} |")
    lines.append(f"| **Post-Temperature Scaling 10-Bin ECE** | {fmt_num(k_b, 'post_scaling_ece_10')} | {fmt_num(k_r, 'post_scaling_ece_10')} | {k_r['post_scaling_ece_10']['mean'] - k_b['post_scaling_ece_10']['mean']:+.4f} |")
    lines.append(f"| **Post-Temperature Scaling Brier Score** | {fmt_num(k_b, 'post_scaling_brier')} | {fmt_num(k_r, 'post_scaling_brier')} | {k_r['post_scaling_brier']['mean'] - k_b['post_scaling_brier']['mean']:+.4f} |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Main execution engine."""
    parser = argparse.ArgumentParser(description="DenseNet-121 Backbone Ablation Evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    args = parser.parse_args()

    device = torch.device(args.device)
    print("\n" + "=" * 75)
    print("BONETRACE DENSENET-121 ABLATION EVALUATION ENGINE")
    print("=" * 75)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Seeds:  {args.seeds}")
    print("=" * 75)

    test_manifest = FROZEN_DIR / "test.csv"
    val_manifest = FROZEN_DIR / "val.csv"

    # Load and cache datasets in RAM for high performance
    print("Loading test and validation datasets into memory...", flush=True)
    t0 = time.time()
    test_ds = FracAtlasDataset(manifest=test_manifest, normalize=True)
    val_ds = FracAtlasDataset(manifest=val_manifest, normalize=True)

    test_samples = [test_ds[i] for i in range(len(test_ds))]
    val_samples = [val_ds[i] for i in range(len(val_ds))]
    print(f"Loaded {len(test_samples)} test samples and {len(val_samples)} val samples in {time.time()-t0:.1f}s.")

    raw_results = {"baseline": [], "region": []}

    # Evaluate Baseline runs
    for seed in args.seeds:
        model = load_densenet_model("baseline", seed, device)
        res = evaluate_densenet_run(model, "baseline", seed, test_samples, val_samples, device)
        raw_results["baseline"].append(res)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Evaluate Region-Supervised runs
    for seed in args.seeds:
        model = load_densenet_model("region", seed, device)
        res = evaluate_densenet_run(model, "region", seed, test_samples, val_samples, device)
        raw_results["region"].append(res)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate Multi-Seed Results
    print("\nSynthesizing multi-seed statistics across arms...", flush=True)
    agg_base = aggregate_across_seeds(raw_results["baseline"])
    agg_reg = aggregate_across_seeds(raw_results["region"])

    # Build Output Directories
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON metrics
    json_path = EVAL_DIR / "densenet121_ablation_metrics.json"
    full_output = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(device),
        "backbone": "densenet121",
        "seeds": args.seeds,
        "n_test_total": len(test_samples),
        "n_test_positives": sum(1 for s in test_samples if s["label"].item() == 1),
        "multi_seed_aggregate": {
            "baseline": agg_base,
            "region": agg_reg,
        },
        "raw_runs": raw_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"--> Saved raw JSON metrics to: {json_path}")

    # Generate Markdown Table
    md_content = generate_markdown_comparison_table(agg_base, agg_reg, raw_results)
    table_path = TABLES_DIR / "T_ablation_densenet.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"--> Saved publication markdown table to: {table_path}")

    # Print Table to Console
    print("\n" + "=" * 75)
    print("DENSENET-121 ABLATION BENCHMARK RESULTS")
    print("=" * 75)
    print(md_content)
    print("=" * 75 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
