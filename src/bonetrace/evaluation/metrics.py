"""Comprehensive Evaluation and Explanation Reliability Benchmarks for BoneTrace.

Implements rigorous evaluation for both classification and explanation quality:
1. Classification Metrics:
   - ROC-AUC, PR-AUC, Precision, Recall, F1, Specificity, Balanced Accuracy,
     Confusion Matrix, Brier Score, and Expected Calibration Error (ECE).
2. Explanation Localization Metrics:
   - Pointing Game Accuracy (peak activation hit rate within verified fracture annotations).
   - Energy Inside Annotated Region (E_in: fraction of heatmap attribution inside bounding boxes).
   - Heatmap-Bounding Box Intersection over Union (IoU).
3. Explanation Stability:
   - Heatmap Pearson correlation and cosine similarity under label-preserving Gaussian perturbations.
4. Comparative Reporting:
   - Controlled quantitative comparison between Baseline and Region-Supervised models.
   - Machine-readable JSON/CSV summaries and Markdown evaluation reports.

Scientific Constraints & Theoretical Definitions:
- Localization Agreement: Spatial congruence between explanation heatmaps and radiologist bounding boxes.
  Does NOT prove the model used those features causally.
- Stability: Invariance of explanation heatmaps under small input perturbations.
  Measures gradient robustness, not correctness.
- Faithfulness: The degree to which an attribution method reflects the true internal decision process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bonetrace.config import AppConfig
from bonetrace.data.dataset import FracAtlasDataset, fracatlas_collate_fn
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.logging_utils import get_logger
from bonetrace.models.factory import build_model

logger = get_logger("evaluation.metrics")


# ==============================================================================
# 1. Classification & Calibration Metrics
# ==============================================================================

def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width probability bins."""
    if len(y_true) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_classification_metrics(
    y_true: Union[Sequence[int], np.ndarray],
    y_prob: Union[Sequence[float], np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute complete set of binary classification and calibration metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities in [0.0, 1.0].
        threshold: Classification decision threshold.

    Returns:
        Dictionary of computed metrics.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    y_pred = (yp >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "sample_count": float(len(yt)),
        "pos_count": float(np.sum(yt == 1)),
        "neg_count": float(np.sum(yt == 0)),
    }

    # ROC-AUC & PR-AUC
    if len(np.unique(yt)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(yt, yp))
        except Exception:
            metrics["roc_auc"] = 0.5
        try:
            metrics["pr_auc"] = float(average_precision_score(yt, yp))
        except Exception:
            metrics["pr_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = 0.0

    # Precision, Recall, F1
    prec, rec, f1, _ = precision_recall_fscore_support(yt, y_pred, average="binary", zero_division=0)
    metrics["precision"] = float(prec)
    metrics["recall"] = float(rec)
    metrics["f1"] = float(f1)

    # Accuracy & Balanced Accuracy
    metrics["accuracy"] = float(np.mean(yt == y_pred)) if len(yt) > 0 else 0.0
    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(yt, y_pred))
    except Exception:
        metrics["balanced_accuracy"] = 0.5

    # Confusion matrix
    if len(yt) > 0:
        cm = confusion_matrix(yt, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["tn"] = float(tn)
        metrics["fp"] = float(fp)
        metrics["fn"] = float(fn)
        metrics["tp"] = float(tp)
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        metrics["tn"] = metrics["fp"] = metrics["fn"] = metrics["tp"] = metrics["specificity"] = 0.0

    # Calibration: Brier score & ECE
    if len(yt) > 0:
        metrics["brier_score"] = float(brier_score_loss(yt, yp))
        metrics["ece"] = compute_expected_calibration_error(yt, yp)
    else:
        metrics["brier_score"] = metrics["ece"] = 0.0

    return metrics


# ==============================================================================
# 2. Explanation Localization Metrics
# ==============================================================================

def pointing_game_hit(
    heatmap: np.ndarray,
    boxes: Sequence[Sequence[float]],
) -> float:
    """Calculate Pointing Game hit indicator for a single image.

    Identifies the peak activation coordinate (argmax of heatmap) and checks
    whether it resides within any ground-truth fracture bounding box.

    Args:
        heatmap: 2D array of shape (H, W).
        boxes: List of bounding boxes [xmin, ymin, xmax, ymax] in canvas pixels.

    Returns:
        1.0 if peak is inside any box, 0.0 otherwise.
    """
    if not boxes or len(boxes) == 0:
        return 0.0

    # Locate peak activation (row, col) = (y, x)
    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_y, peak_x = float(peak_idx[0]), float(peak_idx[1])

    for box in boxes:
        if len(box) != 4:
            continue
        xmin, ymin, xmax, ymax = box
        if xmin <= peak_x <= xmax and ymin <= peak_y <= ymax:
            return 1.0

    return 0.0


def energy_inside_boxes(
    heatmap: np.ndarray,
    boxes: Sequence[Sequence[float]],
    eps: float = 1e-8,
) -> float:
    """Calculate fraction of heatmap energy contained within ground-truth boxes (E_in).

    E_in = (sum of heatmap in boxes) / (total sum of heatmap + eps).

    Args:
        heatmap: 2D float array of shape (H, W) in [0.0, 1.0].
        boxes: List of bounding boxes [xmin, ymin, xmax, ymax].

    Returns:
        Float in [0.0, 1.0] representing energy concentration inside boxes.
    """
    total_energy = float(np.sum(heatmap))
    if total_energy <= eps or not boxes or len(boxes) == 0:
        return 0.0

    h, w = heatmap.shape
    box_mask = np.zeros((h, w), dtype=bool)

    for box in boxes:
        if len(box) != 4:
            continue
        xmin = max(0, min(int(round(box[0])), w))
        ymin = max(0, min(int(round(box[1])), h))
        xmax = max(0, min(int(round(box[2])), w))
        ymax = max(0, min(int(round(box[3])), h))
        box_mask[ymin:ymax, xmin:xmax] = True

    inside_energy = float(np.sum(heatmap[box_mask]))
    return float(inside_energy / (total_energy + eps))


def heatmap_box_iou(
    heatmap: np.ndarray,
    boxes: Sequence[Sequence[float]],
    threshold_ratio: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """Calculate Intersection over Union (IoU) between thresholded heatmap and boxes.

    Args:
        heatmap: 2D float array of shape (H, W).
        boxes: List of bounding boxes [xmin, ymin, xmax, ymax].
        threshold_ratio: Threshold multiplier applied to max(heatmap).

    Returns:
        Float IoU value in [0.0, 1.0].
    """
    if not boxes or len(boxes) == 0:
        return 0.0

    h, w = heatmap.shape
    box_mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        if len(box) != 4:
            continue
        xmin = max(0, min(int(round(box[0])), w))
        ymin = max(0, min(int(round(box[1])), h))
        xmax = max(0, min(int(round(box[2])), w))
        ymax = max(0, min(int(round(box[3])), h))
        box_mask[ymin:ymax, xmin:xmax] = True

    thresh = float(heatmap.max()) * threshold_ratio
    heat_mask = heatmap >= thresh

    intersection = np.logical_and(heat_mask, box_mask).sum()
    union = np.logical_or(heat_mask, box_mask).sum()

    if union == 0:
        return 0.0
    return float(intersection / (union + eps))


# ==============================================================================
# 3. Explanation Stability Metrics
# ==============================================================================

def compute_explanation_stability(
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    noise_std: float = 0.02,
    eps: float = 1e-8,
) -> float:
    """Evaluate Grad-CAM heatmap stability under label-preserving Gaussian perturbation.

    Computes the Pearson correlation between the original explanation H(I)
    and the perturbed explanation H(I + noise).

    Args:
        gradcam: Instantiated GradCAM extractor.
        image_tensor: Normalized input tensor of shape (3, H, W).
        noise_std: Standard deviation of additive Gaussian noise.

    Returns:
        Pearson correlation coefficient in [-1.0, 1.0].
    """
    h_orig = gradcam.generate(image_tensor, class_idx=0).flatten()

    # Apply subtle additive Gaussian noise
    noise = torch.randn_like(image_tensor) * noise_std
    perturbed_tensor = (image_tensor + noise).clamp(-3.0, 3.0)  # normalized range

    h_pert = gradcam.generate(perturbed_tensor, class_idx=0).flatten()

    # Pearson correlation
    h1 = h_orig - np.mean(h_orig)
    h2 = h_pert - np.mean(h_pert)
    denom = np.sqrt(np.sum(h1 ** 2) * np.sum(h2 ** 2)) + eps
    corr = float(np.sum(h1 * h2) / denom)
    return max(-1.0, min(1.0, corr))


# ==============================================================================
# 4. Explanation Faithfulness Metrics (Attribution-Guided Occlusion)
# ==============================================================================

def compute_attribution_occlusion_faithfulness(
    model: nn.Module,
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    top_k_ratio: float = 0.10,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate causal explanation faithfulness via attribution-guided region removal vs control.

    Measures the drop in the model's fracture logit when the top-k% most attributed
    pixels according to Grad-CAM are masked out, compared against the logit drop
    when an equal-area control region (the least-attributed pixels) is masked out.

    Faithfulness Criterion:
        delta_logit_attr > delta_logit_ctrl => positive causal faithfulness score.

    Args:
        model: PyTorch model (in eval mode).
        gradcam: GradCAM explanation extractor.
        image_tensor: Normalized image tensor of shape (3, H, W).
        top_k_ratio: Fraction of pixels to mask (default: 0.10 for top 10%).
        device: Computation device.

    Returns:
        Dictionary with:
        - 'orig_logit': Model output logit on unperturbed image.
        - 'delta_logit_attributed': Drop in fracture logit when masking top attribution region.
        - 'delta_logit_control': Drop in fracture logit when masking equal-area control region.
        - 'faithfulness_delta': delta_logit_attributed - delta_logit_control.
        - 'is_faithful': 1.0 if delta_logit_attributed > delta_logit_control else 0.0.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    inp = image_tensor.unsqueeze(0).to(device) if image_tensor.ndim == 3 else image_tensor.to(device)

    # 1. Unperturbed baseline forward pass
    with torch.no_grad():
        out_orig = model(inp)
        logits_orig = out_orig["logits"] if isinstance(out_orig, dict) else out_orig
        orig_logit = float(logits_orig[0, 0].item())

    # 2. Extract Grad-CAM attribution heatmap (H, W) in [0.0, 1.0]
    cam = gradcam.generate(image_tensor, class_idx=0)

    # Anatomical FOV constraint: exclude letterbox black padding (channel 0 < -2.0)
    pad_mask = (image_tensor[0] < -2.0).cpu().numpy()
    fov_mask = ~pad_mask
    n_fov = int(fov_mask.sum())
    n_mask = max(1, int(round((n_fov if n_fov > 0 else cam.size) * top_k_ratio)))

    cam_for_attr = cam.copy()
    cam_for_attr[pad_mask] = -np.inf  # Prevent sampling padding as top attribution

    cam_for_ctrl = cam.copy()
    cam_for_ctrl[pad_mask] = np.inf   # Prevent sampling padding as control baseline

    # 3. Top-k attribution mask (indices of highest attribution within FOV)
    sorted_desc = np.argsort(-cam_for_attr.flatten())
    attr_indices = sorted_desc[:n_mask]
    attr_mask_2d = np.zeros_like(cam, dtype=bool)
    attr_mask_2d[np.unravel_index(attr_indices, cam.shape)] = True

    # 4. Equal-area control mask: least-attributed pixels strictly within valid FOV
    sorted_asc = np.argsort(cam_for_ctrl.flatten())
    ctrl_indices = sorted_asc[:n_mask]
    ctrl_mask_2d = np.zeros_like(cam, dtype=bool)
    ctrl_mask_2d[np.unravel_index(ctrl_indices, cam.shape)] = True

    # 5. Mask attribution region (replace with 0.0, ImageNet mean in normalized space)
    img_attr = image_tensor.clone()
    mask_attr_t = torch.from_numpy(attr_mask_2d).to(img_attr.device, dtype=torch.bool)
    img_attr[:, mask_attr_t] = 0.0

    with torch.no_grad():
        inp_attr = img_attr.unsqueeze(0).to(device) if img_attr.ndim == 3 else img_attr.to(device)
        out_attr = model(inp_attr)
        logits_attr = out_attr["logits"] if isinstance(out_attr, dict) else out_attr
        attr_logit = float(logits_attr[0, 0].item())

    delta_logit_attr = orig_logit - attr_logit

    # 6. Mask control region (replace with 0.0, ImageNet mean in normalized space)
    img_ctrl = image_tensor.clone()
    mask_ctrl_t = torch.from_numpy(ctrl_mask_2d).to(img_ctrl.device, dtype=torch.bool)
    img_ctrl[:, mask_ctrl_t] = 0.0

    with torch.no_grad():
        inp_ctrl = img_ctrl.unsqueeze(0).to(device) if img_ctrl.ndim == 3 else img_ctrl.to(device)
        out_ctrl = model(inp_ctrl)
        logits_ctrl = out_ctrl["logits"] if isinstance(out_ctrl, dict) else out_ctrl
        ctrl_logit = float(logits_ctrl[0, 0].item())

    delta_logit_ctrl = orig_logit - ctrl_logit

    faithfulness_delta = delta_logit_attr - delta_logit_ctrl
    is_faithful = 1.0 if faithfulness_delta > 0.0 else 0.0

    return {
        "orig_logit": orig_logit,
        "delta_logit_attributed": float(delta_logit_attr),
        "delta_logit_control": float(delta_logit_ctrl),
        "faithfulness_delta": float(faithfulness_delta),
        "is_faithful": is_faithful,
    }


# ==============================================================================
# 5. End-to-End Evaluation Runner
# ==============================================================================

def evaluate_model_pipeline(
    model: nn.Module,
    dataset: FracAtlasDataset,
    device: torch.device,
    max_xai_samples: int = 50,
    sample_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Run full classification and XAI evaluation for a model on a dataset.

    Args:
        model: PyTorch neural network model.
        dataset: FracAtlasDataset instance.
        device: Execution device (cpu or cuda).
        max_xai_samples: Maximum number of annotated samples to evaluate XAI for.
        sample_indices: Optional explicit list of dataset indices to evaluate for XAI,
                        ensuring identical samples and settings across compared models.

    Returns:
        Structured evaluation metrics separated into classification, localization,
        stability, and faithfulness.
    """
    model.eval()
    model.to(device)

    all_probs: List[float] = []
    all_labels: List[int] = []

    # 1. Classification evaluation across all samples
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=fracatlas_collate_fn)
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            out = model(imgs)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits).cpu().flatten().tolist()
            lbls = batch["label"].cpu().flatten().long().tolist()
            all_probs.extend(probs)
            all_labels.extend(lbls)

    cls_metrics = compute_classification_metrics(all_labels, all_probs)

    # 2. XAI Localization, Stability, and Faithfulness evaluation
    gradcam = GradCAM(model=model, target_size=(512, 512))
    pointing_hits: List[float] = []
    energy_in_boxes: List[float] = []
    ious: List[float] = []
    stabilities: List[float] = []
    faith_attr_drops: List[float] = []
    faith_ctrl_drops: List[float] = []
    faith_deltas: List[float] = []
    faith_rates: List[float] = []

    # Determine indices to evaluate
    if sample_indices is not None:
        target_indices = list(sample_indices)
    else:
        target_indices = [
            i for i in range(len(dataset))
            if dataset[i]["has_annotation"].item()
        ][:max_xai_samples]

    try:
        for idx in target_indices:
            sample = dataset[idx]
            img = sample["image"]
            boxes = sample["boxes"].tolist()

            cam = gradcam.generate(img, class_idx=0)
            hit = pointing_game_hit(cam, boxes)
            e_in = energy_inside_boxes(cam, boxes)
            iou = heatmap_box_iou(cam, boxes)
            stab = compute_explanation_stability(gradcam, img, noise_std=0.02)
            faith = compute_attribution_occlusion_faithfulness(
                model=model, gradcam=gradcam, image_tensor=img, top_k_ratio=0.10, device=device
            )

            pointing_hits.append(hit)
            energy_in_boxes.append(e_in)
            ious.append(iou)
            stabilities.append(stab)
            faith_attr_drops.append(faith["delta_logit_attributed"])
            faith_ctrl_drops.append(faith["delta_logit_control"])
            faith_deltas.append(faith["faithfulness_delta"])
            faith_rates.append(faith["is_faithful"])
    finally:
        gradcam.remove_hooks()

    localization_metrics = {
        "xai_annotated_samples_evaluated": len(target_indices),
        "pointing_game_accuracy": float(np.mean(pointing_hits)) if pointing_hits else 0.0,
        "energy_inside_boxes_mean": float(np.mean(energy_in_boxes)) if energy_in_boxes else 0.0,
        "energy_inside_boxes_std": float(np.std(energy_in_boxes)) if energy_in_boxes else 0.0,
        "heatmap_iou_mean": float(np.mean(ious)) if ious else 0.0,
        "heatmap_iou_std": float(np.std(ious)) if ious else 0.0,
    }

    stability_metrics = {
        "explanation_stability_mean": float(np.mean(stabilities)) if stabilities else 0.0,
        "explanation_stability_std": float(np.std(stabilities)) if stabilities else 0.0,
    }

    faithfulness_metrics = {
        "delta_logit_attributed_mean": float(np.mean(faith_attr_drops)) if faith_attr_drops else 0.0,
        "delta_logit_attributed_std": float(np.std(faith_attr_drops)) if faith_attr_drops else 0.0,
        "delta_logit_control_mean": float(np.mean(faith_ctrl_drops)) if faith_ctrl_drops else 0.0,
        "delta_logit_control_std": float(np.std(faith_ctrl_drops)) if faith_ctrl_drops else 0.0,
        "faithfulness_delta_mean": float(np.mean(faith_deltas)) if faith_deltas else 0.0,
        "faithfulness_delta_std": float(np.std(faith_deltas)) if faith_deltas else 0.0,
        "faithfulness_rate": float(np.mean(faith_rates)) if faith_rates else 0.0,
    }

    return {
        "classification": cls_metrics,
        "localization_agreement": localization_metrics,
        "perturbation_stability": stability_metrics,
        "faithfulness": faithfulness_metrics,
        "localization_and_stability": {
            **localization_metrics,
            **stability_metrics,
        },
    }


def run_evaluation(config: AppConfig, allow_untrained: bool = False) -> Dict[str, Any]:
    """Execute evaluation benchmark comparing Baseline and Region-Supervised models.

    Args:
        config: Application configuration.
        allow_untrained: Whether to allow evaluating uninitialized weights when checkpoints
                         are absent (default: False; raises FileNotFoundError if missing).

    Returns:
        Structured evaluation comparison summary.

    Raises:
        FileNotFoundError: If test manifest or trained model checkpoints are missing.
    """
    logger.info("Starting BoneTrace comparative evaluation benchmark...")
    out_dir = Path(config.paths.evaluation_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_csv = config.paths.processed_data_dir / "test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"Test split manifest not found: {test_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() and getattr(config.project, "device", "cpu") == "cuda" else "cpu")
    logger.info("Evaluating on device: %s", device)

    # 1. Checkpoint Safety Verification
    baseline_ckpt = Path(config.paths.checkpoints_dir) / "baseline" / "best_model.pt"
    region_ckpt = Path(config.paths.checkpoints_dir) / "region_supervised" / "best_model.pt"

    if not allow_untrained:
        missing_ckpts = []
        if not baseline_ckpt.is_file():
            missing_ckpts.append(f"Baseline ('{baseline_ckpt}')")
        if not region_ckpt.is_file():
            missing_ckpts.append(f"Region-Supervised ('{region_ckpt}')")
        if missing_ckpts:
            raise FileNotFoundError(
                f"Evaluation aborted: trained model checkpoint(s) missing: {', '.join(missing_ckpts)}. "
                f"BoneTrace prohibits evaluating randomly initialized weights by default. "
                f"Please train models first using 'bonetrace train' or ensure checkpoints exist."
            )

    # 2. Load frozen held-out test split (excluding disputed canonical label conflicts)
    test_dataset = FracAtlasDataset(manifest=test_csv, split="test", exclude_conflicts=True)
    logger.info("Held-out test dataset loaded: %d samples", len(test_dataset))

    # 3. Establish identical test images for both models
    max_xai = 50
    annotated_indices = [
        idx for idx in range(len(test_dataset))
        if test_dataset[idx]["has_annotation"].item()
    ]
    if len(annotated_indices) > max_xai:
        annotated_indices = annotated_indices[:max_xai]

    logger.info(
        "Identified %d annotated test samples for controlled XAI evaluation across both models.",
        len(annotated_indices),
    )

    # 4. Evaluate Baseline Model
    logger.info("Building baseline model...")
    baseline_model = build_model(config=config, model_type="baseline")
    if baseline_ckpt.is_file():
        logger.info("Loading baseline checkpoint: %s", baseline_ckpt)
        ckpt = torch.load(baseline_ckpt, map_location=device)
        baseline_model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning("Baseline checkpoint not found at %s. Evaluating initialized model (allow_untrained=True).", baseline_ckpt)

    baseline_results = evaluate_model_pipeline(
        baseline_model, test_dataset, device=device, sample_indices=annotated_indices
    )

    # 5. Evaluate Region-Supervised Model
    logger.info("Building region-supervised model...")
    region_model = build_model(config=config, model_type="region_supervised")
    if region_ckpt.is_file():
        logger.info("Loading region-supervised checkpoint: %s", region_ckpt)
        ckpt = torch.load(region_ckpt, map_location=device)
        region_model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning("Region-supervised checkpoint not found at %s. Evaluating initialized model (allow_untrained=True).", region_ckpt)

    region_results = evaluate_model_pipeline(
        region_model, test_dataset, device=device, sample_indices=annotated_indices
    )

    # 6. Export machine-readable comparison JSON
    comparison_summary = {
        "held_out_test_set_size": len(test_dataset),
        "baseline": baseline_results,
        "region_supervised": region_results,
        "deliberate_scientific_distinctions": {
            "localization_agreement": (
                "Pointing Game Accuracy and Energy Inside Boxes measure spatial localization agreement "
                "with human annotations. High visual alignment does not establish causal faithfulness."
            ),
            "perturbation_stability": (
                "Stability measures robustness to subtle, label-preserving Gaussian input noise (sigma=0.02). "
                "High stability reflects smooth gradient attributions, not necessarily ground truth correctness."
            ),
            "causal_faithfulness": (
                "Faithfulness evaluates whether features identified by Grad-CAM causally drive the model's "
                "fracture prediction. Evaluated via attribution-guided region removal vs equal-area control removal."
            ),
        },
    }

    json_path = out_dir / "comparative_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # 7. Export Markdown comparison report
    b_cls = baseline_results["classification"]
    r_cls = region_results["classification"]
    b_loc = baseline_results["localization_agreement"]
    r_loc = region_results["localization_agreement"]
    b_stab = baseline_results["perturbation_stability"]
    r_stab = region_results["perturbation_stability"]
    b_faith = baseline_results["faithfulness"]
    r_faith = region_results["faithfulness"]

    md_report_path = out_dir / "EVALUATION_REPORT.md"
    md_content = f"""# BoneTrace: Controlled Comparative Evaluation Report

**Benchmark Cohort:** Held-out Test Set (`data/processed/test.csv`, N = {len(test_dataset)})  
**Device:** `{device}`  

## 1. Classification & Calibration Comparison

| Metric | Baseline (Classification-Only) | Region-Supervised (Dual-Head) |
| :--- | :--- | :--- |
| **ROC-AUC** | {b_cls.get('roc_auc', 0.0):.4f} | {r_cls.get('roc_auc', 0.0):.4f} |
| **PR-AUC** | {b_cls.get('pr_auc', 0.0):.4f} | {r_cls.get('pr_auc', 0.0):.4f} |
| **F1 Score** | {b_cls.get('f1', 0.0):.4f} | {r_cls.get('f1', 0.0):.4f} |
| **Balanced Accuracy** | {b_cls.get('balanced_accuracy', 0.0):.4f} | {r_cls.get('balanced_accuracy', 0.0):.4f} |
| **Sensitivity (Recall)** | {b_cls.get('recall', 0.0):.4f} | {r_cls.get('recall', 0.0):.4f} |
| **Specificity** | {b_cls.get('specificity', 0.0):.4f} | {r_cls.get('specificity', 0.0):.4f} |
| **Brier Score** | {b_cls.get('brier_score', 0.0):.4f} | {r_cls.get('brier_score', 0.0):.4f} |
| **Expected Calibration Error (ECE)** | {b_cls.get('ece', 0.0):.4f} | {r_cls.get('ece', 0.0):.4f} |

## 2. Explanation Localization Agreement Comparison

| Localization Metric | Baseline (Grad-CAM) | Region-Supervised (Grad-CAM) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Pointing Game Hit Rate** | {b_loc.get('pointing_game_accuracy', 0.0):.4f} | {r_loc.get('pointing_game_accuracy', 0.0):.4f} | Fraction of peaks inside ground truth fracture boxes |
| **Energy Inside Region ($E_{{in}}$)** | {b_loc.get('energy_inside_boxes_mean', 0.0):.4f} ± {b_loc.get('energy_inside_boxes_std', 0.0):.4f} | {r_loc.get('energy_inside_boxes_mean', 0.0):.4f} ± {r_loc.get('energy_inside_boxes_std', 0.0):.4f} | Heatmap attribution density within target box |
| **Heatmap BBox IoU** | {b_loc.get('heatmap_iou_mean', 0.0):.4f} ± {b_loc.get('heatmap_iou_std', 0.0):.4f} | {r_loc.get('heatmap_iou_mean', 0.0):.4f} ± {r_loc.get('heatmap_iou_std', 0.0):.4f} | Spatial overlap of thresholded heatmap with target box |

## 3. Explanation Perturbation Stability Comparison

| Stability Metric | Baseline (Grad-CAM) | Region-Supervised (Grad-CAM) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Perturbation Stability (Pearson $r$)** | {b_stab.get('explanation_stability_mean', 0.0):.4f} ± {b_stab.get('explanation_stability_std', 0.0):.4f} | {r_stab.get('explanation_stability_mean', 0.0):.4f} ± {r_stab.get('explanation_stability_std', 0.0):.4f} | Heatmap Pearson correlation under Gaussian noise (σ=0.02) |

## 4. Causal Explanation Faithfulness Comparison (Attribution-Guided Occlusion)

| Faithfulness Metric | Baseline (Grad-CAM) | Region-Supervised (Grad-CAM) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Attributed Region Logit Drop ($\\\\Delta y_{{attr}}$)** | {b_faith.get('delta_logit_attributed_mean', 0.0):.4f} ± {b_faith.get('delta_logit_attributed_std', 0.0):.4f} | {r_faith.get('delta_logit_attributed_mean', 0.0):.4f} ± {r_faith.get('delta_logit_attributed_std', 0.0):.4f} | Logit drop when masking top 10% Grad-CAM attribution |
| **Control Region Logit Drop ($\\\\Delta y_{{ctrl}}$)** | {b_faith.get('delta_logit_control_mean', 0.0):.4f} ± {b_faith.get('delta_logit_control_std', 0.0):.4f} | {r_faith.get('delta_logit_control_mean', 0.0):.4f} ± {r_faith.get('delta_logit_control_std', 0.0):.4f} | Logit drop when masking equal-area least-attributed region |
| **Faithfulness Advantage ($\\\\Delta y_{{attr}} - \\\\Delta y_{{ctrl}}$)** | {b_faith.get('faithfulness_delta_mean', 0.0):.4f} ± {b_faith.get('faithfulness_delta_std', 0.0):.4f} | {r_faith.get('faithfulness_delta_mean', 0.0):.4f} ± {r_faith.get('faithfulness_delta_std', 0.0):.4f} | Causal degradation advantage (higher = more faithful) |
| **Faithfulness Success Rate** | {b_faith.get('faithfulness_rate', 0.0):.4f} | {r_faith.get('faithfulness_rate', 0.0):.4f} | Proportion of samples where $\\\\Delta y_{{attr}} > \\\\Delta y_{{ctrl}}$ |

## 5. Scientific Distinctions: Localization vs Stability vs Faithfulness

1. **Localization Agreement $\\neq$ Faithfulness:**
   - Spatial agreement with annotations demonstrates that the feature map's gradient attribution concentrates within clinically verified fracture sites.
   - However, Grad-CAM is not inherently causal: an attribution map can highlight a region even if the network relies on spurious correlation cues or contextual bone geometry.
2. **Stability $\\neq$ Accuracy:**
   - Explanation stability evaluates gradient invariance under subtle input perturbations $\\mathcal{{N}}(0, \\sigma^2)$. High stability ensures explanations are not brittle noise artifacts, but does not guarantee clinical validity.
3. **Causal Faithfulness:**
   - Evaluated by attribution-guided region removal vs control. True faithfulness requires the model to actively rely on the highlighted features for its prediction ($\\\\Delta y_{{attr}} > \\\\Delta y_{{ctrl}}$).
4. **Controlled Experimental Design:**
   - Both models share the exact same ResNet-18 convolutional backbone stages, layer4 feature resolution, classification head formulation, and identical test image samples.
"""
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Comparative evaluation report saved to: %s", md_report_path)
    return comparison_summary
