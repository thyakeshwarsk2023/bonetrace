"""BoneTrace UI helper utilities for the Streamlit XAI Workstation.

Provides caching-aware model loading, image preprocessing, attribution map generation
(Grad-CAM final/intermediate, MultiScale/HiResCAM), auxiliary segmentation heads,
FOV masking, bounding box parsing/drawing, and dual-model causal perturbation simulation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile

# ── Ensure src/ and repo root are resolved reliably ───────────────────────────
def get_repo_root() -> Path:
    """Resolve repository root directory reliably across environments."""
    p = Path(__file__).resolve().parents[3]
    if (p / "outputs").is_dir():
        return p
    cwd = Path.cwd()
    if (cwd / "outputs").is_dir():
        return cwd
    return p

_REPO_ROOT = get_repo_root()
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from bonetrace.data.preprocess import LetterboxTransform
from bonetrace.models.factory import build_model
from bonetrace.explainability.gradcam import GradCAM, render_heatmap_overlay
from bonetrace.explainability.multiscale_cam import (
    MultiScaleAttributionEngine,
    compute_fov_mask,
    min_max_normalize_fov,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TARGET_SIZE   = (512, 512)   # (W, H)

CHECKPOINTS_DIR = _REPO_ROOT / "outputs" / "checkpoints"


# ── Dynamic Checkpoint Scanner & Path Resolution ──────────────────────────────
def scan_available_checkpoints(checkpoints_dir: Optional[Path] = None) -> Dict[str, List[int]]:
    """Scan outputs/checkpoints/ to identify existing backbone and seed combinations.

    Returns:
        Dict mapping backbone name to sorted list of seeds where both baseline
        and region checkpoints exist, or at least one exists.
        e.g. {'densenet121': [42, 123, 2026], 'resnet18': [123, 2026]}
    """
    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir else CHECKPOINTS_DIR
    if not ckpt_dir.is_dir():
        return {"densenet121": [42, 123, 2026], "resnet18": [123, 2026]}

    patterns = [
        # DenseNet-121 patterns
        (re.compile(r"^densenet121_baseline_seed(\d+)\.pt$", re.I), "densenet121", "baseline"),
        (re.compile(r"^densenet121_region(?:_supervised)?_seed(\d+)\.pt$", re.I), "densenet121", "region"),
        # ResNet-18 patterns
        (re.compile(r"^(?:resnet18_)?baseline_seed(\d+)\.pt$", re.I), "resnet18", "baseline"),
        (re.compile(r"^(?:resnet18_)?region(?:_supervised)?_seed(\d+)\.pt$", re.I), "resnet18", "region"),
    ]

    discovered: Dict[str, Dict[int, Dict[str, Path]]] = {}
    for f in ckpt_dir.glob("*.pt"):
        fname = f.name
        for pat, bb, mt in patterns:
            m = pat.match(fname)
            if m:
                s = int(m.group(1))
                discovered.setdefault(bb, {}).setdefault(s, {})[mt] = f
                break

    # Prioritize seeds where both baseline and region models exist
    result: Dict[str, List[int]] = {}
    for bb, seeds in discovered.items():
        both_exist = [s for s, models in seeds.items() if "baseline" in models and "region" in models]
        if both_exist:
            result[bb] = sorted(both_exist)
        else:
            result[bb] = sorted(seeds.keys())

    # Fallback to sensible defaults if empty
    if not result:
        return {"densenet121": [42, 123, 2026], "resnet18": [123, 2026]}

    return result


def get_checkpoint_path(backbone: str, model_type: str, seed: int) -> Optional[Path]:
    """Resolve checkpoint file path for a given backbone/model_type/seed combo.

    Checks primary and alternative file naming conventions.
    """
    bb_norm = backbone.lower().strip()
    mt_norm = "region" if ("region" in model_type.lower() or "aux" in model_type.lower()) else "baseline"

    candidates: List[Path] = []
    if bb_norm in ["densenet121", "densenet", "dense121"]:
        if mt_norm == "baseline":
            candidates.append(CHECKPOINTS_DIR / f"densenet121_baseline_seed{seed}.pt")
        else:
            candidates.append(CHECKPOINTS_DIR / f"densenet121_region_seed{seed}.pt")
            candidates.append(CHECKPOINTS_DIR / f"densenet121_region_supervised_seed{seed}.pt")
    elif bb_norm in ["resnet18", "resnet", "resnet-18"]:
        if mt_norm == "baseline":
            candidates.append(CHECKPOINTS_DIR / f"baseline_seed{seed}.pt")
            candidates.append(CHECKPOINTS_DIR / f"resnet18_baseline_seed{seed}.pt")
        else:
            candidates.append(CHECKPOINTS_DIR / f"region_supervised_seed{seed}.pt")
            candidates.append(CHECKPOINTS_DIR / f"region_seed{seed}.pt")
            candidates.append(CHECKPOINTS_DIR / f"resnet18_region_supervised_seed{seed}.pt")
            candidates.append(CHECKPOINTS_DIR / f"resnet18_region_seed{seed}.pt")

    for c in candidates:
        if c.is_file():
            return c

    # Return primary candidate path even if not yet on disk (for status checking)
    return candidates[0] if candidates else None


def load_model_for_ui(
    backbone: str,
    model_type: str,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Load and return a BoneTrace model from checkpoint.

    Args:
        backbone: 'resnet18' or 'densenet121'.
        model_type: 'baseline' or 'region'.
        seed: Checkpoint seed (e.g. 42, 123, 2026).
        device: Target torch.device.

    Returns:
        Loaded nn.Module in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist on disk.
    """
    ckpt_path = get_checkpoint_path(backbone, model_type, seed)
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found for backbone='{backbone}', model_type='{model_type}', seed={seed}. "
            f"Checked directory: {CHECKPOINTS_DIR}"
        )

    model = build_model(
        model_type=model_type,
        backbone=backbone,
        memory_efficient=False,  # inplace=False required for Grad-CAM
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# ── Image Loading & Preprocessing ─────────────────────────────────────────────
_letterbox = LetterboxTransform(target_size=TARGET_SIZE)


def load_and_preprocess(image_source: Union[str, Path, Image.Image]) -> Tuple[np.ndarray, torch.Tensor]:
    """Load an image from disk or PIL object; apply letterboxing and ImageNet normalization.

    Args:
        image_source: File path (str/Path) or a PIL.Image.Image.

    Returns:
        Tuple of:
          - display_rgb (np.ndarray, H×W×3, uint8 [0,255]): De-normalized display image.
          - img_tensor  (torch.Tensor, 3×H×W, float32): Normalized ImageNet tensor.
    """
    if isinstance(image_source, (str, Path)):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(str(image_source)) as raw:
            pil_img = raw.convert("RGB")
    else:
        pil_img = image_source.convert("RGB")

    letterboxed_pil, _ = _letterbox.apply_image(pil_img)

    # Raw tensor [0,1]
    raw_tensor = TF.to_tensor(letterboxed_pil)             # (3, H, W) float [0,1]

    # Display array: convert to uint8 before normalization
    display_rgb = (raw_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    # Normalized tensor for inference
    img_tensor = TF.normalize(raw_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return display_rgb, img_tensor


# ── FOV Mask ───────────────────────────────────────────────────────────────────
def get_fov_mask(img_tensor: torch.Tensor, threshold: float = -2.0) -> np.ndarray:
    """Return boolean FOV mask (H, W) numpy array.

    Letterbox black borders have channel-0 ≈ -2.118 < threshold=-2.0.
    """
    mask = (img_tensor[0] >= threshold).cpu().numpy()
    return mask


# ── Attribution Generation (Grad-CAM & Multi-Scale / HiResCAM) ──────────────────
def generate_attribution_map(
    model: nn.Module,
    img_tensor: torch.Tensor,
    method_or_layer: str = "final",
    fov_mask: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run attribution map extraction for a given method or layer.

    Args:
        model: Loaded BoneTrace model (eval mode).
        img_tensor: Normalized (3, H, W) tensor.
        method_or_layer:
            - 'final': Final stage Grad-CAM (denseblock4 / layer4, 16x16 stride-32).
            - 'intermediate': Intermediate stage Grad-CAM (denseblock3 / layer3, 32x32 stride-16).
            - 'multiscale': Multi-Scale / HiResCAM via MultiScaleAttributionEngine.
        fov_mask: If provided, zero out heatmap pixels outside FOV.
        device: Target inference device.

    Returns:
        np.ndarray (H, W) float32 in [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device

    backbone_name = getattr(model, "backbone_name", "resnet18")
    backbone = getattr(model, "backbone", None)

    if method_or_layer == "multiscale":
        # Use MultiScaleAttributionEngine for HiResCAM
        try:
            engine = MultiScaleAttributionEngine(model=model, device=device)
            x_b = img_tensor.unsqueeze(0).to(device)
            attrs = engine.compute_all_attributions(x_b)
            # Prioritize hirescam_layer3 or hirescam_layer4 or layercam
            if "hirescam_layer3" in attrs:
                cam = attrs["hirescam_layer3"].squeeze().cpu().numpy()
            elif "hirescam_layer4" in attrs:
                cam = attrs["hirescam_layer4"].squeeze().cpu().numpy()
            elif "layercam" in attrs:
                cam = attrs["layercam"].squeeze().cpu().numpy()
            else:
                cam = list(attrs.values())[0].squeeze().cpu().numpy()
        except Exception:
            # Fallback to standard final GradCAM
            gradcam = GradCAM(model=model, target_size=TARGET_SIZE)
            try:
                cam = gradcam.generate(img_tensor.to(device), class_idx=0)
            finally:
                gradcam.remove_hooks()
    else:
        target_layer: Optional[nn.Module] = None
        if method_or_layer == "intermediate":
            if backbone_name == "densenet121" and backbone is not None:
                target_layer = backbone.features.denseblock3
            elif backbone is not None and hasattr(backbone, "layer3"):
                target_layer = backbone.layer3[-1]

        gradcam = GradCAM(model=model, target_layer=target_layer, target_size=TARGET_SIZE)
        try:
            cam = gradcam.generate(img_tensor.to(device), class_idx=0)
        finally:
            gradcam.remove_hooks()

    if fov_mask is not None:
        cam = cam * fov_mask.astype(np.float32)

    # Normalize to [0, 1]
    c_min, c_max = float(cam.min()), float(cam.max())
    if c_max - c_min > 1e-8:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32)


def generate_aux_head_map(
    model: nn.Module,
    img_tensor: torch.Tensor,
    fov_mask: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """Generate auxiliary region head sigmoid map σ(M_aux) for RegionSupervisedModel.

    Args:
        model: RegionSupervisedModel instance.
        img_tensor: Normalized (3, H, W) tensor.
        fov_mask: If provided, zero out pixels outside FOV.
        device: Inference device.

    Returns:
        np.ndarray (H, W) float32 in [0, 1], or None if model has no aux head.
    """
    if not hasattr(model, "region_head"):
        return None

    if device is None:
        device = next(model.parameters()).device

    x = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)

    mask_logits = out.get("mask_logits")
    if mask_logits is None:
        return None

    sigma = torch.sigmoid(mask_logits).squeeze().cpu().numpy()  # (H, W)

    if fov_mask is not None:
        sigma = sigma * fov_mask.astype(np.float32)

    return sigma.astype(np.float32)


# ── Heatmap Overlay ────────────────────────────────────────────────────────────
def overlay_heatmap(
    display_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
    fov_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Blend a heatmap onto an RGB display image."""
    cam_masked = cam.copy()
    if fov_mask is not None:
        cam_masked = cam_masked * fov_mask.astype(np.float32)
    return render_heatmap_overlay(display_rgb, cam_masked, alpha=alpha, colormap_name=colormap)


# ── Annotation Drawing ─────────────────────────────────────────────────────────
def parse_boxes(boxes_raw: Any) -> List[List[float]]:
    """Parse bounding boxes from string or list to [[x1,y1,x2,y2], ...]."""
    if boxes_raw is None:
        return []
    if isinstance(boxes_raw, float) and np.isnan(boxes_raw):
        return []
    if isinstance(boxes_raw, str):
        boxes_raw = boxes_raw.strip()
        if not boxes_raw or boxes_raw in ["[]", "nan", ""]:
            return []
        try:
            parsed = json.loads(boxes_raw)
            if isinstance(parsed, list):
                return [b for b in parsed if isinstance(b, list) and len(b) == 4]
        except Exception:
            return []
    if isinstance(boxes_raw, list):
        return [b for b in boxes_raw if isinstance(b, list) and len(b) == 4]
    return []


def draw_boxes_on_image(
    img: np.ndarray,
    boxes: List[List[float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an image (RGB format)."""
    annotated = img.copy()
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = (int(round(c)) for c in box[:4])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    return annotated


def draw_fov_contour(
    img: np.ndarray,
    fov_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Overlay the FOV boundary contour on an image (RGB format)."""
    annotated = img.copy()
    mask_uint8 = (fov_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, color, thickness)
    return annotated


def add_peak_marker(
    img: np.ndarray,
    cam: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = (255, 255, 0),
    size: int = 12,
) -> np.ndarray:
    """Draw a '+' crosshair at the peak attribution pixel."""
    annotated = img.copy()
    masked_cam = cam.copy()
    if fov_mask is not None:
        masked_cam[~fov_mask] = -1.0
    peak_yx = np.unravel_index(np.argmax(masked_cam), masked_cam.shape)
    py, px = int(peak_yx[0]), int(peak_yx[1])
    cv2.line(annotated, (px - size, py), (px + size, py), color, 2)
    cv2.line(annotated, (px, py - size), (px, py + size), color, 2)
    return annotated


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Run forward pass and return (probability, logit)."""
    if device is None:
        device = next(model.parameters()).device
    x = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    logits_raw = out["logits"] if isinstance(out, dict) else out
    logit = float(logits_raw.squeeze().cpu())
    prob  = float(torch.sigmoid(logits_raw.squeeze()).cpu())
    return prob, logit


# ── Pointing Game Metrics ──────────────────────────────────────────────────────
def compute_pointing_game(
    cam: np.ndarray,
    boxes: List[List[float]],
    fov_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, float]:
    """Compute Pointing Game hit/miss and energy-inside-box %."""
    if not boxes:
        return False, 0.0

    masked_cam = cam.copy()
    if fov_mask is not None:
        masked_cam[~fov_mask] = -1.0

    peak_yx = np.unravel_index(np.argmax(masked_cam), masked_cam.shape)
    py, px = int(peak_yx[0]), int(peak_yx[1])

    # Pointing Game: peak inside any GT box
    hit = False
    for box in boxes:
        x1, y1, x2, y2 = (int(round(c)) for c in box[:4])
        if x1 <= px <= x2 and y1 <= py <= y2:
            hit = True
            break

    # Energy inside box
    box_mask = np.zeros(cam.shape, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = (int(round(c)) for c in box[:4])
        x1, x2 = max(0, x1), min(cam.shape[1] - 1, x2)
        y1, y2 = max(0, y1), min(cam.shape[0] - 1, y2)
        box_mask[y1:y2+1, x1:x2+1] = True

    total_energy = float(np.sum(np.maximum(cam, 0.0)))
    box_energy   = float(np.sum(np.maximum(cam[box_mask], 0.0)))
    energy_pct   = (box_energy / total_energy * 100.0) if total_energy > 1e-8 else 0.0

    return hit, energy_pct


# ── Causal Perturbation Simulation (FOV-Constrained) ───────────────────────────
def run_perturbation(
    model: nn.Module,
    img_tensor: torch.Tensor,
    cam: np.ndarray,
    k_pct: float = 10.0,
    fill_strategy: str = "zero",  # "zero", "blur", "mean"
    mode: str = "top_k_vs_random",  # "top_k_vs_random" or "box_vs_outside"
    boxes: Optional[List[List[float]]] = None,
    fov_mask: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run FOV-constrained causal perturbation experiment on a model.

    Supports:
      - Fill Strategies:
          * 'zero': Zero fill (ImageNet Mean = 0.0 in standardized space).
          * 'blur': Strong Gaussian blur of anatomical content.
          * 'mean': Average intensity within Omega_FOV.
      - Modes:
          * 'top_k_vs_random': Top-k% Attributed vs Equal-Area Random FOV Control.
          * 'box_vs_outside': Area-Matched GT Lesion Box vs Outside Context.
    """
    if device is None:
        device = next(model.parameters()).device

    # Original logit & prob
    prob_orig, logit_orig = run_inference(model, img_tensor, device)

    H, W = img_tensor.shape[1], img_tensor.shape[2]
    fov_np = fov_mask if fov_mask is not None else np.ones((H, W), dtype=bool)

    # Prepare fill replacement tensor
    if fill_strategy == "zero":
        # 0.0 in normalized space is exactly the ImageNet Mean
        fill_tensor = torch.zeros_like(img_tensor)
    elif fill_strategy == "blur":
        # Strong Gaussian blur applied to standardized image
        fill_tensor = TF.gaussian_blur(img_tensor, kernel_size=[31, 31], sigma=[10.0, 10.0])
    else:  # "mean"
        fov_t = torch.from_numpy(fov_np)
        mean_val = img_tensor[:, fov_t].mean(dim=1, keepdim=True).unsqueeze(-1)
        fill_tensor = mean_val.expand_as(img_tensor)

    def mask_and_infer(pixel_mask_2d: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Mask indicated pixels (True = mask) and run inference."""
        x_m = img_tensor.clone()
        m_t = torch.from_numpy(pixel_mask_2d)
        x_m[:, m_t] = fill_tensor[:, m_t]
        prob_m, logit_m = run_inference(model, x_m, device)

        # De-normalize for display
        disp = x_m.clone().cpu() * IMAGENET_STD + IMAGENET_MEAN
        disp_uint8 = (disp.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return prob_m, logit_m, disp_uint8

    if mode == "top_k_vs_random":
        cam_fov = cam.copy()
        cam_fov[~fov_np] = -np.inf
        fov_count = int(fov_np.sum())
        k_count = max(1, int(k_pct / 100.0 * fov_count))

        # Top-k% attributed pixels
        flat_idx = np.argpartition(cam_fov.ravel(), -k_count)[-k_count:]
        attr_mask = np.zeros(H * W, dtype=bool)
        attr_mask[flat_idx] = True
        attr_mask = attr_mask.reshape(H, W)

        # Equal-area random control within FOV (disjoint or independent)
        rng = np.random.default_rng(42)
        fov_indices = np.where(fov_np.ravel())[0]
        ctrl_flat = rng.choice(fov_indices, size=k_count, replace=False)
        ctrl_mask = np.zeros(H * W, dtype=bool)
        ctrl_mask[ctrl_flat] = True
        ctrl_mask = ctrl_mask.reshape(H, W)

    elif mode == "box_vs_outside" and boxes:
        # GT Box mask
        box_mask = np.zeros((H, W), dtype=bool)
        for box in boxes:
            x1, y1, x2, y2 = (int(round(c)) for c in box[:4])
            x1, x2 = max(0, x1), min(W - 1, x2)
            y1, y2 = max(0, y1), min(H - 1, y2)
            box_mask[y1:y2+1, x1:x2+1] = True
        attr_mask = box_mask & fov_np

        # Area-matched outside-box mask: top attribution pixels outside GT box
        n_box = int(attr_mask.sum())
        if n_box == 0:
            n_box = max(1, int(0.05 * fov_np.sum()))
        cam_outside = cam.copy()
        cam_outside[attr_mask | ~fov_np] = -np.inf
        flat_idx = np.argpartition(cam_outside.ravel(), -n_box)[-n_box:]
        ctrl_mask = np.zeros(H * W, dtype=bool)
        ctrl_mask[flat_idx] = True
        ctrl_mask = ctrl_mask.reshape(H, W)
    else:
        # Fallback to top_k
        cam_fov = cam.copy()
        cam_fov[~fov_np] = -np.inf
        fov_count = int(fov_np.sum())
        k_count = max(1, int(k_pct / 100.0 * fov_count))
        flat_idx = np.argpartition(cam_fov.ravel(), -k_count)[-k_count:]
        attr_mask = np.zeros(H * W, dtype=bool)
        attr_mask[flat_idx] = True
        attr_mask = attr_mask.reshape(H, W)
        rng = np.random.default_rng(42)
        fov_indices = np.where(fov_np.ravel())[0]
        ctrl_flat = rng.choice(fov_indices, size=k_count, replace=False)
        ctrl_mask = np.zeros(H * W, dtype=bool)
        ctrl_mask[ctrl_flat] = True
        ctrl_mask = ctrl_mask.reshape(H, W)

    prob_attr, logit_attr, img_masked_attr = mask_and_infer(attr_mask)
    prob_ctrl, logit_ctrl, img_masked_ctrl = mask_and_infer(ctrl_mask)

    dy_attr = logit_orig - logit_attr
    dy_ctrl = logit_orig - logit_ctrl
    causal_advantage = dy_attr - dy_ctrl

    return {
        "logit_orig": logit_orig,
        "prob_orig": prob_orig,
        "logit_attr": logit_attr,
        "prob_attr": prob_attr,
        "logit_ctrl": logit_ctrl,
        "prob_ctrl": prob_ctrl,
        "dy_attr": dy_attr,
        "dy_ctrl": dy_ctrl,
        "causal_advantage": causal_advantage,
        "img_masked_attr": img_masked_attr,
        "img_masked_ctrl": img_masked_ctrl,
        "attr_mask": attr_mask,
        "ctrl_mask": ctrl_mask,
    }


# ── Table / JSON helpers ───────────────────────────────────────────────────────
def load_markdown_table(path: Union[str, Path]) -> Optional[str]:
    """Read a Markdown table file; return content string or None if not found."""
    p = Path(path)
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8")


def load_json_safe(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load JSON file; return None on error."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
