"""BoneTrace UI helper utilities for the Streamlit XAI Workstation.

Provides caching-aware model loading, universal image preprocessing, attribution map generation
(Grad-CAM final/intermediate, MultiScale/HiResCAM), auxiliary segmentation heads,
automatic FOV masking, bounding box drawing, and dual-model causal perturbation simulation.
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
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


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DYNAMIC CHECKPOINT DISCOVERY & ZERO-CRASH LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def get_checkpoint_path(backbone: str, model_type: str, seed: int, checkpoints_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve checkpoint file path for a given backbone/model_type/seed combo.

    Checks primary and alternative file naming conventions.
    """
    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir else CHECKPOINTS_DIR
    bb_norm = backbone.lower().strip()
    mt_norm = "region" if ("region" in model_type.lower() or "aux" in model_type.lower()) else "baseline"

    candidates: List[Path] = []
    if bb_norm in ["densenet121", "densenet", "dense121"]:
        if mt_norm == "baseline":
            candidates.append(ckpt_dir / f"densenet121_baseline_seed{seed}.pt")
        else:
            candidates.append(ckpt_dir / f"densenet121_region_seed{seed}.pt")
            candidates.append(ckpt_dir / f"densenet121_region_supervised_seed{seed}.pt")
    elif bb_norm in ["resnet18", "resnet", "resnet-18"]:
        if mt_norm == "baseline":
            candidates.append(ckpt_dir / f"baseline_seed{seed}.pt")
            candidates.append(ckpt_dir / f"resnet18_baseline_seed{seed}.pt")
        else:
            candidates.append(ckpt_dir / f"region_supervised_seed{seed}.pt")
            candidates.append(ckpt_dir / f"region_seed{seed}.pt")
            candidates.append(ckpt_dir / f"resnet18_region_supervised_seed{seed}.pt")
            candidates.append(ckpt_dir / f"resnet18_region_seed{seed}.pt")

    for c in candidates:
        if c.is_file():
            return c

    return candidates[0] if candidates else None


def discover_available_model_pairs(checkpoints_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan checkpoints directory and find every valid (backbone, seed) pair where
    both Model A (Baseline) and Model B (Region-Supervised) exist on disk.

    Returns:
        List of dicts ordered by priority (DenseNet-121 Seed 42 first, followed by
        ResNet-18 Seed 123/2026):
          - 'id': str identifier (e.g. 'densenet121_42')
          - 'backbone': 'densenet121' or 'resnet18'
          - 'seed': int
          - 'label': Clean human-readable display string
          - 'baseline_path': Path
          - 'region_path': Path
    """
    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir else CHECKPOINTS_DIR
    if not ckpt_dir.is_dir():
        return []

    # File patterns
    patterns = [
        # DenseNet-121
        (re.compile(r"^densenet121_baseline_seed(\d+)\.pt$", re.I), "densenet121", "baseline"),
        (re.compile(r"^densenet121_region(?:_supervised)?_seed(\d+)\.pt$", re.I), "densenet121", "region"),
        # ResNet-18
        (re.compile(r"^(?:resnet18_)?baseline_seed(\d+)\.pt$", re.I), "resnet18", "baseline"),
        (re.compile(r"^(?:resnet18_)?region(?:_supervised)?_seed(\d+)\.pt$", re.I), "resnet18", "region"),
    ]

    discovered: Dict[str, Dict[int, Dict[str, Path]]] = {}
    for f in ckpt_dir.glob("*.pt"):
        for pat, bb, mt in patterns:
            m = pat.match(f.name)
            if m:
                s = int(m.group(1))
                discovered.setdefault(bb, {}).setdefault(s, {})[mt] = f
                break

    # Build valid pairs where both baseline and region exist
    valid_pairs: List[Dict[str, Any]] = []
    for bb, seeds in discovered.items():
        for s, models in seeds.items():
            if "baseline" in models and "region" in models:
                if bb == "densenet121":
                    lbl = f"DenseNet-121 — Seed {s}" + (" (Primary Audit Ablation)" if s == 42 else " (Multi-Seed)")
                else:
                    lbl = f"ResNet-18 — Seed {s}" + (" (Matched Multi-Seed)" if s == 123 else " (Audited Reference)")

                valid_pairs.append({
                    "id": f"{bb}_{s}",
                    "backbone": bb,
                    "seed": s,
                    "label": lbl,
                    "baseline_path": models["baseline"],
                    "region_path": models["region"],
                })

    # Strict priority ordering: DenseNet-121 seed 42 -> ResNet-18 seed 123 -> ResNet-18 seed 2026 -> others
    def priority_key(pair: Dict[str, Any]) -> int:
        bb, s = pair["backbone"], pair["seed"]
        if bb == "densenet121" and s == 42:
            return 0
        if bb == "resnet18" and s == 123:
            return 1
        if bb == "resnet18" and s == 2026:
            return 2
        if bb == "densenet121" and s == 123:
            return 3
        if bb == "densenet121" and s == 2026:
            return 4
        return 10 + s

    valid_pairs.sort(key=priority_key)
    return valid_pairs


def scan_available_checkpoints(checkpoints_dir: Optional[Path] = None) -> Dict[str, List[int]]:
    """Convenience wrapper returning dict of backbone -> sorted seeds."""
    pairs = discover_available_model_pairs(checkpoints_dir)
    res: Dict[str, List[int]] = {}
    for p in pairs:
        res.setdefault(p["backbone"], []).append(p["seed"])
    for bb in res:
        res[bb] = sorted(set(res[bb]))
    if not res:
        return {"densenet121": [42, 123, 2026], "resnet18": [123, 2026]}
    return res


def load_model_for_ui(
    backbone: str,
    model_type: str,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Load and return a BoneTrace model from checkpoint with robust key harmonization.

    Handles architectural state_dict variations ('backbone.fc' vs 'fc') cleanly.
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

    # Harmonize fc keys between submodule aliases
    if "backbone.fc.weight" in state and "fc.weight" not in state:
        state["fc.weight"] = state["backbone.fc.weight"]
        state["fc.bias"] = state["backbone.fc.bias"]
    elif "fc.weight" in state and "backbone.fc.weight" not in state:
        state["backbone.fc.weight"] = state["fc.weight"]
        state["backbone.fc.bias"] = state["fc.bias"]

    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: UNIVERSAL ARBITRARY IMAGE PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

_letterbox = LetterboxTransform(target_size=TARGET_SIZE)


def to_pil_rgb(image_source: Any) -> Image.Image:
    """Convert any arbitrary image source to an 8-bit RGB PIL Image.

    Supports:
      - File paths (str, Path)
      - Bytes, BytesIO, Streamlit UploadedFile
      - PIL Images (L, 1, I, F, RGB, RGBA, CMYK)
      - NumPy arrays (2D grayscale, 3D RGB/RGBA, uint8, uint16, float32)
      - 16-bit medical radiographs
    """
    # 1. Path or filename
    if isinstance(image_source, (str, Path)):
        p = Path(image_source)
        if not p.is_file():
            raise FileNotFoundError(f"Image file not found: {p}")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(str(p)) as raw:
            return _convert_pil_to_rgb(raw)

    # 2. File-like object / Streamlit UploadedFile / BytesIO
    if hasattr(image_source, "read"):
        try:
            image_source.seek(0)
        except Exception:
            pass
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        raw = Image.open(image_source)
        return _convert_pil_to_rgb(raw)

    # 3. NumPy Array
    if isinstance(image_source, np.ndarray):
        arr = image_source
        if arr.ndim == 2:
            # Grayscale 2D
            if arr.dtype == np.uint16:
                arr = (arr / 256.0).clip(0, 255).astype(np.uint8)
            elif arr.dtype in (np.float32, np.float64):
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L").convert("RGB")

        elif arr.ndim == 3:
            # 3D Array
            H, W, C = arr.shape
            if C == 1:
                arr_2d = arr[:, :, 0]
                if arr_2d.dtype == np.uint16:
                    arr_2d = (arr_2d / 256.0).clip(0, 255).astype(np.uint8)
                else:
                    arr_2d = arr_2d.clip(0, 255).astype(np.uint8)
                return Image.fromarray(arr_2d, mode="L").convert("RGB")
            elif C == 3:
                arr = arr.clip(0, 255).astype(np.uint8)
                return Image.fromarray(arr, mode="RGB")
            elif C == 4:
                # RGBA
                arr = arr.clip(0, 255).astype(np.uint8)
                pil_rgba = Image.fromarray(arr, mode="RGBA")
                return _convert_pil_to_rgb(pil_rgba)

    # 4. PIL Image
    if isinstance(image_source, Image.Image):
        return _convert_pil_to_rgb(image_source)

    raise TypeError(f"Unsupported image source type: {type(image_source)}")


def _convert_pil_to_rgb(img: Image.Image) -> Image.Image:
    """Helper to convert any PIL Image mode to RGB safely."""
    if img.mode == "RGB":
        return img.copy()

    # Handle 16-bit / integer modes
    if img.mode in ("I;16", "I", "F"):
        arr = np.array(img, dtype=np.float32)
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-6:
            arr = (arr - mn) / (mx - mn) * 255.0
        else:
            arr = np.zeros_like(arr)
        arr_u8 = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr_u8, mode="L").convert("RGB")

    # Handle RGBA with composite against black
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg

    return img.convert("RGB")


def load_and_preprocess(image_source: Any) -> Tuple[np.ndarray, torch.Tensor]:
    """Universal loader and preprocessor for arbitrary medical radiographs.

    Accepts any format, aspect ratio, or bit-depth, applies aspect-ratio-preserving
    512×512 letterboxing with neutral black borders, and returns normalized tensor.

    Returns:
        Tuple of:
          - display_rgb: (512, 512, 3) uint8 numpy array [0, 255]
          - img_tensor: (3, 512, 512) float32 ImageNet normalized torch.Tensor
    """
    pil_rgb = to_pil_rgb(image_source)
    letterboxed_pil, _ = _letterbox.apply_image(pil_rgb)

    # Raw tensor in [0, 1]
    raw_tensor = TF.to_tensor(letterboxed_pil)  # (3, 512, 512) float32

    # Display image in uint8
    display_rgb = (raw_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    # ImageNet standardized tensor
    img_tensor = TF.normalize(raw_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return display_rgb, img_tensor


# ── Automatic Anatomical FOV Guardrail ──────────────────────────────────────────
def get_fov_mask(img_tensor: torch.Tensor, threshold: float = -2.0) -> np.ndarray:
    """Compute anatomical Field of View mask (Ω_FOV) automatically on normalized tensor.

    Letterbox black borders have Channel-0 ≈ -2.118 < threshold=-2.0.
    Returns (512, 512) boolean numpy array.
    """
    mask = (img_tensor[0] >= threshold).cpu().numpy()
    if not mask.any():
        # Fallback if image was entirely black
        mask = np.ones((img_tensor.shape[1], img_tensor.shape[2]), dtype=bool)
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTION MAP EXTRACTION & AUXILIARY HEAD
# ═══════════════════════════════════════════════════════════════════════════════

def generate_attribution_map(
    model: nn.Module,
    img_tensor: torch.Tensor,
    method_or_layer: str = "final",
    fov_mask: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract attribution map for a given method or layer.

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
        try:
            engine = MultiScaleAttributionEngine(model=model, device=device)
            x_b = img_tensor.unsqueeze(0).to(device)
            attrs = engine.compute_all_attributions(x_b)
            if "hirescam_layer3" in attrs:
                cam = attrs["hirescam_layer3"].squeeze().cpu().numpy()
            elif "hirescam_layer4" in attrs:
                cam = attrs["hirescam_layer4"].squeeze().cpu().numpy()
            elif "layercam" in attrs:
                cam = attrs["layercam"].squeeze().cpu().numpy()
            else:
                cam = list(attrs.values())[0].squeeze().cpu().numpy()
        except Exception:
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

    # Min-max normalize
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
    """Generate auxiliary region head sigmoid probability map σ(M_aux).

    Returns (512, 512) float32 in [0, 1] or None if model has no auxiliary head.
    """
    if not hasattr(model, "region_head"):
        return None

    if device is None:
        device = next(model.parameters()).device

    x = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)

    mask_logits = out.get("mask_logits") if isinstance(out, dict) else None
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
    """Draw bounding boxes on an RGB image."""
    if not boxes:
        return img
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
    """Overlay the FOV boundary contour on an RGB image."""
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
    """Draw a '+' crosshair at the peak attribution pixel within FOV."""
    annotated = img.copy()
    masked_cam = cam.copy()
    if fov_mask is not None:
        masked_cam[~fov_mask] = -1.0
    if (masked_cam > -0.5).any():
        peak_yx = np.unravel_index(np.argmax(masked_cam), masked_cam.shape)
        py, px = int(peak_yx[0]), int(peak_yx[1])
        cv2.line(annotated, (px - size, py), (px + size, py), color, 2)
        cv2.line(annotated, (px, py - size), (px, py + size), color, 2)
    return annotated


# ── Forward Inference ──────────────────────────────────────────────────────────
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


# ── Pointing Game & Unsupervised Saliency Metrics ─────────────────────────────
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


def compute_unsupervised_metrics(
    cam: np.ndarray,
    aux_map: Optional[np.ndarray],
    fov_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute attribution confidence metrics for images without GT annotations.

    Returns:
        Dict with:
          - 'peak_cam_confidence': max Grad-CAM value in FOV
          - 'fov_concentration_pct': % of total energy contained in top 10% FOV pixels
          - 'aux_peak_activation': max σ(M_aux) probability in FOV (or 0.0)
    """
    masked_cam = cam.copy()
    masked_cam[~fov_mask] = 0.0
    total_energy = float(np.sum(masked_cam))

    # Top 10% concentration
    fov_pixels = int(fov_mask.sum())
    top_10_count = max(1, int(0.10 * fov_pixels))
    cam_fov_vals = cam[fov_mask]
    if len(cam_fov_vals) > 0:
        peak_cam = float(np.max(cam_fov_vals))
        sorted_vals = np.sort(cam_fov_vals)[::-1]
        top_energy = float(np.sum(sorted_vals[:top_10_count]))
        concentration = (top_energy / total_energy * 100.0) if total_energy > 1e-8 else 0.0
    else:
        peak_cam = 0.0
        concentration = 0.0

    # Aux map max
    if aux_map is not None:
        aux_fov_vals = aux_map[fov_mask]
        peak_aux = float(np.max(aux_fov_vals)) if len(aux_fov_vals) > 0 else 0.0
    else:
        peak_aux = 0.0

    return {
        "peak_cam_confidence": peak_cam,
        "fov_concentration_pct": concentration,
        "aux_peak_activation": peak_aux,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: FOV-CONSTRAINED CAUSAL PERTURBATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_perturbation(
    model: nn.Module,
    img_tensor: torch.Tensor,
    cam: np.ndarray,
    k_pct: float = 10.0,
    fill_strategy: str = "zero",  # "zero" or "blur"
    mode: str = "top_k_vs_random",  # "top_k_vs_random" or "box_vs_outside"
    boxes: Optional[List[List[float]]] = None,
    fov_mask: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run FOV-constrained causal perturbation experiment on any radiograph.

    Operates cleanly with or without ground-truth boxes.
    """
    if device is None:
        device = next(model.parameters()).device

    prob_orig, logit_orig = run_inference(model, img_tensor, device)

    H, W = img_tensor.shape[1], img_tensor.shape[2]
    fov_np = fov_mask if fov_mask is not None else np.ones((H, W), dtype=bool)

    if fill_strategy == "zero":
        fill_tensor = torch.zeros_like(img_tensor)
    elif fill_strategy == "blur":
        fill_tensor = TF.gaussian_blur(img_tensor, kernel_size=[31, 31], sigma=[10.0, 10.0])
    else:
        fov_t = torch.from_numpy(fov_np)
        mean_val = img_tensor[:, fov_t].mean(dim=1, keepdim=True).unsqueeze(-1)
        fill_tensor = mean_val.expand_as(img_tensor)

    def mask_and_infer(pixel_mask_2d: np.ndarray) -> Tuple[float, float, np.ndarray]:
        x_m = img_tensor.clone()
        m_t = torch.from_numpy(pixel_mask_2d)
        x_m[:, m_t] = fill_tensor[:, m_t]
        prob_m, logit_m = run_inference(model, x_m, device)

        disp = x_m.clone().cpu() * IMAGENET_STD + IMAGENET_MEAN
        disp_uint8 = (disp.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return prob_m, logit_m, disp_uint8

    # Force top_k_vs_random if no boxes are present
    if mode == "box_vs_outside" and not boxes:
        mode = "top_k_vs_random"

    if mode == "box_vs_outside" and boxes:
        box_mask = np.zeros((H, W), dtype=bool)
        for box in boxes:
            x1, y1, x2, y2 = (int(round(c)) for c in box[:4])
            x1, x2 = max(0, x1), min(W - 1, x2)
            y1, y2 = max(0, y1), min(H - 1, y2)
            box_mask[y1:y2+1, x1:x2+1] = True
        attr_mask = box_mask & fov_np

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
        if len(fov_indices) >= k_count:
            ctrl_flat = rng.choice(fov_indices, size=k_count, replace=False)
        else:
            ctrl_flat = fov_indices
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


# ── File Utility Helpers ───────────────────────────────────────────────────────
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
