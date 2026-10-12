"""Grad-CAM Post-Hoc Explanation Extraction for BoneTrace.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) with:
- Forward/backward feature extraction hooks targeting the final residual block (layer4[-1]).
- Clean activation and gradient caching with automatic hook removal.
- Bilinear upsampling to the standardized 512x512 canvas with min-max normalization.
- Controlled comparison across baseline (classification-only) and region-supervised models.
- Visual diagnostic overlay generation and machine-readable NumPy/JSON export.

Scientific Note:
Grad-CAM heatmaps are first-order gradient attribution approximations, not causal explanations.
Visual agreement with radiologist annotations constitutes spatial localization agreement,
not mathematical proof of algorithmic faithfulness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from bonetrace.config import AppConfig
from bonetrace.logging_utils import get_logger

logger = get_logger("explainability.gradcam")


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) extractor.

    Attaches forward and backward hooks to a target convolutional layer,
    computes channel-wise importance weights via global average pooled gradients,
    and constructs a normalized 2D attribution heatmap.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        target_size: Tuple[int, int] = (512, 512),
    ) -> None:
        """Initialize GradCAM.

        Args:
            model: PyTorch model (BaselineClassifier or RegionSupervisedModel).
            target_layer: Target convolutional layer. If None, queries model.get_cam_target_layer().
            target_size: Output spatial resolution (width, height) for upsampled heatmaps.
        """
        self.model = model
        self.target_size = target_size

        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_cam_target_layer"):
            self.target_layer = model.get_cam_target_layer()
        else:
            # Fallback: search for last Conv2d or residual block
            convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
            if not convs:
                raise ValueError("Could not find convolutional layer for Grad-CAM.")
            self.target_layer = convs[-1]

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.extend([h_fwd, h_bwd])

    def remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int = 0,
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM heatmap for an input image.

        Args:
            image_tensor: Tensor of shape (1, 3, H, W) or (3, H, W).
            class_idx: Target logit index (0 for binary fracture classification).

        Returns:
            np.ndarray of shape (H, W) with values normalized to [0.0, 1.0].
        """
        self.model.eval()

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        x = image_tensor.to(device).clone().requires_grad_(True)

        self.model.zero_grad()
        outputs = self.model(x)

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs

        # Select target logit for backpropagation
        target_score = logits[:, class_idx] if logits.shape[1] > class_idx else logits[:, 0]
        target_score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Channel-wise global average pooling of gradients: (1, C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # Rectified Linear Unit (ReLU) to isolate positive contributions to fracture class
        cam = F.relu(cam)

        # Bilinear interpolation to target canvas resolution (512, 512)
        cam = F.interpolate(
            cam,
            size=(self.target_size[1], self.target_size[0]),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization into [0.0, 1.0]
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        denom = cam_max - cam_min
        if denom > 1e-8:
            cam_norm = (cam_np - cam_min) / denom
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm.astype(np.float32)


def render_heatmap_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Render a visual color overlay combining an RGB radiograph with a Grad-CAM heatmap.

    Args:
        image_rgb: HxWx3 uint8 numpy array [0, 255] or float32 [0.0, 1.0].
        heatmap: HxW float32 numpy array [0.0, 1.0].
        alpha: Blending weight for the color heatmap.
        colormap_name: Matplotlib colormap identifier.

    Returns:
        HxWx3 uint8 RGB numpy array.
    """
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            img_uint8 = (image_rgb * 255).clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = image_rgb.clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = image_rgb

    # Apply colormap to heatmap
    import matplotlib
    try:
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = cm.get_cmap(colormap_name)
    heat_rgba = cmap(heatmap)  # HxWx4 float in [0, 1]
    heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)

    # Alpha blend
    blended = (1.0 - alpha) * img_uint8.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)


def compute_explanations(
    config: AppConfig,
    model: Optional[nn.Module] = None,
    dataset: Optional[Any] = None,
    num_samples: int = 10,
    output_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute and save post-hoc Grad-CAM explanations for verification samples.

    Args:
        config: Application configuration.
        model: Model instance to explain. If None, builds baseline model from config.
        dataset: Dataset instance. If None, loads test split.
        num_samples: Number of examples to explain and visualize.
        output_subdir: Optional subfolder inside config.paths.explanations_dir.

    Returns:
        Dictionary summarizing explanation generation results.
    """
    logger.info("Initializing explanation extraction pipeline...")
    exp_dir = Path(config.paths.explanations_dir)
    if output_subdir:
        exp_dir = exp_dir / output_subdir
    exp_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        from bonetrace.models.factory import build_model
        model = build_model(config=config)

    device = torch.device("cuda" if torch.cuda.is_available() and getattr(config.project, "device", "cpu") == "cuda" else "cpu")
    model.to(device)
    model.eval()

    if dataset is None:
        from bonetrace.data.dataset import FracAtlasDataset
        test_csv = config.paths.processed_data_dir / "test.csv"
        if not test_csv.is_file():
            test_csv = config.paths.processed_data_dir / "canonical_manifest.csv"
        dataset = FracAtlasDataset(manifest=test_csv, split="test", exclude_conflicts=True)

    gradcam = GradCAM(model=model, target_size=(512, 512))
    saved_records = []

    n = min(len(dataset), num_samples)
    logger.info("Generating Grad-CAM explanations for %d samples...", n)

    try:
        for idx in range(n):
            sample = dataset[idx]
            cid = sample["canonical_id"]
            img_tensor = sample["image"]
            label_val = int(sample["label"].item())

            cam = gradcam.generate(img_tensor, class_idx=0)

            # Convert normalized tensor to displayable RGB image
            img_disp = img_tensor.clone()
            if hasattr(dataset, "normalize") and dataset.normalize:
                # Invert ImageNet normalization for visualization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_disp = img_disp * std + mean
            img_disp = img_disp.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()

            overlay = render_heatmap_overlay(img_disp, cam, alpha=0.45)

            # Save overlay image and raw numpy heatmap
            overlay_file = exp_dir / f"{cid}_gradcam_overlay.png"
            npy_file = exp_dir / f"{cid}_gradcam.npy"

            Image.fromarray(overlay).save(overlay_file)
            np.save(npy_file, cam)

            saved_records.append({
                "canonical_id": cid,
                "label": label_val,
                "has_annotation": bool(sample["has_annotation"].item()),
                "overlay_path": str(overlay_file),
                "heatmap_npy_path": str(npy_file),
                "cam_min": float(cam.min()),
                "cam_max": float(cam.max()),
                "cam_mean": float(cam.mean()),
            })
    finally:
        gradcam.remove_hooks()

    manifest_path = exp_dir / "explanations_manifest.json"
    summary = {
        "model_type": getattr(model, "model_type", "unknown"),
        "total_explanations_generated": len(saved_records),
        "output_directory": str(exp_dir),
        "records": saved_records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved %d explanations to %s", len(saved_records), exp_dir)
    return summary
