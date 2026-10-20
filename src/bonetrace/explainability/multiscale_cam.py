"""
src/bonetrace/explainability/multiscale_cam.py

Track B Multi-Scale and FOV-Aware Attribution Engine:
- FOV constraint: Omega_FOV defined by normalized channel-0 >= -2.0.
- Target: fracture classification logit.
- Methods:
  1. Grad-CAM: layer4[-1] (16x16), layer3[-1] (32x32), layer2[-1] (64x64).
  2. HiResCAM: layer3[-1] (attribution method) and layer4[-1] (identity test).
     Formula: ReLU(sum_k A_k * dy/dA_k) with no spatial gradient averaging.
  3. LayerCAM (layers 2+3+4): positive-gradient-weighted activations per layer,
     ReLU, upsampled to 512x512, min-max normalized within Omega_FOV, averaged across layers.
  4. Direct maps:
     - Model B & B_placebo: sigmoid(M_aux) masked to Omega_FOV.
     - Model C: CAM logit map s_up masked to Omega_FOV.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_fov_mask(image_tensor: torch.Tensor, threshold: float = -2.0) -> torch.Tensor:
    """Compute anatomical Field of View (Omega_FOV) boolean mask.

    Letterbox black border pixels have raw value 0.0. Under ImageNet normalization
    with mean 0.485 and std 0.229, channel-0 border values are (0 - 0.485)/0.229 = -2.1179.
    Pixels with channel-0 >= -2.0 belong to the anatomical radiograph field of view.

    Args:
        image_tensor: Normalized image tensor of shape (3, H, W) or (B, 3, H, W).
        threshold: Normalized intensity threshold (default: -2.0).

    Returns:
        Boolean mask tensor of shape (H, W) or (B, 1, H, W).
    """
    if image_tensor.dim() == 3:
        return image_tensor[0] >= threshold
    elif image_tensor.dim() == 4:
        return (image_tensor[:, 0:1] >= threshold)
    else:
        raise ValueError(f"Unexpected image tensor shape: {image_tensor.shape}")


def min_max_normalize_fov(cam_map: torch.Tensor, fov_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize attribution map strictly within Omega_FOV.

    Pixels outside Omega_FOV are set to 0.0.
    """
    if cam_map.dim() == 2:
        cam_map = cam_map.unsqueeze(0).unsqueeze(0)
    elif cam_map.dim() == 3:
        cam_map = cam_map.unsqueeze(1)
        
    if fov_mask.dim() == 2:
        fov_mask = fov_mask.unsqueeze(0).unsqueeze(0)
    elif fov_mask.dim() == 3:
        fov_mask = fov_mask.unsqueeze(1)

    normalized = torch.zeros_like(cam_map)
    B = cam_map.shape[0]

    for b in range(B):
        mask_b = fov_mask[b, 0]
        vals = cam_map[b, 0][mask_b]
        if len(vals) > 0 and (vals.max() - vals.min()) > eps:
            v_min = vals.min()
            v_max = vals.max()
            norm_b = (cam_map[b, 0] - v_min) / (v_max - v_min)
            normalized[b, 0] = torch.where(mask_b, norm_b, torch.zeros_like(norm_b))
        else:
            normalized[b, 0] = torch.zeros_like(cam_map[b, 0])

    return normalized


class MultiScaleAttributionEngine:
    """Multi-Scale and FOV-Aware attribution extractor for Track B models."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

        # Identify layers in ResNet-18
        if hasattr(model, "backbone"):
            self.layer4 = model.backbone.layer4[-1]
            self.layer3 = model.backbone.layer3[-1]
            self.layer2 = model.backbone.layer2[-1]
        else:
            self.layer4 = model.layer4[-1]
            self.layer3 = model.layer3[-1]
            self.layer2 = model.layer2[-1]

    def _extract_activations_and_gradients(
        self,
        target_layers: List[nn.Module],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Hook target layers, compute classification logit, and backpropagate."""
        activations: Dict[int, torch.Tensor] = {}
        gradients: Dict[int, torch.Tensor] = {}
        handles = []

        for idx, layer in enumerate(target_layers):
            def make_fwd_hook(layer_idx):
                def hook(module, inp, out):
                    activations[layer_idx] = out
                    out.retain_grad()
                return hook

            def make_bwd_hook(layer_idx):
                def hook(module, grad_in, grad_out):
                    gradients[layer_idx] = grad_out[0]
                return hook

            handles.append(layer.register_forward_hook(make_fwd_hook(idx)))

        self.model.zero_grad()
        out = self.model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        
        # Target is the fracture classification logit
        loss = logits.sum()
        loss.backward(retain_graph=True)

        for h in handles:
            h.remove()

        acts = [activations[i] for i in range(len(target_layers))]
        grads = [acts[i].grad if acts[i].grad is not None else gradients.get(i) for i in range(len(target_layers))]

        return logits.detach(), acts, grads

    def compute_all_attributions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all pre-registered attribution maps for an input batch.

        Args:
            x: Input image tensor (B, 3, 512, 512) on self.device.

        Returns:
            Dictionary mapping attribution names to normalized maps (B, 1, 512, 512) in [0, 1].
        """
        B, C, H, W = x.shape
        fov_mask = compute_fov_mask(x)

        layers = [self.layer4, self.layer3, self.layer2]
        logits, acts, grads = self._extract_activations_and_gradients(layers, x)

        A4, A3, A2 = acts[0], acts[1], acts[2]
        G4, G3, G2 = grads[0], grads[1], grads[2]

        results: Dict[str, torch.Tensor] = {}

        # 1. Grad-CAM layer4[-1]
        alpha4 = G4.mean(dim=(2, 3), keepdim=True)
        raw_gc4 = F.relu((alpha4 * A4).sum(dim=1, keepdim=True))
        up_gc4 = F.interpolate(raw_gc4, size=(H, W), mode="bilinear", align_corners=False)
        results["gradcam_layer4"] = min_max_normalize_fov(up_gc4, fov_mask)

        # 2. Grad-CAM layer3[-1]
        alpha3 = G3.mean(dim=(2, 3), keepdim=True)
        raw_gc3 = F.relu((alpha3 * A3).sum(dim=1, keepdim=True))
        up_gc3 = F.interpolate(raw_gc3, size=(H, W), mode="bilinear", align_corners=False)
        results["gradcam_layer3"] = min_max_normalize_fov(up_gc3, fov_mask)

        # 3. Grad-CAM layer2[-1]
        alpha2 = G2.mean(dim=(2, 3), keepdim=True)
        raw_gc2 = F.relu((alpha2 * A2).sum(dim=1, keepdim=True))
        up_gc2 = F.interpolate(raw_gc2, size=(H, W), mode="bilinear", align_corners=False)
        results["gradcam_layer2"] = min_max_normalize_fov(up_gc2, fov_mask)

        # 4. HiResCAM layer4[-1] (Identity check)
        raw_hr4 = F.relu((A4 * G4).sum(dim=1, keepdim=True))
        up_hr4 = F.interpolate(raw_hr4, size=(H, W), mode="bilinear", align_corners=False)
        results["hirescam_layer4"] = min_max_normalize_fov(up_hr4, fov_mask)

        # 5. HiResCAM layer3[-1] (Exploratory attribution method)
        raw_hr3 = F.relu((A3 * G3).sum(dim=1, keepdim=True))
        up_hr3 = F.interpolate(raw_hr3, size=(H, W), mode="bilinear", align_corners=False)
        results["hirescam_layer3"] = min_max_normalize_fov(up_hr3, fov_mask)

        # 6. LayerCAM (layers 2+3+4)
        # Element-wise positive gradient weighted activations per layer
        w2 = F.relu(G2)
        w3 = F.relu(G3)
        w4 = F.relu(G4)

        raw_lc2 = F.relu((w2 * A2).sum(dim=1, keepdim=True))
        raw_lc3 = F.relu((w3 * A3).sum(dim=1, keepdim=True))
        raw_lc4 = F.relu((w4 * A4).sum(dim=1, keepdim=True))

        up_lc2 = min_max_normalize_fov(F.interpolate(raw_lc2, size=(H, W), mode="bilinear", align_corners=False), fov_mask)
        up_lc3 = min_max_normalize_fov(F.interpolate(raw_lc3, size=(H, W), mode="bilinear", align_corners=False), fov_mask)
        up_lc4 = min_max_normalize_fov(F.interpolate(raw_lc4, size=(H, W), mode="bilinear", align_corners=False), fov_mask)

        layercam = (up_lc2 + up_lc3 + up_lc4) / 3.0
        results["layercam"] = min_max_normalize_fov(layercam, fov_mask)

        # 7. Direct maps for Model B / B_placebo and Model C
        with torch.no_grad():
            fwd_out = self.model(x)
            if isinstance(fwd_out, dict):
                if "mask_logits" in fwd_out:
                    # Model B / B_placebo
                    raw_direct = torch.sigmoid(fwd_out["mask_logits"])
                    up_direct = F.interpolate(raw_direct, size=(H, W), mode="bilinear", align_corners=False)
                    results["direct_map"] = min_max_normalize_fov(up_direct, fov_mask)
                elif "cam_up" in fwd_out:
                    # Model C
                    raw_direct = fwd_out["cam_up"]
                    results["direct_map"] = min_max_normalize_fov(raw_direct, fov_mask)

        results["fov_mask"] = fov_mask.float()
        return {k: v.detach() for k, v in results.items()}
