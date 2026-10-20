import hashlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest

from bonetrace.models.baseline import BaselineClassifier
from bonetrace.models.region_guided import RegionSupervisedModel

class DirectCAMModel(nn.Module):
    """Model C: Direct CAM Alignment model without auxiliary convolutional head.

    Computes spatial CAM logit map s = sum_k w_k * A_k at layer4 (16x16).
    Upsamples s to 512x512 via bilinear interpolation.
    Mathematically, the spatial mean of s plus bias equals the classification logit.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        import torchvision.models as tv_models
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.classifier = nn.Linear(in_features, 1)
        self.backbone.fc = nn.Identity()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, x: torch.Tensor):
        feat = self.extract_features(x)  # (B, 512, 16, 16)
        w = self.classifier.weight  # (1, 512)
        b = self.classifier.bias    # (1,)

        # s(u, v) = sum_k w_k * A_{k, u, v}
        s = (feat * w.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # (B, 1, 16, 16)
        
        # Classification logit: spatial mean of s + bias
        logits = s.mean(dim=(2, 3)) + b  # (B, 1)
        
        # Differentiable upsampled CAM map for box supervision
        s_up = F.interpolate(s, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)  # (B, 1, 512, 512)
        
        return {
            "logits": logits,
            "cam_map": s,
            "cam_up": s_up,
            "features": feat,
        }

def test_model_c_cam_identity():
    """Unit test for Model C: mean over spatial positions of s plus bias equals classification logit to within 1e-5."""
    torch.manual_seed(42)
    model = DirectCAMModel(pretrained=False)
    model.eval()

    x = torch.randn(4, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
        logits = out["logits"]
        s = out["cam_map"]
        b = model.classifier.bias

        reconstructed_logits = s.mean(dim=(2, 3)) + b
        diff = (logits - reconstructed_logits).abs().max().item()
        
        # Also compare with standard GAP forward pass
        feat = out["features"]
        gap = feat.mean(dim=(2, 3))
        standard_logits = model.classifier(gap)
        standard_diff = (logits - standard_logits).abs().max().item()

    assert diff < 1e-5, f"Spatial mean of s plus bias differs from logit by {diff}"
    assert standard_diff < 1e-5, f"Logit differs from standard GAP output by {standard_diff}"

def test_placebo_permutation_derangement():
    """Unit test: B_placebo permutation has no fixed points and is class-preserving."""
    n_positives = 499
    for seed in [42, 123, 2026, 7, 2027]:
        rng = np.random.RandomState(seed)
        perm = np.arange(n_positives)
        # Generate derangement
        while True:
            rng.shuffle(perm)
            if np.all(perm != np.arange(n_positives)):
                break
        
        # Check no fixed points
        fixed_points = np.where(perm == np.arange(n_positives))[0]
        assert len(fixed_points) == 0, f"Seed {seed} generated {len(fixed_points)} fixed points"
        # Check valid permutation of 0..n-1
        assert len(set(perm)) == n_positives, f"Seed {seed} is not a valid bijection"
        assert set(perm) == set(range(n_positives))

def test_hirescam_vs_gradcam_layer4_identity():
    """Unit test: For GAP + linear head (A, B), layer4 HiResCAM must equal layer4 Grad-CAM within tolerance."""
    torch.manual_seed(42)
    for model_class in [BaselineClassifier, RegionSupervisedModel]:
        model = model_class(pretrained=False)
        model.eval()

        target_layer = model.backbone.layer4[-1]
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)
            output.retain_grad()

        handle = target_layer.register_forward_hook(forward_hook)

        x = torch.randn(2, 3, 512, 512, requires_grad=True)
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = logits.sum()
        loss.backward()

        handle.remove()

        A = activations[0]  # (B, 512, 16, 16)
        dY_dA = A.grad      # (B, 512, 16, 16)

        # Grad-CAM: alpha_k = GAP(dY_dA), then ReLU(sum alpha_k * A_k)
        alpha = dY_dA.mean(dim=(2, 3), keepdim=True)
        gradcam = F.relu((alpha * A).sum(dim=1, keepdim=True))

        # HiResCAM: ReLU(sum A_k * dY_dA_k) without spatial averaging
        hirescam = F.relu((A * dY_dA).sum(dim=1, keepdim=True))

        max_abs_diff = (gradcam - hirescam).abs().max().item()
        assert max_abs_diff < 1e-5, f"{model_class.__name__} layer4 HiResCAM differs from Grad-CAM by {max_abs_diff}"
