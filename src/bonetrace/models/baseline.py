"""Baseline ResNet-18 classification model for BoneTrace.

Implements standard binary fracture classification using a ResNet-18 backbone,
global average pooling, and a single linear classification projection.
Serves as the controlled experimental baseline to compare post-hoc Grad-CAM explanations
against the region-supervised model variant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as tv_models


class BaselineClassifier(nn.Module):
    """ResNet-18 binary fracture classifier.

    Outputs unnormalized binary classification logits (shape [B, 1]).
    Target convolutional layer for Grad-CAM post-hoc explanations is backbone.layer4[-1].
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        num_classes: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize BaselineClassifier.

        Args:
            pretrained: Whether to load ImageNet pretrained weights (optional, offline default is False).
            in_channels: Number of input image channels (default: 3 for RGB).
            num_classes: Number of output classification targets (default: 1 for binary fracture logit).
            dropout_rate: Optional dropout probability before classification head.
        """
        super().__init__()
        self.model_type = "baseline"
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Instantiate ResNet-18 backbone
        weights = None
        if pretrained:
            try:
                weights = tv_models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None

        self.backbone = tv_models.resnet18(weights=weights)

        # Handle in_channels if different from 3
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        # Replace standard 1000-class fc layer with 1-class binary linear head
        in_features = self.backbone.fc.in_features  # 512 for ResNet-18
        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary containing 'logits' tensor of shape (B, 1).
        """
        logits = self.backbone(x)
        return {"logits": logits}

    def get_cam_target_layer(self) -> nn.Module:
        """Return the target convolutional layer for Grad-CAM extraction.

        For ResNet-18, the last convolutional unit of the final residual stage is
        layer4[-1].conv2 (or the entire residual block layer4[-1]).
        """
        return self.backbone.layer4[-1]
