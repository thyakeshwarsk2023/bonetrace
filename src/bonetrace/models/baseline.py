"""Baseline classification model for BoneTrace.

Implements standard binary fracture classification supporting both ResNet-18
and DenseNet-121 backbones, global average pooling, and a linear classification projection.
Serves as the controlled experimental baseline to compare post-hoc Grad-CAM explanations
against the region-supervised model variant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class BaselineClassifier(nn.Module):
    """Binary fracture classifier supporting ResNet-18 and DenseNet-121 backbones.

    Outputs unnormalized binary classification logits (shape [B, 1]).
    Target convolutional layer for Grad-CAM post-hoc explanations is:
    - backbone.layer4[-1] for ResNet-18
    - backbone.features.denseblock4 for DenseNet-121
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        num_classes: int = 1,
        dropout_rate: float = 0.0,
        backbone: str = "resnet18",
        memory_efficient: bool = True,
    ) -> None:
        """Initialize BaselineClassifier.

        Args:
            pretrained: Whether to load ImageNet pretrained weights (default: False).
            in_channels: Number of input image channels (default: 3 for RGB).
            num_classes: Number of output classification targets (default: 1).
            dropout_rate: Optional dropout probability before classification head.
            backbone: Backbone architecture ('resnet18' or 'densenet121').
            memory_efficient: Whether to use gradient checkpointing in DenseNet-121.
        """
        super().__init__()
        self.model_type = "baseline"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone.lower().strip()

        if self.backbone_name in ["densenet121", "densenet", "dense121"]:
            self.backbone_name = "densenet121"
            weights = None
            if pretrained:
                try:
                    weights = tv_models.DenseNet121_Weights.DEFAULT
                except Exception:
                    weights = None

            self.backbone = tv_models.densenet121(weights=weights, memory_efficient=memory_efficient)

            # Ensure all ReLUs use inplace=False so autograd backward hooks function without view errors
            for m in self.backbone.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False

            if in_channels != 3:
                old_conv = self.backbone.features.conv0
                self.backbone.features.conv0 = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )

            in_features = self.backbone.classifier.in_features  # 1024
            self.backbone.classifier = nn.Identity()

            if dropout_rate > 0.0:
                self.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(in_features, num_classes),
                )
            else:
                self.fc = nn.Linear(in_features, num_classes)

        elif self.backbone_name in ["resnet18", "resnet", "resnet-18"]:
            self.backbone_name = "resnet18"
            weights = None
            if pretrained:
                try:
                    weights = tv_models.ResNet18_Weights.DEFAULT
                except Exception:
                    weights = None

            self.backbone = tv_models.resnet18(weights=weights)

            # Ensure all ReLUs use inplace=False
            for m in self.backbone.modules():
                if isinstance(m, nn.ReLU):
                    m.inplace = False

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

            in_features = self.backbone.fc.in_features  # 512
            if dropout_rate > 0.0:
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(in_features, num_classes),
                )
            else:
                self.backbone.fc = nn.Linear(in_features, num_classes)
            self.fc = self.backbone.fc

        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone}'. Must be 'resnet18' or 'densenet121'."
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map before pooling.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Spatial feature tensor:
            - Shape (B, 1024, 16, 16) for DenseNet-121 on 512x512 input.
            - Shape (B, 512, 16, 16) for ResNet-18 on 512x512 input.
        """
        if self.backbone_name == "densenet121":
            features = self.backbone.features(x)
            return F.relu(features, inplace=False)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            return self.backbone.layer4(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary containing:
            - 'logits': Classification logits tensor of shape (B, 1).
            - 'features': Final spatial feature map tensor of shape (B, channels, 16, 16).
        """
        if self.backbone_name == "densenet121":
            feat = self.extract_features(x)
            pooled = F.adaptive_avg_pool2d(feat, (1, 1))
            pooled = torch.flatten(pooled, 1)
            logits = self.fc(pooled)
            return {"logits": logits, "features": feat}
        else:
            feat = self.extract_features(x)
            pooled = self.backbone.avgpool(feat)
            pooled = torch.flatten(pooled, 1)
            logits = self.backbone.fc(pooled)
            return {"logits": logits, "features": feat}

    def get_cam_target_layer(self) -> nn.Module:
        """Return the target convolutional layer for Grad-CAM extraction.

        For ResNet-18: layer4[-1].
        For DenseNet-121: features.denseblock4.
        """
        if self.backbone_name == "densenet121":
            return self.backbone.features.denseblock4
        return self.backbone.layer4[-1]
