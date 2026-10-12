"""Region-Supervised ResNet-18 model for BoneTrace.

Implements a dual-head architecture sharing an identical ResNet-18 backbone:
1. Primary Classification Head: Global average pooling + linear projection for
   binary fracture classification logits.
2. Auxiliary Region-Supervision Head: Convolutional localization head operating on
   layer4 spatial feature maps to predict spatial fracture probability masks.

Enables controlled experimental investigation of whether explicit fracture-region
supervision improves the faithfulness and stability of post-hoc explanations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class RegionSupervisedModel(nn.Module):
    """Dual-head ResNet-18 with primary classification and auxiliary region supervision.

    Retains the exact same convolutional backbone stages as BaselineClassifier to ensure
    direct, controlled comparability between classification-only and region-supervised models.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        num_classes: int = 1,
        dropout_rate: float = 0.0,
        aux_channels: int = 128,
        target_size: Tuple[int, int] = (512, 512),
    ) -> None:
        """Initialize RegionSupervisedModel.

        Args:
            pretrained: Whether to load ImageNet pretrained weights (optional, offline default is False).
            in_channels: Number of input channels (default: 3).
            num_classes: Number of classification output logits (default: 1).
            dropout_rate: Optional dropout probability in classification head.
            aux_channels: Intermediate channel dimension for auxiliary region head.
            target_size: Output spatial resolution (width, height) for mask logits.
        """
        super().__init__()
        self.model_type = "region_supervised"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_size = target_size

        # Load ResNet-18 backbone
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

        # Primary classification head
        in_features = self.backbone.fc.in_features  # 512
        if dropout_rate > 0.0:
            self.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.fc = nn.Linear(in_features, num_classes)

        # Remove default fc from backbone since we compute features explicitly
        self.backbone.fc = nn.Identity()

        # Auxiliary Region-Supervision Head (convolutional localization / segmentation)
        # Operates on the 512-channel feature map output from layer4
        self.region_head = nn.Sequential(
            nn.Conv2d(in_features, aux_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aux_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aux_channels, 1, kernel_size=1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map from the final residual block (layer4).

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Feature map tensor of shape (B, 512, H//32, W//32).
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feat_map = self.backbone.layer4(x)
        return feat_map

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass computing both classification logits and auxiliary mask logits.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary with:
            - 'logits': Primary binary classification logits (B, 1).
            - 'mask_logits': Auxiliary spatial region logits (B, 1, H, W).
        """
        # 1. Extract shared spatial feature map from layer4
        feat_map = self.extract_features(x)  # (B, 512, 16, 16) for 512x512 input

        # 2. Classification branch: Global Average Pooling + FC
        pooled = self.backbone.avgpool(feat_map)
        pooled = torch.flatten(pooled, 1)
        cls_logits = self.fc(pooled)

        # 3. Auxiliary Region Supervision branch: Conv Head + Bilinear Upsampling
        raw_mask_logits = self.region_head(feat_map)  # (B, 1, 16, 16)
        mask_logits = F.interpolate(
            raw_mask_logits,
            size=(self.target_size[1], self.target_size[0]),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "logits": cls_logits,
            "mask_logits": mask_logits,
        }

    def get_cam_target_layer(self) -> nn.Module:
        """Return target layer for Grad-CAM post-hoc explanation extraction.

        Matches BaselineClassifier target layer exactly (layer4[-1]).
        """
        return self.backbone.layer4[-1]
