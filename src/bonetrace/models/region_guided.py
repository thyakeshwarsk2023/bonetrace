"""Region-Supervised model for BoneTrace.

Implements a dual-head architecture supporting ResNet-18 and DenseNet-121 backbones:
1. Primary Classification Head: Global average pooling + linear projection for
   binary fracture classification logits.
2. Auxiliary Region-Supervision Head: Convolutional localization head operating on
   final spatial feature maps (stride 32, 16x16) to predict spatial fracture probability masks.

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
    """Dual-head model with primary classification and auxiliary region supervision.

    Retains the exact same convolutional backbone stages as BaselineClassifier to ensure
    direct, controlled comparability between classification-only and region-supervised models.
    Supports ResNet-18 and DenseNet-121 backbones.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        num_classes: int = 1,
        dropout_rate: float = 0.0,
        aux_channels: int = 128,
        target_size: Tuple[int, int] = (512, 512),
        backbone: str = "resnet18",
        memory_efficient: bool = True,
    ) -> None:
        """Initialize RegionSupervisedModel.

        Args:
            pretrained: Whether to load ImageNet pretrained weights (default: False).
            in_channels: Number of input channels (default: 3).
            num_classes: Number of classification output logits (default: 1).
            dropout_rate: Optional dropout probability in classification head.
            aux_channels: Intermediate channel dimension for auxiliary region head.
            target_size: Output spatial resolution (width, height) for mask logits.
            backbone: Backbone architecture ('resnet18' or 'densenet121').
            memory_efficient: Whether to use gradient checkpointing in DenseNet-121.
        """
        super().__init__()
        self.model_type = "region_supervised"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_size = target_size
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

            # Ensure all ReLUs use inplace=False for Grad-CAM backward hook safety
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
            self.backbone.fc = nn.Identity()

        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone}'. Must be 'resnet18' or 'densenet121'."
            )

        # Primary classification head
        if dropout_rate > 0.0:
            self.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.fc = nn.Linear(in_features, num_classes)

        # Auxiliary Region-Supervision Head (convolutional localization / segmentation)
        # Operates on spatial feature maps (512 channels for ResNet-18, 1024 channels for DenseNet-121)
        # Uses inplace=False for all activations to ensure autograd hook stability
        self.region_head = nn.Sequential(
            nn.Conv2d(in_features, aux_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aux_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(aux_channels, 1, kernel_size=1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map before pooling.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Feature map tensor:
            - Shape (B, 1024, H//32, W//32) for DenseNet-121.
            - Shape (B, 512, H//32, W//32) for ResNet-18.
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
        # 1. Extract shared spatial feature map (B, in_features, 16, 16) for 512x512 input
        feat_map = self.extract_features(x)

        # 2. Classification branch: Global Average Pooling + FC
        if hasattr(self.backbone, "avgpool") and self.backbone_name == "resnet18":
            pooled = self.backbone.avgpool(feat_map)
        else:
            pooled = F.adaptive_avg_pool2d(feat_map, (1, 1))
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
            "features": feat_map,
        }

    def get_cam_target_layer(self) -> nn.Module:
        """Return target layer for Grad-CAM post-hoc explanation extraction.

        For ResNet-18: layer4[-1].
        For DenseNet-121: features.denseblock4.
        """
        if self.backbone_name == "densenet121":
            return self.backbone.features.denseblock4
        return self.backbone.layer4[-1]
