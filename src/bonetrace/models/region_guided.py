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


class FeaturePyramidRegionHead(nn.Module):
    """Multi-Scale Feature Pyramid Network (FPN) Decoder for fracture localization.

    Fuses multi-scale feature hierarchies:
      - C2: Stride 8,  64x64, 128 channels
      - C3: Stride 16, 32x32, 256 channels
      - C4: Stride 32, 16x16, 512 (ResNet-18) or 1024 (DenseNet-121) channels

    Constructs top-down feature pyramid:
      - P4 = Lateral_4(C4)                          (16x16, fpn_channels)
      - P3 = Lateral_3(C3) + Upsample2x(P4)         (32x32, fpn_channels)
      - P2 = Lateral_2(C2) + Upsample2x(P3)         (64x64, fpn_channels)

    Smooths pyramid maps with 3x3 convs and unifies multi-scale context
    at 64x64 resolution (stride 8) before final 1x1 projection and bilinear
    upsampling to full target resolution (512x512).
    """

    def __init__(
        self,
        c2_channels: int = 128,
        c3_channels: int = 256,
        c4_channels: int = 512,
        fpn_channels: int = 128,
    ) -> None:
        super().__init__()
        self.fpn_channels = fpn_channels

        # Lateral 1x1 convolutions to project all stages to fpn_channels
        self.lateral_c4 = nn.Conv2d(c4_channels, fpn_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(c3_channels, fpn_channels, kernel_size=1)
        self.lateral_c2 = nn.Conv2d(c2_channels, fpn_channels, kernel_size=1)

        # Anti-aliasing smooth 3x3 convolutions
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=False),
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=False),
        )
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=False),
        )

        # Multi-scale fusion and final mask projection
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(fpn_channels, 1, kernel_size=1),
        )

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        """Forward pass through FPN pyramid.

        Returns:
            High-resolution spatial mask logits at stride 8 (64x64 for 512x512 input).
        """
        # 1. Top-down pyramid construction
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        # 2. Smooth feature maps
        p4_smooth = self.smooth_p4(p4)
        p3_smooth = self.smooth_p3(p3)
        p2_smooth = self.smooth_p2(p2)

        # 3. Fuse multi-scale features at stride 8 (size of c2: 64x64)
        p3_up = F.interpolate(p3_smooth, size=p2_smooth.shape[-2:], mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4_smooth, size=p2_smooth.shape[-2:], mode="bilinear", align_corners=False)
        fused = p2_smooth + p3_up + p4_up

        # 4. Predict fine-grained mask logits (stride 8)
        mask_stride8 = self.fusion_conv(fused)
        return mask_stride8


class RegionSupervisedModel(nn.Module):
    """Dual-head model with primary classification and auxiliary region supervision.

    Retains the exact same convolutional backbone stages as BaselineClassifier to ensure
    direct, controlled comparability between classification-only and region-supervised models.
    Supports ResNet-18 and DenseNet-121 backbones with optional Multi-Scale Feature Pyramid
    Decoder (FPN) for resolving the 16x16 spatial resolution bottleneck.
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
        use_fpn: bool = True,
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
            use_fpn: Whether to use Multi-Scale Feature Pyramid Network (FPN) decoder (default: True).
        """
        super().__init__()
        self.model_type = "region_supervised"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_size = target_size
        self.backbone_name = backbone.lower().strip()
        self.use_fpn = use_fpn

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

        # Auxiliary Region-Supervision Head (Single-scale legacy fallback)
        # Operates on spatial feature maps (512 channels for ResNet-18, 1024 channels for DenseNet-121)
        # Uses inplace=False for all activations to ensure autograd hook stability
        self.region_head = nn.Sequential(
            nn.Conv2d(in_features, aux_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aux_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(aux_channels, 1, kernel_size=1),
        )

        # Multi-Scale Feature Pyramid Decoder (FPN) Head
        # Resolves the 16x16 spatial bottleneck by fusing C2 (stride 8), C3 (stride 16), and C4 (stride 32)
        self.fpn_head = FeaturePyramidRegionHead(
            c2_channels=128,
            c3_channels=256,
            c4_channels=in_features,
            fpn_channels=aux_channels,
        )

    def extract_multiscale_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale spatial feature maps from stage 2, stage 3, and stage 4.

        Returns:
            Tuple of (c2, c3, c4):
            - c2: Stride 8,  shape (B, 128, H//8, W//8)
            - c3: Stride 16, shape (B, 256, H//16, W//16)
            - c4: Stride 32, shape (B, in_features, H//32, W//32)
        """
        if self.backbone_name == "densenet121":
            feats = self.backbone.features
            x = feats.conv0(x)
            x = feats.norm0(x)
            x = feats.relu0(x)
            x = feats.pool0(x)

            x = feats.denseblock1(x)
            c2 = feats.transition1(x)

            x = feats.denseblock2(c2)
            c3 = feats.transition2(x)

            x = feats.denseblock3(c3)
            x = feats.transition3(x)
            x = feats.denseblock4(x)
            c4 = F.relu(feats.norm5(x), inplace=False)
            return c2, c3, c4
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            c2 = self.backbone.layer2(x)
            c3 = self.backbone.layer3(c2)
            c4 = self.backbone.layer4(c3)
            return c2, c3, c4

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
            - 'features': Stride-32 feature map (B, in_features, 16, 16).
        """
        if self.use_fpn:
            c2, c3, c4 = self.extract_multiscale_features(x)
            feat_map = c4
            raw_mask_logits = self.fpn_head(c2, c3, c4)  # High-resolution stride 8 (64x64)
        else:
            # 1. Extract shared spatial feature map (B, in_features, 16, 16) for 512x512 input
            feat_map = self.extract_features(x)
            raw_mask_logits = self.region_head(feat_map)  # Legacy stride 32 (16x16)

        # 2. Classification branch: Global Average Pooling + FC
        if hasattr(self.backbone, "avgpool") and self.backbone_name == "resnet18":
            pooled = self.backbone.avgpool(feat_map)
        else:
            pooled = F.adaptive_avg_pool2d(feat_map, (1, 1))
        pooled = torch.flatten(pooled, 1)
        cls_logits = self.fc(pooled)

        # 3. Auxiliary Region Supervision branch: Upsample to target resolution
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

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Harmonize state dict loading: auto-detect FPN vs single-scale checkpoints."""
        has_fpn_in_ckpt = any(k.startswith(f"{prefix}fpn_head.") for k in state_dict)
        has_region_in_ckpt = any(k.startswith(f"{prefix}region_head.") for k in state_dict)
        if not has_fpn_in_ckpt and has_region_in_ckpt:
            self.use_fpn = False
        elif has_fpn_in_ckpt:
            self.use_fpn = True

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def get_cam_target_layer(self) -> nn.Module:
        """Return target layer for Grad-CAM post-hoc explanation extraction.

        For ResNet-18: layer4[-1].
        For DenseNet-121: features.denseblock4.
        """
        if self.backbone_name == "densenet121":
            return self.backbone.features.denseblock4
        return self.backbone.layer4[-1]
