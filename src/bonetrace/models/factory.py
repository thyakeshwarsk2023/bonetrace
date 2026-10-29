"""Model architecture factory for BoneTrace.

Instantiates baseline classification and region-supervised models
using consistent backbone configurations and deterministic seed initialization.
Supports ResNet-18 and DenseNet-121 backbones.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from bonetrace.logging_utils import get_logger
from bonetrace.models.baseline import BaselineClassifier
from bonetrace.models.region_guided import RegionSupervisedModel

if TYPE_CHECKING:
    from bonetrace.config import AppConfig

logger = get_logger("models.factory")


def build_model(
    config: Optional[AppConfig] = None,
    model_type: Optional[str] = None,
    pretrained: bool = False,
    dropout_rate: float = 0.0,
    target_size: tuple[int, int] = (512, 512),
    backbone: Optional[str] = None,
    memory_efficient: bool = True,
) -> nn.Module:
    """Build and return a model instance based on configuration and model type.

    Args:
        config: Optional AppConfig object.
        model_type: Explicit model type override ('baseline' or 'region_supervised').
        pretrained: Whether to load ImageNet pretrained weights (default: False).
        dropout_rate: Dropout rate for classification head.
        target_size: Image spatial resolution (width, height).
        backbone: Explicit backbone override ('resnet18' or 'densenet121').
        memory_efficient: Whether to enable memory efficient checkpointing for DenseNet-121.

    Returns:
        Instantiated nn.Module (BaselineClassifier or RegionSupervisedModel).

    Raises:
        ValueError: If model_type or backbone is unsupported.
    """
    selected_type = model_type
    selected_backbone = backbone
    seed = 42

    if config is not None:
        seed = getattr(config.project, "seed", 42)
        raw_training = config.raw_config.get("training", {}) if hasattr(config, "raw_config") else {}

        if selected_type is None:
            selected_type = (
                raw_training.get("model_type")
                or raw_training.get("model_name")
                or getattr(getattr(config, "training", None), "model_type", None)
                or getattr(getattr(config, "training", None), "model_name", None)
                or "baseline"
            )

        if selected_backbone is None:
            selected_backbone = (
                raw_training.get("backbone")
                or raw_training.get("model_backbone")
                or getattr(getattr(config, "training", None), "backbone", None)
            )

        if "pretrained" in raw_training:
            pretrained = bool(raw_training["pretrained"])
        if "dropout_rate" in raw_training:
            dropout_rate = float(raw_training["dropout_rate"])
        if "memory_efficient" in raw_training:
            memory_efficient = bool(raw_training["memory_efficient"])

    if selected_type is None:
        selected_type = "baseline"

    # Normalize backbone and model_type
    selected_type_norm = selected_type.lower().strip()

    # Detect backbone from model_type if prefixed (e.g., 'densenet121_baseline')
    if "densenet" in selected_type_norm:
        if selected_backbone is None:
            selected_backbone = "densenet121"
        if "region" in selected_type_norm or "aux" in selected_type_norm:
            selected_type_norm = "region_supervised"
        else:
            selected_type_norm = "baseline"

    if selected_backbone is None:
        selected_backbone = "resnet18"

    backbone_norm = selected_backbone.lower().strip()
    if backbone_norm in ["densenet121", "densenet", "dense121"]:
        canonical_backbone = "densenet121"
    elif backbone_norm in ["resnet18", "resnet", "resnet-18"]:
        canonical_backbone = "resnet18"
    else:
        raise ValueError(
            f"Unsupported backbone: '{selected_backbone}'. Must be 'resnet18' or 'densenet121'."
        )

    # Normalize canonical type
    if selected_type_norm in ["baseline", "resnet18", "resnet-18", "classification"]:
        canonical_type = "baseline"
    elif selected_type_norm in ["region_supervised", "region", "dual_head", "auxiliary"]:
        canonical_type = "region_supervised"
    else:
        raise ValueError(
            f"Unsupported model_type: '{selected_type}'. "
            f"Must be one of ['baseline', 'region_supervised']."
        )

    # Ensure deterministic initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.info(
        "Building model '%s' (backbone=%s, pretrained=%s, dropout=%.2f, seed=%d)",
        canonical_type,
        canonical_backbone,
        pretrained,
        dropout_rate,
        seed,
    )

    if canonical_type == "baseline":
        model = BaselineClassifier(
            pretrained=pretrained,
            in_channels=3,
            num_classes=1,
            dropout_rate=dropout_rate,
            backbone=canonical_backbone,
            memory_efficient=memory_efficient,
        )
    else:
        model = RegionSupervisedModel(
            pretrained=pretrained,
            in_channels=3,
            num_classes=1,
            dropout_rate=dropout_rate,
            target_size=target_size,
            backbone=canonical_backbone,
            memory_efficient=memory_efficient,
        )

    return model
