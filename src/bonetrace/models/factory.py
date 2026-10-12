"""Model architecture factory for BoneTrace.

Instantiates baseline classification and region-supervised models
using consistent backbone configurations and deterministic seed initialization.
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
) -> nn.Module:
    """Build and return a model instance based on configuration and model type.

    Args:
        config: Optional AppConfig object.
        model_type: Explicit model type override ('baseline' or 'region_supervised').
        pretrained: Whether to load ImageNet pretrained weights (default: False for deterministic CPU execution).
        dropout_rate: Dropout rate for classification head.
        target_size: Image spatial resolution (width, height).

    Returns:
        Instantiated nn.Module (BaselineClassifier or RegionSupervisedModel).

    Raises:
        ValueError: If model_type is unsupported.
    """
    # 1. Determine model_type
    selected_type = model_type
    seed = 42

    if config is not None:
        seed = getattr(config.project, "seed", 42)
        raw_training = config.raw_config.get("training", {}) if hasattr(config, "raw_config") else {}

        if selected_type is None:
            # Check config hierarchy
            selected_type = (
                raw_training.get("model_type")
                or raw_training.get("model_name")
                or getattr(getattr(config, "training", None), "model_type", None)
                or getattr(getattr(config, "training", None), "model_name", None)
                or "baseline"
            )

        if "pretrained" in raw_training:
            pretrained = bool(raw_training["pretrained"])
        if "dropout_rate" in raw_training:
            dropout_rate = float(raw_training["dropout_rate"])

    if selected_type is None:
        selected_type = "baseline"

    # Normalize name (e.g. resnet18 / baseline -> baseline; region_supervised / aux -> region_supervised)
    selected_type_norm = selected_type.lower().strip()
    if selected_type_norm in ["baseline", "resnet18", "resnet-18", "classification"]:
        canonical_type = "baseline"
    elif selected_type_norm in ["region_supervised", "region", "dual_head", "auxiliary"]:
        canonical_type = "region_supervised"
    else:
        raise ValueError(
            f"Unsupported model_type: '{selected_type}'. "
            f"Must be one of ['baseline', 'region_supervised']."
        )

    # 2. Ensure deterministic initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.info(
        "Building model '%s' (pretrained=%s, dropout=%.2f, seed=%d)",
        canonical_type,
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
        )
    else:
        model = RegionSupervisedModel(
            pretrained=pretrained,
            in_channels=3,
            num_classes=1,
            dropout_rate=dropout_rate,
            target_size=target_size,
        )

    return model
