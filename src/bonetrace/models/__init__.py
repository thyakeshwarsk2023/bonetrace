"""Model definitions and factory for BoneTrace."""

from bonetrace.models.baseline import BaselineClassifier
from bonetrace.models.factory import build_model
from bonetrace.models.region_guided import RegionSupervisedModel

__all__ = ["BaselineClassifier", "RegionSupervisedModel", "build_model"]
