"""Training package for BoneTrace (training loops, optimization, checkpointing)."""

from bonetrace.training.trainer import Trainer, run_training

__all__ = ["Trainer", "run_training"]
