"""Training infrastructure for BoneTrace.

Implements the Trainer class and run_training entrypoint for controlled training
of baseline (classification-only) and region-supervised (dual-head) models.
Features:
- Reproducible deterministic seeding across Python, NumPy, and PyTorch.
- Configurable loss weighting accounting for FracAtlas 17.6% class imbalance (~4.6:1 pos_weight).
- Separate classification loss and auxiliary region localization loss logging.
- Comprehensive validation metrics: ROC-AUC, PR-AUC, Precision, Recall, F1, Balanced Accuracy, Confusion Matrix.
- Checkpoint persistence (latest_checkpoint.pt, best_model.pt) and machine-readable training logs.
- Strict isolation of test set (validation-set-only checkpoint selection).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bonetrace.config import AppConfig
from bonetrace.data.dataset import FracAtlasDataset, fracatlas_collate_fn
from bonetrace.logging_utils import get_logger
from bonetrace.models.factory import build_model

logger = get_logger("training.trainer")


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """Orchestrates model optimization, loss tracking, validation, and checkpointing."""

    def __init__(
        self,
        config: AppConfig,
        model: Optional[nn.Module] = None,
        model_type: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        pos_weight: Optional[float] = None,
    ) -> None:
        """Initialize Trainer.

        Args:
            config: Application configuration.
            model: Optional instantiated model. If None, builds model via factory.
            model_type: Optional model type ('baseline' or 'region_supervised').
            device: Optional torch device override ('cpu' or 'cuda').
            pos_weight: Optional positive class loss weight override.
        """
        self.config = config
        raw_training = config.raw_config.get("training", {}) if hasattr(config, "raw_config") else {}

        # 1. Device selection
        if device is not None:
            self.device = torch.device(device)
        else:
            cfg_device = getattr(config.project, "device", "cpu")
            if cfg_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested in configuration but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(cfg_device if torch.cuda.is_available() else "cpu")

        # 2. Seed initialization
        self.seed = getattr(config.project, "seed", 42)
        set_seed(self.seed)

        # 3. Model construction
        if model is not None:
            self.model = model
        else:
            self.model = build_model(config=config, model_type=model_type)

        self.model.to(self.device)
        self.model_type = getattr(self.model, "model_type", "baseline")
        self.is_region_supervised = (self.model_type == "region_supervised")

        # 4. Hyperparameters
        self.batch_size = int(raw_training.get("batch_size", 16))
        self.num_epochs = int(raw_training.get("num_epochs", 20))
        self.learning_rate = float(raw_training.get("learning_rate", 1e-4))
        self.weight_decay = float(raw_training.get("weight_decay", 1e-4))
        self.aux_loss_weight = float(raw_training.get("aux_loss_weight", 1.0))

        # Class imbalance handling: default ~4.616 derived from FracAtlas train split (2336 neg / 506 pos)
        if pos_weight is not None:
            self.pos_weight = float(pos_weight)
        else:
            self.pos_weight = float(raw_training.get("pos_weight", 4.6166))

        # 5. Loss criteria
        self.pos_weight_tensor = torch.tensor([self.pos_weight], device=self.device)
        self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        self.aux_criterion = nn.BCEWithLogitsLoss()

        # 6. Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.num_epochs),
            eta_min=self.learning_rate * 0.01,
        )

        # 7. Checkpoints directory
        self.checkpoints_dir = Path(config.paths.checkpoints_dir) / self.model_type
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_auc = 0.0
        self.best_epoch = -1
        self.current_epoch = 0
        self.training_history: List[Dict[str, Any]] = []

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute primary classification loss and auxiliary region supervision loss.

        Args:
            outputs: Dictionary from model forward pass ('logits', optional 'mask_logits').
            batch: Dictionary from dataset loader ('label', 'mask', 'has_annotation').

        Returns:
            Tuple of (total_loss tensor, dictionary of loss scalar values).
        """
        cls_logits = outputs["logits"]
        labels = batch["label"].to(self.device)

        cls_loss = self.cls_criterion(cls_logits, labels)
        aux_loss = torch.tensor(0.0, device=self.device)

        if self.is_region_supervised and "mask_logits" in outputs:
            mask_logits = outputs["mask_logits"]
            target_masks = batch["mask"].to(self.device)
            has_ann = batch["has_annotation"].to(self.device)

            # Supervision mask: supervise negative images (zero mask) and positive images with verified annotations.
            # Mask out unannotated positive images so missing annotations are not treated as true negative regions.
            valid_mask = (labels.squeeze(-1) < 0.5) | has_ann
            if valid_mask.any():
                aux_loss = self.aux_criterion(mask_logits[valid_mask], target_masks[valid_mask])
            else:
                aux_loss = torch.tensor(0.0, device=self.device)

            total_loss = cls_loss + (self.aux_loss_weight * aux_loss)
        else:
            total_loss = cls_loss

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "cls_loss": float(cls_loss.item()),
            "aux_loss": float(aux_loss.item()),
        }
        return total_loss, loss_dict

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one complete training epoch across dataloader batches."""
        self.model.train()
        total_loss_accum = 0.0
        cls_loss_accum = 0.0
        aux_loss_accum = 0.0
        n_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            total_loss, loss_dict = self.compute_loss(outputs, batch)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss_accum += loss_dict["total_loss"]
            cls_loss_accum += loss_dict["cls_loss"]
            aux_loss_accum += loss_dict["aux_loss"]
            n_batches += 1

        if n_batches == 0:
            return {"loss": 0.0, "cls_loss": 0.0, "aux_loss": 0.0}

        return {
            "loss": total_loss_accum / n_batches,
            "cls_loss": cls_loss_accum / n_batches,
            "aux_loss": aux_loss_accum / n_batches,
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate validation metrics without computing gradients."""
        self.model.eval()
        total_loss_accum = 0.0
        cls_loss_accum = 0.0
        aux_loss_accum = 0.0
        n_batches = 0

        all_probs: List[float] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                outputs = self.model(images)
                _, loss_dict = self.compute_loss(outputs, batch)

                logits = outputs["logits"]
                probs = torch.sigmoid(logits).cpu().flatten().tolist()
                lbls = batch["label"].cpu().flatten().long().tolist()

                all_probs.extend(probs)
                all_labels.extend(lbls)

                total_loss_accum += loss_dict["total_loss"]
                cls_loss_accum += loss_dict["cls_loss"]
                aux_loss_accum += loss_dict["aux_loss"]
                n_batches += 1

        val_total_loss = (total_loss_accum / n_batches) if n_batches > 0 else 0.0
        val_cls_loss = (cls_loss_accum / n_batches) if n_batches > 0 else 0.0
        val_aux_loss = (aux_loss_accum / n_batches) if n_batches > 0 else 0.0

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = (y_prob >= 0.5).astype(int)

        # Robust metric calculation handling edge cases (e.g. single class)
        metrics: Dict[str, float] = {
            "val_loss": float(val_total_loss),
            "val_cls_loss": float(val_cls_loss),
            "val_aux_loss": float(val_aux_loss),
        }

        # ROC-AUC
        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                metrics["roc_auc"] = 0.5
            try:
                metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
            except Exception:
                metrics["pr_auc"] = 0.0
        else:
            metrics["roc_auc"] = 0.5
            metrics["pr_auc"] = 0.0

        # Precision, Recall, F1
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics["precision"] = float(prec)
        metrics["recall"] = float(rec)
        metrics["f1"] = float(f1)

        # Balanced Accuracy
        try:
            metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            metrics["balanced_acc"] = 0.5

        # Confusion Matrix
        if len(y_true) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            metrics["tn"] = float(tn)
            metrics["fp"] = float(fp)
            metrics["fn"] = float(fn)
            metrics["tp"] = float(tp)
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            metrics["tn"] = metrics["fp"] = metrics["fn"] = metrics["tp"] = metrics["specificity"] = 0.0

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None,
    ) -> Path:
        """Persist model state, optimizer state, and training metrics."""
        state = {
            "epoch": epoch,
            "model_type": self.model_type,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_val_auc": self.best_val_auc,
            "config": self.config.raw_config if hasattr(self.config, "raw_config") else {},
        }
        if filename is None:
            filename = "best_model.pt" if is_best else "latest_checkpoint.pt"

        save_path = self.checkpoints_dir / filename
        torch.save(state, save_path)
        logger.info("Saved checkpoint: %s (epoch %d, val_auc=%.4f)", save_path, epoch, metrics.get("roc_auc", 0.0))
        return save_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load weights and state from checkpoint file."""
        cpath = Path(checkpoint_path)
        if not cpath.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {cpath}")

        checkpoint = torch.load(cpath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        logger.info("Loaded checkpoint from %s (resumed from epoch %d)", cpath, self.current_epoch)
        return checkpoint

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[Dict[str, Any]]:
        """Execute full training and validation lifecycle across configured epochs."""
        logger.info(
            "Starting training for '%s' (%d epochs, batch_size=%d, lr=%.5f, device=%s)",
            self.model_type,
            self.num_epochs,
            self.batch_size,
            self.learning_rate,
            self.device,
        )

        history_csv_path = self.checkpoints_dir / "train_history.csv"
        metrics_json_path = self.checkpoints_dir / "final_metrics.json"

        fieldnames = [
            "epoch",
            "train_loss",
            "train_cls_loss",
            "train_aux_loss",
            "val_loss",
            "val_cls_loss",
            "val_aux_loss",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "balanced_acc",
            "lr",
        ]

        with open(history_csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()

            for epoch in range(1, self.num_epochs + 1):
                self.current_epoch = epoch
                train_m = self.train_epoch(train_loader)
                val_m = self.validate(val_loader)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()

                epoch_record = {
                    "epoch": epoch,
                    "train_loss": train_m["loss"],
                    "train_cls_loss": train_m["cls_loss"],
                    "train_aux_loss": train_m["aux_loss"],
                    "val_loss": val_m["val_loss"],
                    "val_cls_loss": val_m["val_cls_loss"],
                    "val_aux_loss": val_m["val_aux_loss"],
                    "roc_auc": val_m.get("roc_auc", 0.0),
                    "pr_auc": val_m.get("pr_auc", 0.0),
                    "precision": val_m.get("precision", 0.0),
                    "recall": val_m.get("recall", 0.0),
                    "f1": val_m.get("f1", 0.0),
                    "balanced_acc": val_m.get("balanced_acc", 0.0),
                    "lr": current_lr,
                }
                self.training_history.append(epoch_record)
                writer.writerow(epoch_record)
                f_csv.flush()

                logger.info(
                    "Epoch %02d/%02d - Train Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f | Val F1: %.4f",
                    epoch,
                    self.num_epochs,
                    train_m["loss"],
                    val_m["val_loss"],
                    val_m.get("roc_auc", 0.0),
                    val_m.get("f1", 0.0),
                )

                # Save latest checkpoint
                self.save_checkpoint(epoch, val_m, is_best=False, filename="latest_checkpoint.pt")

                # Check best model on validation ROC-AUC
                val_auc = val_m.get("roc_auc", 0.0)
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, val_m, is_best=True, filename="best_model.pt")

        final_summary = {
            "model_type": self.model_type,
            "best_epoch": self.best_epoch,
            "best_val_auc": self.best_val_auc,
            "total_epochs": self.num_epochs,
            "final_metrics": self.training_history[-1] if self.training_history else {},
        }
        with open(metrics_json_path, "w", encoding="utf-8") as f_json:
            json.dump(final_summary, f_json, indent=2)

        logger.info("Training complete. Best model at epoch %d with val_auc=%.4f", self.best_epoch, self.best_val_auc)
        return self.training_history


def run_training(config: AppConfig, model_type: Optional[str] = None) -> Trainer:
    """Entry point for executing model training experiments from CLI or scripts.

    Initializes data pipelines, model architecture, optimizer, and runs training loops.

    Args:
        config: Application configuration.
        model_type: Optional model type ('baseline' or 'region_supervised').

    Returns:
        Trainer instance ready for training execution.
    """
    logger.info("Setting up BoneTrace training experiment...")
    raw_training = config.raw_config.get("training", {}) if hasattr(config, "raw_config") else {}

    if model_type is None:
        model_type = raw_training.get("model_type", "baseline")

    # Resolve manifest files
    train_csv = config.paths.processed_data_dir / "train.csv"
    val_csv = config.paths.processed_data_dir / "val.csv"

    if not train_csv.is_file() or not val_csv.is_file():
        raise FileNotFoundError(
            f"Processed split files not found ({train_csv}, {val_csv}). "
            f"Please run 'bonetrace preprocess' first."
        )

    # Instantiate datasets with auditable conflict exclusion policy
    train_dataset = FracAtlasDataset(
        manifest=train_csv,
        split="train",
        exclude_conflicts=True,
    )
    val_dataset = FracAtlasDataset(
        manifest=val_csv,
        split="val",
        exclude_conflicts=True,
    )

    batch_size = int(raw_training.get("batch_size", 16))
    num_workers = int(raw_training.get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=fracatlas_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=fracatlas_collate_fn,
    )

    trainer = Trainer(
        config=config,
        model_type=model_type,
        pos_weight=train_dataset.get_pos_weight(),
    )

    # Check if dry_run requested or zero epochs
    dry_run = bool(raw_training.get("dry_run", False))
    if dry_run or trainer.num_epochs == 0:
        logger.info("Dry-run requested: skipping full epoch iterations.")
        return trainer

    # Execute full training
    trainer.fit(train_loader, val_loader)
    return trainer
