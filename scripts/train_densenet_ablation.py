#!/usr/bin/env python
"""
scripts/train_densenet_ablation.py

Controlled Backbone Ablation Study: DenseNet-121 (Baseline vs Region-Supervised).
Strictly adheres to the matched BoneTrace experimental protocol:
- Backbone: torchvision.models.densenet121 (trained from scratch)
- Input: 512x512 letterboxed radiographs with ImageNet normalization
- Optimization: AdamW (lr=1e-4, weight_decay=1e-4), CosineAnnealingLR (T_max=20, eta_min=1e-6)
- Imbalance: pos_weight = 4.70340681 on binary classification BCEWithLogitsLoss
- Auxiliary supervision: spatial BCEWithLogitsLoss with lambda_aux = 1.0 for Model B (region)
- Precision: Automatic Mixed Precision (AMP) with GradScaler on CUDA
- Batch size: 16 matched across both Model A and Model B
- Checkpoint: saved on best validation ROC-AUC
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bonetrace.data.dataset import FracAtlasDataset, fracatlas_collate_fn
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.models.baseline import BaselineClassifier
from bonetrace.models.region_guided import RegionSupervisedModel

# Protocol Constants
FROZEN_DIR = REPO_ROOT / "data" / "frozen"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CHECKPOINTS_DIR = REPO_ROOT / "outputs" / "checkpoints"
POS_WEIGHT = 4.70340681  # Exact ratio: 2347 negative / 499 positive on train split


def set_seed(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_split_path(filename: str) -> Path:
    """Resolve path to split manifest, prioritizing frozen audited splits."""
    if (FROZEN_DIR / filename).exists():
        return FROZEN_DIR / filename
    elif (PROCESSED_DIR / filename).exists():
        return PROCESSED_DIR / filename
    else:
        raise FileNotFoundError(f"Manifest '{filename}' not found in {FROZEN_DIR} or {PROCESSED_DIR}.")


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    pos_weight: float = POS_WEIGHT,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Evaluate model on validation split.

    Returns:
        Tuple of (val_loss, val_roc_auc, val_pr_auc).
    """
    model.eval()
    pos_weight_t = torch.tensor([pos_weight], device=device)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    all_logits: List[float] = []
    all_targets: List[int] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                outputs = model(images)
                cls_logits = outputs["logits"]
                loss = cls_criterion(cls_logits, labels)

            b_size = labels.size(0)
            total_loss += loss.item() * b_size
            total_samples += b_size

            logits_np = cls_logits.cpu().flatten().float().tolist()
            targets_np = labels.cpu().flatten().long().tolist()
            all_logits.extend(logits_np)
            all_targets.extend(targets_np)

    val_loss = total_loss / max(1, total_samples)
    y_true = np.array(all_targets)
    y_scores = 1.0 / (1.0 + np.exp(-np.array(all_logits)))  # sigmoid

    # Compute ROC-AUC and PR-AUC
    if len(np.unique(y_true)) > 1:
        try:
            val_roc_auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            val_roc_auc = 0.5
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            val_pr_auc = float(auc(rec, prec))
        except Exception:
            val_pr_auc = 0.0
    else:
        val_roc_auc = 0.5
        val_pr_auc = 0.0

    return val_loss, val_roc_auc, val_pr_auc


def run_gradcam_diagnostic(model: nn.Module, device: torch.device) -> Tuple[bool, Tuple[int, int], str]:
    """Verify Grad-CAM forward/backward hook extraction on a single synthetic sample.

    Returns:
        Tuple of (success_flag, heatmap_shape, diagnostic_message).
    """
    model.eval()
    try:
        extractor = GradCAM(model=model, target_size=(512, 512))
        dummy_input = torch.randn(1, 3, 512, 512, device=device)
        heatmap = extractor.generate(dummy_input, class_idx=0)
        shape = heatmap.shape
        extractor.remove_hooks()

        if shape == (512, 512) and not np.isnan(heatmap).any():
            return True, shape, "Grad-CAM extraction verified successfully (shape 512x512, no NaNs)."
        else:
            return False, shape, f"Unexpected Grad-CAM output: shape={shape}, NaNs={np.isnan(heatmap).any()}"
    except Exception as e:
        return False, (0, 0), f"Grad-CAM extraction failed: {str(e)}"


def train_single_arm(
    model_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    smoke_test: bool,
    device: torch.device,
    train_manifest: Path,
    val_manifest: Path,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """Train a single model arm (baseline or region) for a specific seed."""
    print(f"\n{'='*70}", flush=True)
    mode_str = "SMOKE TEST (2 batches)" if smoke_test else f"FULL RUN ({epochs} epochs)"
    print(f"STARTING DENSENET-121: model={model_name.upper()} | seed={seed} | mode={mode_str}", flush=True)
    print(f"{'='*70}", flush=True)

    set_seed(seed)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate Datasets & DataLoaders
    train_dataset = FracAtlasDataset(manifest=train_manifest, normalize=True)
    val_dataset = FracAtlasDataset(manifest=val_manifest, normalize=True)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        collate_fn=fracatlas_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=fracatlas_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Build Model
    is_region = (model_name.lower() in ["region", "region_supervised"])
    if is_region:
        model = RegionSupervisedModel(
            backbone="densenet121",
            pretrained=False,
            target_size=(512, 512),
            memory_efficient=True,
        )
    else:
        model = BaselineClassifier(
            backbone="densenet121",
            pretrained=False,
            memory_efficient=True,
        )

    model.to(device)

    # 3. Setup Optimizer, Scheduler, Loss, and Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    pos_weight_t = torch.tensor([POS_WEIGHT], device=device)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    aux_criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_epoch = -1
    best_pr_auc = 0.0

    if smoke_test:
        ckpt_filename = f"smoke_test_densenet121_{model_name}.pt"
    else:
        ckpt_filename = f"densenet121_{model_name}_seed{seed}.pt"
    ckpt_path = CHECKPOINTS_DIR / ckpt_filename

    # Diagnostics verification storage
    diagnostics: Dict[str, Any] = {}

    max_train_batches = 2 if smoke_test else None
    max_val_batches = 2 if smoke_test else None
    actual_epochs = 1 if smoke_test else epochs

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, actual_epochs + 1):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_aux_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{actual_epochs:02d} [{model_name} s{seed}]",
            total=max_train_batches or len(train_loader),
            leave=True,
        )

        for step, batch in enumerate(pbar):
            if max_train_batches is not None and step >= max_train_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            has_ann = batch.get("has_annotation")
            if has_ann is not None:
                has_ann = has_ann.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                # Forward pass
                outputs = model(images)
                cls_logits = outputs["logits"]
                cls_loss = cls_criterion(cls_logits, labels)

                if is_region:
                    mask_logits = outputs["mask_logits"]
                    if has_ann is not None:
                        valid_mask = (labels.squeeze(-1) < 0.5) | has_ann
                    else:
                        valid_mask = torch.ones(labels.size(0), dtype=torch.bool, device=device)

                    if valid_mask.any():
                        aux_loss = aux_criterion(mask_logits[valid_mask], masks[valid_mask])
                    else:
                        aux_loss = torch.tensor(0.0, device=device)
                    total_loss = cls_loss + (1.0 * aux_loss)
                else:
                    aux_loss = torch.tensor(0.0, device=device)
                    total_loss = cls_loss

            # Backward and optimizer step with AMP GradScaler
            if use_amp and device.type == "cuda":
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            # Record diagnostic shapes and values on first step
            if step == 0 and epoch == 1:
                diagnostics["feature_shape"] = tuple(outputs["features"].shape)
                diagnostics["cls_logit_shape"] = tuple(cls_logits.shape)
                if is_region:
                    diagnostics["aux_mask_shape"] = tuple(outputs["mask_logits"].shape)
                diagnostics["amp_loss_finite"] = bool(torch.isfinite(total_loss).item())
                diagnostics["initial_loss"] = float(total_loss.item())

            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_aux_loss += aux_loss.item()
            n_batches += 1

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}",
                aux=f"{aux_loss.item():.4f}",
                lr=f"{current_lr:.1e}",
            )

        scheduler.step()
        train_loss = running_loss / max(1, n_batches)

        # Validation phase
        val_loss, val_roc_auc, val_pr_auc = evaluate_validation(
            model=model,
            val_loader=val_loader,
            device=device,
            use_amp=use_amp,
            pos_weight=POS_WEIGHT,
            max_batches=max_val_batches,
        )

        print(
            f"Epoch {epoch:02d} Summary: Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | Val ROC-AUC={val_roc_auc:.4f} | Val PR-AUC={val_pr_auc:.4f}",
            flush=True,
        )

        # Save checkpoint if improved or during smoke test
        if val_roc_auc > best_val_auc or smoke_test:
            best_val_auc = val_roc_auc
            best_pr_auc = val_pr_auc
            best_epoch = epoch

            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": best_epoch,
                "val_roc_auc": best_val_auc,
                "val_pr_auc": best_pr_auc,
                "val_loss": val_loss,
                "seed": seed,
                "model_type": model_name,
                "backbone": "densenet121",
                "hyperparameters": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "pos_weight": POS_WEIGHT,
                    "lambda_aux": 1.0 if is_region else 0.0,
                    "amp": use_amp,
                    "smoke_test": smoke_test,
                },
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            torch.save(checkpoint_payload, ckpt_path)
            print(f"--> Saved best checkpoint to: {ckpt_path.name} (Val AUC={best_val_auc:.4f})", flush=True)

    # 4. Measure Peak VRAM
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    # 5. Run Grad-CAM Hook Diagnostic
    cam_ok, cam_shape, cam_msg = run_gradcam_diagnostic(model=model, device=device)

    diagnostics.update({
        "model_name": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_pr_auc": best_pr_auc,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_path": str(ckpt_path),
        "gradcam_success": cam_ok,
        "gradcam_shape": cam_shape,
        "gradcam_message": cam_msg,
    })

    return diagnostics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Controlled DenseNet-121 Backbone Ablation Training for BoneTrace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["baseline", "region"],
        choices=["baseline", "region"],
        help="Model variants to train.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
        help="Random seeds for matched multi-seed training runs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Matched batch size across baseline and region models.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW base learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (AMP FP16) with GradScaler.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable Automatic Mixed Precision.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast 2-batch smoke test (1 epoch, 2 train batches, 2 val batches).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target PyTorch device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker subprocess count (0 recommended for Windows/smoke-test).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    device = torch.device(args.device)

    train_path = resolve_split_path("train.csv")
    val_path = resolve_split_path("val.csv")

    print("\n" + "=" * 70)
    print("BONETRACE DENSENET-121 ABLATION TRAINING ENGINE")
    print("=" * 70)
    print(f"Device:            {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Models:            {args.models}")
    print(f"Seeds:             {args.seeds}")
    print(f"Epochs:            {args.epochs if not args.smoke_test else '1 (Smoke Test)'}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Learning Rate:     {args.lr}")
    print(f"AMP Enabled:       {args.amp}")
    print(f"Smoke Test Mode:   {args.smoke_test}")
    print(f"Train Manifest:    {train_path}")
    print(f"Val Manifest:      {val_path}")
    print("=" * 70)

    all_diagnostics = []

    for seed in args.seeds:
        for model_name in args.models:
            diag = train_single_arm(
                model_name=model_name,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                use_amp=args.amp,
                smoke_test=args.smoke_test,
                device=device,
                train_manifest=train_path,
                val_manifest=val_path,
                num_workers=args.num_workers,
            )
            all_diagnostics.append(diag)

    # Print Diagnostic Summary Table
    print("\n" + "=" * 70)
    print("ABLATION RUN DIAGNOSTIC SUMMARY")
    print("=" * 70)
    for d in all_diagnostics:
        m = d["model_name"].upper()
        s = d["seed"]
        print(f"[{m} - Seed {s}]")
        if "feature_shape" in d:
            print(f"  * Spatial Feature Map Shape: {d['feature_shape']} (stride 32)")
        if "cls_logit_shape" in d:
            print(f"  * Classification Logit Shape: {d['cls_logit_shape']}")
        if "aux_mask_shape" in d:
            print(f"  * Auxiliary Mask Logit Shape: {d['aux_mask_shape']}")
        print(f"  * Loss Finite:                {d.get('amp_loss_finite', True)} (initial: {d.get('initial_loss', 0.0):.4f})")
        print(f"  * Peak CUDA VRAM:             {d['peak_vram_mb']:.2f} MB (RTX 3050 limit: 4,096 MB)")
        print(f"  * Grad-CAM Hook Extraction:   {'PASS' if d['gradcam_success'] else 'FAIL'} {d['gradcam_shape']}")
        print(f"  * Checkpoint Saved:           {d['checkpoint_path']}")
        print(f"  * Best Val ROC-AUC:           {d['best_val_auc']:.4f}")
        print("-" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
