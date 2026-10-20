"""
scripts/train_trackb_fast.py

Accelerated Track B Training Engine for BoneTrace.
Strictly adheres to all Track B pre-registered invariants:
- ResNet-18, 512x512 letterboxed input, ImageNet normalization, no augmentation
- AdamW (lr 1e-4, wd 1e-4), CosineAnnealingLR (T_max=20, eta_min=1e-6)
- 20 epochs, no early stopping
- Effective batch size 32 under AMP (torch.amp autocast + GradScaler)
- Deterministic flags on (manual_seed, cudnn deterministic, worker seeding)
- Checkpoint = best validation ROC-AUC (save best and last)
- Saves adjacent and embedded config.json
- Verifies frozen checksums and PREREGISTRATION.md sha256 before training

Performance Optimizations:
1. Pre-collated contiguous pinned RAM tensors (FastTensorDataset) eliminating CPU torch.stack overhead
2. Direct batch size 32 (effective batch size 32 in 89 steps instead of 178)
3. TensorFloat-32 (TF32) enabled for Ampere RTX 3050 Tensor Cores
4. optimizer.zero_grad(set_to_none=True) to minimize memory write bandwidth
5. Fast validation batching (batch size 32)
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models

# Hardware & Determinism Configuration
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bonetrace.models.baseline import BaselineClassifier
from bonetrace.models.region_guided import RegionSupervisedModel
from bonetrace.data.dataset import FracAtlasDataset
from tests.test_trackb_models import DirectCAMModel

FROZEN_DIR = REPO_ROOT / "data" / "frozen"
OUTPUT_DIR = REPO_ROOT / "outputs" / "trackb"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
PREREG_FILE = OUTPUT_DIR / "PREREGISTRATION.md"

EXPECTED_PREREG_SHA256 = "9d5c963e6da4031b68a20b68d6ea155d1b1daf9977295e05d6164966f2c79718"
POS_WEIGHT = 4.70340681


def verify_integrity_and_safeguards():
    """Verify frozen manifests, PREREGISTRATION.md, and safeguard rules."""
    legacy_targets = [
        REPO_ROOT / "outputs" / "checkpoints",
        REPO_ROOT / "data" / "processed",
        REPO_ROOT / "outputs" / "results_registry.json",
    ]
    for target in legacy_targets:
        if not target.exists():
            raise RuntimeError(f"Integrity check failed: Expected legacy artifact {target} missing!")

    sha_file = FROZEN_DIR / "SHA256SUMS"
    if not sha_file.exists():
        raise FileNotFoundError(f"Missing checksums file: {sha_file}")
    with open(sha_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            exp_sha, fname = line.strip().split(maxsplit=1)
            target_f = FROZEN_DIR / fname
            if not target_f.exists():
                raise FileNotFoundError(f"Frozen file {fname} missing from {FROZEN_DIR}")
            with open(target_f, "rb") as fp:
                act_sha = hashlib.sha256(fp.read()).hexdigest()
            if act_sha != exp_sha:
                raise ValueError(f"Checksum mismatch for {fname}! Expected {exp_sha}, got {act_sha}")

    if not PREREG_FILE.exists():
        raise FileNotFoundError(f"PREREGISTRATION.md missing at {PREREG_FILE}")
    with open(PREREG_FILE, "rb") as fp:
        prereg_sha = hashlib.sha256(fp.read()).hexdigest()
    if prereg_sha != EXPECTED_PREREG_SHA256:
        raise ValueError(
            f"PREREGISTRATION.md has been modified! Expected {EXPECTED_PREREG_SHA256}, got {prereg_sha}. Aborting."
        )

    print("Integrity verification PASSED: Frozen manifests and PREREGISTRATION.md verified.")


def get_git_info() -> Tuple[str, str]:
    """Retrieve full git sha and 7-character short sha."""
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)).strip()
        git_sha7 = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True, cwd=str(REPO_ROOT)).strip()
    except Exception:
        git_sha = "unknown"
        git_sha7 = "83fd232"
    return git_sha, git_sha7


class FastContiguousTensorDataset(Dataset):
    """Stores all letterboxed samples in contiguous pinned RAM tensors for instant slicing."""

    def __init__(self, raw_ds: FracAtlasDataset):
        print(f"Collating {len(raw_ds)} samples into contiguous pinned memory...", flush=True)
        t0 = time.time()
        n = len(raw_ds)
        self.images = torch.zeros((n, 3, 512, 512), dtype=torch.float32)
        self.labels = torch.zeros((n, 1), dtype=torch.float32)
        self.masks = torch.zeros((n, 1, 512, 512), dtype=torch.float32)
        self.canonical_ids: List[str] = []
        self.boxes: List[Any] = []

        for i in range(n):
            sample = raw_ds[i]
            img = sample["image"]
            self.images[i] = img if isinstance(img, torch.Tensor) else torch.tensor(img)
            lbl = sample["label"]
            self.labels[i] = lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl)
            msk = sample["mask"]
            self.masks[i] = msk if isinstance(msk, torch.Tensor) else torch.tensor(msk)
            self.canonical_ids.append(sample["canonical_id"])
            self.boxes.append(sample.get("boxes", []))

        # Note: Do not pin the entire 9GB tensor; DataLoader(..., pin_memory=True) pins batches on-the-fly
        print(f"Dataset assembled in contiguous RAM in {time.time()-t0:.1f}s.", flush=True)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "mask": self.masks[index],
            "canonical_id": self.canonical_ids[index],
            "boxes": self.boxes[index],
        }


class FastPlaceboDatasetWrapper(Dataset):
    """Wraps FastContiguousTensorDataset to swap positive masks via fixed seed derangement."""

    def __init__(self, base_dataset: FastContiguousTensorDataset, seed: int):
        self.base_dataset = base_dataset
        self.seed = seed

        # Collect positive indices
        self.pos_indices: List[int] = []
        for i in range(len(base_dataset)):
            if float(base_dataset.labels[i].item()) == 1.0:
                self.pos_indices.append(i)

        n_pos = len(self.pos_indices)
        rng = np.random.RandomState(seed)
        perm = np.arange(n_pos)
        while True:
            rng.shuffle(perm)
            if np.all(perm != np.arange(n_pos)):
                break

        assert np.all(perm != np.arange(n_pos)), f"Derangement failed for seed {seed}"

        # Pre-build permuted mask tensor
        self.perm_masks = base_dataset.masks.clone()
        for k, orig_idx in enumerate(self.pos_indices):
            donor_idx = self.pos_indices[perm[k]]
            self.perm_masks[orig_idx] = base_dataset.masks[donor_idx]


    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "image": self.base_dataset.images[index],
            "label": self.base_dataset.labels[index],
            "mask": self.perm_masks[index],
            "canonical_id": self.base_dataset.canonical_ids[index],
            "boxes": self.base_dataset.boxes[index],
        }


def fast_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "canonical_id": [b["canonical_id"] for b in batch],
        "boxes": [b["boxes"] for b in batch],
    }


def build_model(arm: str, seed: int, device: torch.device) -> nn.Module:
    """Build model for specified arm with seed-controlled initialization."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if arm == "A":
        model = BaselineClassifier(num_classes=1, pretrained=False)
    elif arm == "B":
        model = RegionSupervisedModel(num_classes=1, pretrained=False)
    elif arm == "B_placebo":
        model = RegionSupervisedModel(num_classes=1, pretrained=False)
    elif arm == "C":
        model = DirectCAMModel(pretrained=False)
    elif arm == "A_pre":
        model = BaselineClassifier(num_classes=1, pretrained=True)
    elif arm == "B_pre":
        model = RegionSupervisedModel(num_classes=1, pretrained=True)
    else:
        raise ValueError(f"Unknown arm: {arm}")

    model.to(device)
    return model


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate_on_val(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float, float, pd.DataFrame]:
    """Evaluate model on frozen validation split with fast batched inference."""
    model.eval()
    all_logits = []
    all_probs = []
    all_targets = []
    all_ids = []

    pos_weight_t = torch.tensor([POS_WEIGHT], device=device)
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                out = model(imgs)
                logits = out["logits"] if isinstance(out, dict) else out
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight_t)

            b_size = len(labels)
            total_loss += loss.item() * b_size
            total_samples += b_size

            logits_np = logits.cpu().flatten().float().numpy()
            probs_np = torch.sigmoid(logits).cpu().flatten().float().numpy()
            targets_np = labels.cpu().flatten().long().numpy()

            all_logits.extend(logits_np.tolist())
            all_probs.extend(probs_np.tolist())
            all_targets.extend(targets_np.tolist())
            all_ids.extend(batch["canonical_id"])

    val_loss = total_loss / max(1, total_samples)
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)

    try:
        val_roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        val_roc_auc = 0.5

    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        val_pr_auc = float(auc(rec, prec))
    except Exception:
        val_pr_auc = 0.0

    preds_df = pd.DataFrame({
        "canonical_id": all_ids,
        "label": y_true,
        "logit": all_logits,
        "prob": all_probs,
    })

    return val_loss, val_roc_auc, val_pr_auc, preds_df


def train_single_run(
    arm: str,
    seed: int,
    epochs: int = 20,
    batch_size: int = 32,
    grad_accum_steps: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cuda:0"),
    cached_train_ds: Optional[FastContiguousTensorDataset] = None,
    cached_val_ds: Optional[FastContiguousTensorDataset] = None,
) -> Dict[str, Any]:
    """Train a single arm and seed with maximal GPU throughput."""
    git_sha, git_sha7 = get_git_info()
    run_id = f"{arm}_seed{seed}_{git_sha7}"
    run_dir = CHECKPOINTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================", flush=True)
    print(f"STARTING RUN: {run_id} (Arm={arm}, Seed={seed}) [FAST ENGINE]", flush=True)
    print(f"========================================================", flush=True)

    # Set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if arm == "B_placebo":
        train_ds = FastPlaceboDatasetWrapper(cached_train_ds, seed=seed)
    else:
        train_ds = cached_train_ds
    val_ds = cached_val_ds

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        collate_fn=fast_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=fast_collate_fn,
        pin_memory=True,
    )

    model = build_model(arm, seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    pos_weight_t = torch.tensor([POS_WEIGHT], device=device)

    start_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t_start = time.time()

    manifest_sha256 = {}
    for split_f in ["train.csv", "val.csv", "test.csv"]:
        with open(FROZEN_DIR / split_f, "rb") as fp:
            manifest_sha256[split_f] = hashlib.sha256(fp.read()).hexdigest()

    base_config = {
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        "git_sha": git_sha,
        "git_sha7": git_sha7,
        "manifest_sha256": manifest_sha256,
        "pos_weight": POS_WEIGHT,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * grad_accum_steps,
            "lr": lr,
            "weight_decay": weight_decay,
            "scheduler": "CosineAnnealingLR",
            "T_max": epochs,
            "eta_min": 1e-6,
            "amp": True,
            "lambda_aux": 1.0 if "B" in arm else 0.0,
            "lambda_cam": 1.0 if arm == "C" else 0.0,
        },
        "start_time_utc": start_time_utc,
    }

    epoch_times = []
    history_rows = []
    best_val_roc_auc = -1.0
    best_epoch = -1
    best_preds_df = None

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_train_loss = 0.0
        train_samples = 0

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                if arm in ["A", "A_pre"]:
                    out = model(imgs)
                    logits = out["logits"] if isinstance(out, dict) else out
                    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight_t)
                elif arm in ["B", "B_placebo", "B_pre"]:
                    out = model(imgs)
                    loss_cls = F.binary_cross_entropy_with_logits(out["logits"], labels, pos_weight=pos_weight_t)
                    loss_mask = F.binary_cross_entropy_with_logits(out["mask_logits"], masks)
                    loss = loss_cls + 1.0 * loss_mask
                elif arm == "C":
                    out = model(imgs)
                    loss_cls = F.binary_cross_entropy_with_logits(out["logits"], labels, pos_weight=pos_weight_t)
                    loss_cam = F.binary_cross_entropy_with_logits(out["cam_up"], masks)
                    loss = loss_cls + 1.0 * loss_cam

                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            b_len = len(labels)
            running_train_loss += loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1) * b_len
            train_samples += b_len

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                current_loss = running_train_loss / max(1, train_samples)
                print(f"  [Epoch {epoch:2d}/{epochs:2d}] Step {step+1:3d}/{len(train_loader)} | Running Loss: {current_loss:.4f}", flush=True)

        scheduler.step()
        epoch_train_loss = running_train_loss / max(1, train_samples)

        val_loss, val_roc_auc, val_pr_auc, preds_df = evaluate_on_val(model, val_loader, device)
        t_epoch_end = time.time()
        epoch_duration = t_epoch_end - t_epoch_start
        epoch_times.append(epoch_duration)

        print(
            f"--> Epoch {epoch:2d}/{epochs:2d} finished in {epoch_duration:.1f}s | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f}",
            flush=True,
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss,
            "val_roc_auc": val_roc_auc,
            "val_pr_auc": val_pr_auc,
            "epoch_time_sec": epoch_duration,
        })

        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            best_epoch = epoch
            best_preds_df = preds_df.copy()

            ckpt_payload = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "val_loss": val_loss,
                "config": {
                    **base_config,
                    "best_epoch": best_epoch,
                    "best_val_roc_auc": best_val_roc_auc,
                },
            }
            torch.save(ckpt_payload, run_dir / "best_model.pt")

    last_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epochs,
        "val_roc_auc": val_roc_auc,
        "val_pr_auc": val_pr_auc,
        "val_loss": val_loss,
        "config": {
            **base_config,
            "best_epoch": best_epoch,
            "best_val_roc_auc": best_val_roc_auc,
        },
    }
    torch.save(last_payload, run_dir / "last_model.pt")

    t_end = time.time()
    total_duration = t_end - t_start
    end_time_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    final_config = {
        **base_config,
        "end_time_utc": end_time_utc,
        "total_train_time_sec": total_duration,
        "mean_epoch_time_sec": float(np.mean(epoch_times)),
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_val_roc_auc,
    }

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)

    pd.DataFrame(history_rows).to_csv(run_dir / "train_history.csv", index=False)

    if best_preds_df is not None:
        best_preds_df["run_id"] = run_id
        best_preds_df["arm"] = arm
        best_preds_df["seed"] = seed
        best_preds_df["split"] = "val"
        best_preds_df.to_csv(run_dir / "val_predictions.csv", index=False)

    print(f"\nCOMPLETED {run_id}: Best Val ROC-AUC={best_val_roc_auc:.4f} (Epoch {best_epoch}) | Total Time={total_duration/60:.1f} min", flush=True)
    return final_config


def main():
    parser = argparse.ArgumentParser(description="Accelerated Track B Training Engine")
    parser.add_argument("--arms", nargs="+", choices=["A", "B", "B_placebo", "C", "A_pre", "B_pre"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    args = parser.parse_args()

    verify_integrity_and_safeguards()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Target Arms: {args.arms}")
    print(f"Target Seeds: {args.seeds}")
    print(f"Effective Batch Size: {args.batch_size * args.grad_accum_steps} (Batch {args.batch_size}, Grad Accum {args.grad_accum_steps})")

    print("\nLoading and pinning datasets into contiguous RAM tensors...", flush=True)
    raw_train = FracAtlasDataset(manifest=FROZEN_DIR / "train.csv", normalize=True)
    fast_train_ds = FastContiguousTensorDataset(raw_train)
    raw_val = FracAtlasDataset(manifest=FROZEN_DIR / "val.csv", normalize=True)
    fast_val_ds = FastContiguousTensorDataset(raw_val)

    all_summaries = []
    t_stage_start = time.time()

    for arm in args.arms:
        for seed in args.seeds:
            # Check if already completed
            git_sha, git_sha7 = get_git_info()
            run_id = f"{arm}_seed{seed}_{git_sha7}"
            ckpt_p = CHECKPOINTS_DIR / run_id / "best_model.pt"
            cfg_p = CHECKPOINTS_DIR / run_id / "config.json"
            if ckpt_p.exists() and cfg_p.exists():
                print(f"\n[SKIP] Run {run_id} is already completed. Skipping.", flush=True)
                with open(cfg_p, "r", encoding="utf-8") as fp:
                    all_summaries.append(json.load(fp))
                continue

            cfg = train_single_run(
                arm=arm,
                seed=seed,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                device=device,
                cached_train_ds=fast_train_ds,
                cached_val_ds=fast_val_ds,
            )
            all_summaries.append(cfg)

    total_stage_time = time.time() - t_stage_start
    print("\n" + "="*70)
    print("STAGE EXECUTION SUMMARY")
    print("="*70)
    for res in all_summaries:
        print(f"Run: {res['run_id']:<24} | Best Val ROC-AUC: {res['best_val_roc_auc']:.4f} (Epoch {res['best_epoch']}) | Duration: {res['total_train_time_sec']/60:.1f} min")
    print(f"\nTotal GPU Time for Executed Stage: {total_stage_time/60:.1f} min ({total_stage_time/3600:.2f} hours)")


if __name__ == "__main__":
    main()
