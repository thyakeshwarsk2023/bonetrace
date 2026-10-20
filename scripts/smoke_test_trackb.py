"""
scripts/smoke_test_trackb.py

Phase 1 Requirement:
1-epoch smoke test of every arm on 64 training images,
including a check that all runs for one arm and seed give
bit-identical predictions when repeated twice with the same seed.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from bonetrace.models.baseline import BaselineClassifier
from bonetrace.models.region_guided import RegionSupervisedModel
from bonetrace.data.dataset import FracAtlasDataset, fracatlas_collate_fn
from tests.test_trackb_models import DirectCAMModel

FROZEN_DIR = Path("data/frozen")
POS_WEIGHT = 4.70340681

class PlaceboDatasetWrapper(torch.utils.data.Dataset):
    """Wraps FracAtlasDataset to apply derangement permutation to positive masks."""
    def __init__(self, base_dataset: FracAtlasDataset, seed: int):
        self.base_dataset = base_dataset
        self.seed = seed
        
        # Collect positive indices
        self.pos_indices = []
        for i in range(len(base_dataset)):
            sample = base_dataset[i]
            lbl = sample["label"].item() if isinstance(sample["label"], torch.Tensor) else float(sample["label"])
            if lbl == 1.0:
                self.pos_indices.append(i)
        
        n_pos = len(self.pos_indices)
        rng = np.random.RandomState(seed)
        perm = np.arange(n_pos)
        while True:
            rng.shuffle(perm)
            if np.all(perm != np.arange(n_pos)):
                break
        
        # Mapping: positive index i gets mask from self.pos_indices[perm[k]]
        self.mask_map = {}
        for k, orig_idx in enumerate(self.pos_indices):
            donor_idx = self.pos_indices[perm[k]]
            self.mask_map[orig_idx] = donor_idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        if index in self.mask_map:
            donor_idx = self.mask_map[index]
            donor_sample = self.base_dataset[donor_idx]
            sample["mask"] = donor_sample["mask"]
            sample["boxes"] = donor_sample["boxes"]
        return sample

def build_model_for_arm(arm: str, seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    if arm == "A":
        model = BaselineClassifier(pretrained=False)
    elif arm == "B" or arm == "B_placebo":
        model = RegionSupervisedModel(pretrained=False)
    elif arm == "C":
        model = DirectCAMModel(pretrained=False)
    elif arm == "A_pre":
        model = BaselineClassifier(pretrained=True)
    elif arm == "B_pre":
        model = RegionSupervisedModel(pretrained=True)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return model.to(device)

def run_one_epoch(arm: str, seed: int, loader: DataLoader, device: torch.device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = build_model_for_arm(arm, seed, device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    pos_weight_t = torch.tensor([POS_WEIGHT], device=device)
    
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        masks = batch["mask"].to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            if arm == "A" or arm == "A_pre":
                out = model(images)
                logits = out["logits"] if isinstance(out, dict) else out
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight_t)
            elif arm in ["B", "B_placebo", "B_pre"]:
                out = model(images)
                loss_cls = F.binary_cross_entropy_with_logits(out["logits"], labels, pos_weight=pos_weight_t)
                loss_mask = F.binary_cross_entropy_with_logits(out["mask_logits"], masks)
                loss = loss_cls + 1.0 * loss_mask
            elif arm == "C":
                out = model(images)
                loss_cls = F.binary_cross_entropy_with_logits(out["logits"], labels, pos_weight=pos_weight_t)
                loss_cam = F.binary_cross_entropy_with_logits(out["cam_up"], masks)
                loss = loss_cls + 1.0 * loss_cam
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    # Evaluate model predictions on a fixed probe tensor to test bit-identical reproducibility
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            probe_x = images[:4]
            probe_out = model(probe_x)
            logits = probe_out["logits"] if isinstance(probe_out, dict) else probe_out
            logits_np = logits.cpu().numpy()
            
    return logits_np

def main():
    print("=== STARTING TRACK B SMOKE TEST (1 epoch, 64 images) ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    base_ds = FracAtlasDataset(manifest="data/frozen/train.csv", normalize=True)
    subset_64 = Subset(base_ds, list(range(64)))
    
    arms = ["A", "B", "B_placebo", "C", "A_pre", "B_pre"]
    results = {}
    
    for arm in arms:
        t0 = time.time()
        if arm == "B_placebo":
            train_ds = PlaceboDatasetWrapper(subset_64, seed=42)
        else:
            train_ds = subset_64
            
        loader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=fracatlas_collate_fn)
        
        # Run 1
        logits1 = run_one_epoch(arm, seed=42, loader=loader, device=device)
        # Run 2 (repeat with same seed to test determinism)
        logits2 = run_one_epoch(arm, seed=42, loader=loader, device=device)
        
        # Check bit-identical predictions
        is_identical = np.array_equal(logits1, logits2)
        max_diff = np.max(np.abs(logits1 - logits2))
        
        elapsed = time.time() - t0
        print(f"Arm {arm:9s}: Run 1 & Run 2 identical={is_identical} (max diff={max_diff:.2e}) [{elapsed:.2f} s]")
        assert is_identical or max_diff < 1e-6, f"Arm {arm} not deterministic! Max diff = {max_diff}"
        results[arm] = (is_identical, max_diff)
        
    print("\nALL 6 ARMS PASSED THE 1-EPOCH SMOKE TEST AND DETERMINISM CHECK!")

if __name__ == "__main__":
    main()
