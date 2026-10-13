"""Comprehensive methodological audit and sanity check of Grad-CAM perturbation faithfulness.

Audits:
1. Masking implementation (ImageNet mean replacement in normalized space).
2. Control mask sampling and letterbox padding bias.
3. Full test cohort (N=610) vs Fracture-Positive (N=100) vs Non-Fracture (N=510).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from bonetrace.config import load_config
from bonetrace.data.dataset import FracAtlasDataset
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.models.factory import build_model


def get_fov_mask(image_tensor: torch.Tensor) -> np.ndarray:
    """Identify valid non-padding image field-of-view (FOV).
    
    Letterbox padding is black (0, 0, 0), which under ImageNet normalization
    evaluates to approximately -2.11 in channel 0.
    """
    # Channel 0 threshold: >= -2.0 identifies non-padding active frame
    return (image_tensor[0] >= -2.0).cpu().numpy()


def compute_faithfulness_single(
    model: nn.Module,
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    top_k_ratio: float = 0.10,
    constrain_to_fov: bool = False,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, float]:
    """Compute attribution-guided logit drop for attributed and control masks."""
    model.eval()
    inp = image_tensor.unsqueeze(0).to(device)

    # 1. Baseline logit
    with torch.no_grad():
        out_orig = model(inp)
        logits_orig = out_orig["logits"] if isinstance(out_orig, dict) else out_orig
        orig_logit = float(logits_orig[0, 0].item())

    # 2. Extract Grad-CAM
    cam = gradcam.generate(image_tensor, class_idx=0)
    flat_cam = cam.flatten()

    pad_mask = ~get_fov_mask(image_tensor)

    if constrain_to_fov:
        fov_mask = ~pad_mask
        # Mask out padding from CAM consideration for control sampling
        cam_valid = cam.copy()
        # Set padding regions to +infinity so they are never picked for lowest attribution
        cam_for_ctrl = cam.copy()
        cam_for_ctrl[pad_mask] = np.inf
        # Set padding regions to -infinity so they are never picked for highest attribution
        cam_for_attr = cam.copy()
        cam_for_attr[pad_mask] = -np.inf

        n_valid = int(fov_mask.sum())
        n_mask = max(1, int(round(n_valid * top_k_ratio)))

        attr_flat_idx = np.argsort(-cam_for_attr.flatten())[:n_mask]
        ctrl_flat_idx = np.argsort(cam_for_ctrl.flatten())[:n_mask]

        attr_mask = np.zeros_like(cam, dtype=bool)
        attr_mask[np.unravel_index(attr_flat_idx, cam.shape)] = True

        ctrl_mask = np.zeros_like(cam, dtype=bool)
        ctrl_mask[np.unravel_index(ctrl_flat_idx, cam.shape)] = True
    else:
        n_total = len(flat_cam)
        n_mask = max(1, int(round(n_total * top_k_ratio)))

        attr_flat_idx = np.argsort(-flat_cam)[:n_mask]
        ctrl_flat_idx = np.argsort(flat_cam)[:n_mask]

        attr_mask = np.zeros_like(cam, dtype=bool)
        attr_mask[np.unravel_index(attr_flat_idx, cam.shape)] = True

        ctrl_mask = np.zeros_like(cam, dtype=bool)
        ctrl_mask[np.unravel_index(ctrl_flat_idx, cam.shape)] = True

    # Audit padding overlap
    ctrl_pad_overlap = float(np.sum(ctrl_mask & pad_mask) / n_mask)
    attr_pad_overlap = float(np.sum(attr_mask & pad_mask) / n_mask)

    # Mask attributed region (replace with 0.0, ImageNet mean in normalized space)
    img_attr = image_tensor.clone()
    img_attr[:, torch.from_numpy(attr_mask).to(img_attr.device)] = 0.0

    with torch.no_grad():
        out_attr = model(img_attr.unsqueeze(0).to(device))
        logits_attr = out_attr["logits"] if isinstance(out_attr, dict) else out_attr
        attr_logit = float(logits_attr[0, 0].item())

    # Mask control region
    img_ctrl = image_tensor.clone()
    img_ctrl[:, torch.from_numpy(ctrl_mask).to(img_ctrl.device)] = 0.0

    with torch.no_grad():
        out_ctrl = model(img_ctrl.unsqueeze(0).to(device))
        logits_ctrl = out_ctrl["logits"] if isinstance(out_ctrl, dict) else out_ctrl
        ctrl_logit = float(logits_ctrl[0, 0].item())

    delta_attr = orig_logit - attr_logit
    delta_ctrl = orig_logit - ctrl_logit
    advantage = delta_attr - delta_ctrl

    return {
        "orig_logit": orig_logit,
        "delta_attr": delta_attr,
        "delta_ctrl": delta_ctrl,
        "advantage": advantage,
        "is_faithful": 1.0 if advantage > 0.0 else 0.0,
        "ctrl_pad_overlap": ctrl_pad_overlap,
        "attr_pad_overlap": attr_pad_overlap,
    }


def run_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing audit on: {device}")

    cfg = load_config()
    test_csv = cfg.paths.processed_data_dir / "test.csv"
    dataset = FracAtlasDataset(manifest=test_csv, split="test", exclude_conflicts=True)
    n_total = len(dataset)
    print(f"Test dataset loaded: {n_total} samples (excluding disputed clones)")

    # Identify cohorts
    labels = [int(dataset[i]["label"].item()) for i in range(n_total)]
    pos_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
    neg_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
    annotated_indices = [i for i in range(n_total) if dataset[i]["has_annotation"].item()][:50]

    print(f"Cohorts: Full Test={len(labels)}, Fracture-Positive={len(pos_indices)}, Non-Fracture={len(neg_indices)}")

    models = {}
    gradcams = {}
    for mtype in ["baseline", "region_supervised"]:
        m = build_model(config=cfg, model_type=mtype)
        ckpt_path = Path(cfg.paths.checkpoints_dir) / mtype / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device)
        m.eval()
        models[mtype] = m
        gradcams[mtype] = GradCAM(m)

    cohorts = {
        "Full Test Cohort (N=610)": list(range(n_total)),
        "Fracture-Positive Subset (N=100)": pos_indices,
        "Non-Fracture Control Subset (N=510)": neg_indices,
        "Original Reported Annotated (N=50)": annotated_indices,
    }

    results = {}

    for mode_name, constrain_fov in [("Unconstrained (Original)", False), ("FOV-Constrained (Patched)", True)]:
        results[mode_name] = {}
        for cohort_name, indices in cohorts.items():
            results[mode_name][cohort_name] = {}
            for mtype in ["baseline", "region_supervised"]:
                m = models[mtype]
                g = gradcams[mtype]

                records = []
                for idx in indices:
                    sample = dataset[idx]
                    res = compute_faithfulness_single(
                        model=m,
                        gradcam=g,
                        image_tensor=sample["image"],
                        top_k_ratio=0.10,
                        constrain_to_fov=constrain_fov,
                        device=device,
                    )
                    records.append(res)

                df = pd.DataFrame(records)
                results[mode_name][cohort_name][mtype] = {
                    "delta_attr_mean": float(df["delta_attr"].mean()),
                    "delta_attr_std": float(df["delta_attr"].std()),
                    "delta_ctrl_mean": float(df["delta_ctrl"].mean()),
                    "delta_ctrl_std": float(df["delta_ctrl"].std()),
                    "advantage_mean": float(df["advantage"].mean()),
                    "advantage_std": float(df["advantage"].std()),
                    "success_rate": float(df["is_faithful"].mean()),
                    "ctrl_pad_overlap_mean": float(df["ctrl_pad_overlap"].mean()),
                }

    # Save summary
    out_path = Path(cfg.paths.output_dir) / "evaluation" / "faithfulness_audit_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Audit evaluation completed successfully.")
    print("\n--- RESULTS SUMMARY ---")
    for mode_name, cohorts_res in results.items():
        print(f"\n=================== {mode_name} ===================")
        for cname, m_res in cohorts_res.items():
            print(f"\nCohort: {cname}")
            base = m_res["baseline"]
            reg = m_res["region_supervised"]
            print(f"  [Baseline]          Delta_Attr: {base['delta_attr_mean']:.4f} | Delta_Ctrl: {base['delta_ctrl_mean']:.4f} | Advantage: {base['advantage_mean']:.4f} | Pad_Overlap: {base['ctrl_pad_overlap_mean']*100:.1f}%")
            print(f"  [Region-Supervised] Delta_Attr: {reg['delta_attr_mean']:.4f} | Delta_Ctrl: {reg['delta_ctrl_mean']:.4f} | Advantage: {reg['advantage_mean']:.4f} | Pad_Overlap: {reg['ctrl_pad_overlap_mean']*100:.1f}%")


if __name__ == "__main__":
    run_audit()
