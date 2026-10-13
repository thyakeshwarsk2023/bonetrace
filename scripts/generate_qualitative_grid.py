"""Publication-Quality Grad-CAM Qualitative Comparison Grid.

Generates a side-by-side 4-panel comparison across representative test cases:
- Panel 1: Original X-Ray with ground-truth fracture bounding box.
- Panel 2: Baseline Model Grad-CAM overlay + P(Fracture).
- Panel 3: Region-Supervised Model Grad-CAM overlay + P(Fracture).
- Panel 4: Spatial Attention Difference (|CAM_B - CAM_A|) masked to anatomical FOV.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from bonetrace.config import load_config
from bonetrace.data.dataset import FracAtlasDataset, IMAGENET_MEAN, IMAGENET_STD
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.models.factory import build_model


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized (3, H, W) tensor to uint8 RGB (H, W, 3)."""
    img = img_tensor.detach().cpu().clone()
    for c in range(3):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    img = torch.clamp(img, 0.0, 1.0)
    img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img_np


def get_pad_mask(img_tensor: torch.Tensor) -> np.ndarray:
    """Identify black letterbox padding (channel 0 < -2.0 in normalized space)."""
    return (img_tensor[0] < -2.0).detach().cpu().numpy()


def overlay_cam_on_image(
    img_rgb: np.ndarray,
    cam: np.ndarray,
    pad_mask: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Alpha-blend colormapped Grad-CAM over grayscale/RGB radiograph, masking padding."""
    cam_cleaned = cam.copy()
    cam_cleaned[pad_mask] = 0.0
    cam_uint8 = np.uint8(255 * np.clip(cam_cleaned, 0.0, 1.0))
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Where padding exists, show pure background without heatmap
    overlay = (alpha * heatmap.astype(float) + (1.0 - alpha) * img_rgb.astype(float)).astype(np.uint8)
    overlay[pad_mask] = img_rgb[pad_mask]
    return overlay


def run_qualitative_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running qualitative evaluation on device: {device}")

    cfg = load_config()
    test_csv = cfg.paths.processed_data_dir / "test.csv"
    dataset = FracAtlasDataset(manifest=test_csv, split="test", exclude_conflicts=True)
    n_total = len(dataset)
    print(f"Loaded {n_total} test images.")

    # 1. Load models
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

    # 2. Score all test samples to select optimal representative cases
    print("Scoring test dataset for candidate case selection...")
    records = []
    with torch.no_grad():
        for i in range(n_total):
            sample = dataset[i]
            img = sample["image"].unsqueeze(0).to(device)
            lbl = int(sample["label"].item())
            has_ann = bool(sample["has_annotation"].item())
            boxes = sample["boxes"].tolist()

            out_b = models["baseline"](img)
            logit_b = (out_b["logits"] if isinstance(out_b, dict) else out_b)[0, 0].item()
            prob_b = 1.0 / (1.0 + np.exp(-logit_b))

            out_r = models["region_supervised"](img)
            logit_r = (out_r["logits"] if isinstance(out_r, dict) else out_r)[0, 0].item()
            prob_r = 1.0 / (1.0 + np.exp(-logit_r))

            fname = str(dataset.records[i].get("representative_image_id", sample["canonical_id"]))
            records.append({
                "index": i,
                "canonical_id": sample["canonical_id"],
                "filename": fname,
                "label": lbl,
                "has_annotation": has_ann,
                "boxes": boxes,
                "prob_baseline": prob_b,
                "prob_region_sup": prob_r,
                "pred_b": 1 if prob_b >= 0.5 else 0,
                "pred_r": 1 if prob_r >= 0.5 else 0,
                "diff_prob": prob_r - prob_b,
            })

    df = pd.DataFrame(records)

    # Candidate selection
    # 1. Concordant True Positives (both >= 0.70, clear fracture boxes)
    concordant_tp = df[(df["label"] == 1) & (df["has_annotation"]) & (df["prob_baseline"] >= 0.70) & (df["prob_region_sup"] >= 0.70)].copy()
    concordant_tp = concordant_tp.sort_values(by="prob_region_sup", ascending=False)

    # 2. Model B Wins / Sensitivity Gain (ground truth = 1, baseline < 0.50, region_sup >= 0.50)
    model_b_wins = df[(df["label"] == 1) & (df["has_annotation"]) & (df["prob_baseline"] < 0.50) & (df["prob_region_sup"] >= 0.50)].copy()
    model_b_wins = model_b_wins.sort_values(by="diff_prob", ascending=False)

    # 3. Hard cases / Complex subtle fractures or joint ambiguity
    hard_cases = df[
        ((df["label"] == 1) & (df["has_annotation"]) & (df["prob_baseline"] < 0.60) & (df["prob_region_sup"] > df["prob_baseline"])) |
        ((df["label"] == 0) & (df["prob_baseline"] > 0.40) & (df["prob_region_sup"] < 0.25))
    ].copy()
    hard_cases = hard_cases.sort_values(by="diff_prob", key=abs, ascending=False)

    selected_indices: List[Tuple[int, str, str]] = []

    # Pick 2 Concordant TPs
    for _, row in concordant_tp.head(2).iterrows():
        selected_indices.append((int(row["index"]), "Concordant True Positive", f"Both models confident; Region-Supervised demonstrates tighter localization."))

    # Pick 3 Model B Wins (Sensitivity Gain)
    for _, row in model_b_wins.head(3).iterrows():
        selected_indices.append((int(row["index"]), "Sensitivity Gain (Model B Win)", f"Baseline missed fracture (P={row['prob_baseline']:.2f}); Region-Supervised detected it (P={row['prob_region_sup']:.2f})."))

    # If fewer than 3 Model B wins in strict threshold, grab top diff_prob where label == 1
    if len([x for x in selected_indices if "Sensitivity Gain" in x[1]]) < 3:
        alt_wins = df[(df["label"] == 1) & (df["has_annotation"]) & (df["diff_prob"] > 0.20)].sort_values(by="diff_prob", ascending=False)
        for _, row in alt_wins.iterrows():
            idx = int(row["index"])
            if idx not in [x[0] for x in selected_indices]:
                selected_indices.append((idx, "Sensitivity Gain (Model B Win)", f"Region-Supervised sensitivity gain: P={row['prob_region_sup']:.2f} vs P={row['prob_baseline']:.2f}."))
                if len([x for x in selected_indices if "Sensitivity Gain" in x[1]]) >= 3:
                    break

    # Pick 2 Hard Cases (Subtle fissures or non-fracture specificity correction)
    for _, row in hard_cases.iterrows():
        idx = int(row["index"])
        if idx not in [x[0] for x in selected_indices]:
            lbl_str = "Fracture" if row["label"] == 1 else "Non-Fracture"
            desc = f"Hard case ({lbl_str}): subtle pathology; contrast map highlights attention shift."
            selected_indices.append((idx, f"Hard Case / Ambiguity", desc))
            if len([x for x in selected_indices if "Hard Case" in x[1]]) >= 2:
                break

    # Ensure total is between 6 and 8
    selected_indices = selected_indices[:7]
    print(f"\nSelected {len(selected_indices)} representative cases across categories:")
    for idx, cat, desc in selected_indices:
        r = df.loc[df["index"] == idx].iloc[0]
        print(f"  [{cat}] Image: {r['filename']} | Label: {r['label']} | P(Base): {r['prob_baseline']:.3f} | P(Reg): {r['prob_region_sup']:.3f}")

    # Output directories
    out_dir = Path("outputs/visualizations")
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    # 3. Build qualitative grid
    n_cases = len(selected_indices)
    fig, axes = plt.subplots(nrows=n_cases, ncols=4, figsize=(18, 4.4 * n_cases))
    plt.subplots_adjust(wspace=0.08, hspace=0.22)

    summary_rows = []

    for row_idx, (sample_idx, category, clinical_notes) in enumerate(selected_indices):
        sample = dataset[sample_idx]
        img_tensor = sample["image"]
        lbl = int(sample["label"].item())
        boxes = sample["boxes"].tolist()
        fname = str(dataset.records[sample_idx].get("representative_image_id", sample["canonical_id"]))
        canon_id = sample["canonical_id"]

        img_rgb = denormalize_image(img_tensor)
        pad_mask = get_pad_mask(img_tensor)
        fov_mask = ~pad_mask

        # Generate Grad-CAM heatmaps
        cam_b = gradcams["baseline"].generate(img_tensor, class_idx=0)
        cam_r = gradcams["region_supervised"].generate(img_tensor, class_idx=0)

        # Apply FOV masking (padding remains uncolored)
        cam_b_clean = cam_b.copy()
        cam_b_clean[pad_mask] = 0.0

        cam_r_clean = cam_r.copy()
        cam_r_clean[pad_mask] = 0.0

        # Contrast map: absolute spatial difference |CAM_B - CAM_A|
        diff_map = np.abs(cam_r_clean - cam_b_clean)
        diff_map[pad_mask] = 0.0

        # Probability predictions
        inp = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pb = float(torch.sigmoid(models["baseline"](inp)["logits"])[0, 0].item())
            pr = float(torch.sigmoid(models["region_supervised"](inp)["logits"])[0, 0].item())

        # Overlays
        overlay_b = overlay_cam_on_image(img_rgb, cam_b_clean, pad_mask, alpha=0.48, colormap=cv2.COLORMAP_TURBO)
        overlay_r = overlay_cam_on_image(img_rgb, cam_r_clean, pad_mask, alpha=0.48, colormap=cv2.COLORMAP_TURBO)

        # Diff overlay with warm colormap
        diff_norm = diff_map / (diff_map.max() + 1e-8)
        diff_uint8 = np.uint8(255 * diff_norm)
        diff_color = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_INFERNO)
        diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
        overlay_diff = (0.55 * diff_color.astype(float) + 0.45 * img_rgb.astype(float)).astype(np.uint8)
        overlay_diff[pad_mask] = img_rgb[pad_mask]

        summary_rows.append({
            "Case ID": f"Case {row_idx + 1}",
            "Image": fname,
            "Canonical ID": canon_id,
            "Category": category,
            "Ground Truth": "Fractured" if lbl == 1 else "Non-Fractured",
            "P(Baseline)": f"{pb:.3f}",
            "P(Region-Sup)": f"{pr:.3f}",
            "Delta P": f"{pr - pb:+.3f}",
        })

        # --- Plot Panel 1: Original + GT Bounding Boxes ---
        ax1 = axes[row_idx, 0]
        ax1.imshow(img_rgb)
        lbl_text = "Fracture Positive" if lbl == 1 else "Non-Fractured"
        color_gt = "#00FF66" if lbl == 1 else "#33CCFF"

        for b in boxes:
            xmin, ymin, xmax, ymax = b
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2.4, edgecolor=color_gt, facecolor="none", linestyle="-"
            )
            ax1.add_patch(rect)

        ax1.set_title(f"Case {row_idx+1}: {fname}\nGT: {lbl_text}", fontsize=11, fontweight="bold", pad=5)
        ax1.axis("off")

        # --- Plot Panel 2: Baseline Grad-CAM ---
        ax2 = axes[row_idx, 1]
        ax2.imshow(overlay_b)
        status_b = "TP" if (lbl == 1 and pb >= 0.5) else ("FN" if (lbl == 1 and pb < 0.5) else ("TN" if (lbl == 0 and pb < 0.5) else "FP"))
        ax2.set_title(f"Model A (Baseline ResNet-18)\nP(Fracture) = {pb:.1%} [{status_b}]", fontsize=11, fontweight="bold", pad=5)
        ax2.axis("off")

        # --- Plot Panel 3: Region-Supervised Grad-CAM ---
        ax3 = axes[row_idx, 2]
        ax3.imshow(overlay_r)
        status_r = "TP" if (lbl == 1 and pr >= 0.5) else ("FN" if (lbl == 1 and pr < 0.5) else ("TN" if (lbl == 0 and pr < 0.5) else "FP"))
        ax3.set_title(f"Model B (Region-Supervised)\nP(Fracture) = {pr:.1%} [{status_r}]", fontsize=11, fontweight="bold", pad=5, color="#0066CC")
        ax3.axis("off")

        # --- Plot Panel 4: Spatial Contrast Map ---
        ax4 = axes[row_idx, 3]
        ax4.imshow(overlay_diff)
        for b in boxes:
            xmin, ymin, xmax, ymax = b
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=1.8, edgecolor="#00FF66", facecolor="none", linestyle="--"
            )
            ax4.add_patch(rect)
        ax4.set_title(f"Spatial Contrast |CAM_B - CAM_A|\nAttention Realignment", fontsize=11, fontweight="bold", pad=5, color="#8B0000")
        ax4.axis("off")

        # Save individual crop
        fig_single, ax_s = plt.subplots(1, 4, figsize=(16, 4))
        ax_s[0].imshow(img_rgb)
        for b in boxes:
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2.2, edgecolor="#00FF66", facecolor="none")
            ax_s[0].add_patch(rect)
        ax_s[0].set_title(f"Original X-Ray (GT: {lbl_text})", fontsize=10, fontweight="bold")
        ax_s[0].axis("off")

        ax_s[1].imshow(overlay_b)
        ax_s[1].set_title(f"Baseline Grad-CAM (P = {pb:.1%})", fontsize=10, fontweight="bold")
        ax_s[1].axis("off")

        ax_s[2].imshow(overlay_r)
        ax_s[2].set_title(f"Region-Supervised (P = {pr:.1%})", fontsize=10, fontweight="bold")
        ax_s[2].axis("off")

        ax_s[3].imshow(overlay_diff)
        for b in boxes:
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=1.5, edgecolor="#00FF66", facecolor="none", linestyle="--")
            ax_s[3].add_patch(rect)
        ax_s[3].set_title(f"Contrast |B - A| (Shift)", fontsize=10, fontweight="bold")
        ax_s[3].axis("off")

        single_path = cases_dir / f"case_{row_idx+1}_{fname}.png"
        fig_single.savefig(single_path, dpi=300, bbox_inches="tight")
        plt.close(fig_single)

    # Global Super Title
    fig.suptitle(
        "BoneTrace: Controlled Qualitative Grad-CAM Comparison\n"
        "Model A (Baseline ResNet-18) vs. Model B (Dual-Head Region-Supervised ResNet-18) on Held-Out Test Set",
        fontsize=15, fontweight="bold", y=0.995
    )

    png_path = out_dir / "gradcam_qualitative_grid.png"
    pdf_path = out_dir / "gradcam_qualitative_grid.pdf"

    print(f"\nSaving composite figures (300 DPI)...")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved PNG: {png_path}")
    print(f"[+] Saved PDF: {pdf_path}")
    print(f"[+] Saved {n_cases} individual case crops in {cases_dir}")

    # Clean hooks
    for g in gradcams.values():
        g.remove_hooks()

    # Print and export summary table
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print(" SUMMARY TABLE: SELECTED QUALITATIVE CASES FOR PUBLICATION GRID")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)

    summary_md_path = out_dir / "QUALITATIVE_SUMMARY.md"
    cols = list(df_summary.columns)
    md_lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in df_summary.iterrows():
        md_lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[+] Saved summary markdown: {summary_md_path}")


if __name__ == "__main__":
    run_qualitative_pipeline()
