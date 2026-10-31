"""
scripts/generate_extended_visuals.py

Generates extended publication-grade visual Grad-CAM and diagnostic figures:
1. fig_clinical_failure_modes_and_limitations.png:
   Rigorous failure analysis showing false negatives (missed fractures) on subtle non-displaced fractures.
2. fig_anatomical_diversity_spotlights.png:
   Cross-anatomical evaluation spanning Hip/Pelvis, Shoulder/Clavicle, Hardware Implants, and Hand/Wrist.
3. fig_hirescam_vs_gradcam_comparison.png:
   Methodological comparison between Standard Grad-CAM and HiResCAM (element-wise gradient weighting).
4. case_spotlight_CANON_0044_hip.png:
   High-resolution 5-panel spotlight on complex pelvic/hip fracture triage.
5. case_spotlight_CANON_0647_shoulder.png:
   High-resolution 5-panel spotlight on clavicle/shoulder fracture.
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 console output and instant flushing on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bonetrace.ui.helpers import (
    load_model_for_ui,
    load_and_preprocess,
    get_fov_mask,
    generate_attribution_map,
    generate_aux_head_map,
    overlay_heatmap,
    draw_boxes_on_image,
    draw_fov_contour,
    add_peak_marker,
    parse_boxes,
    run_inference,
)
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.explainability.multiscale_cam import MultiScaleAttributionEngine

VIS_DIR = REPO_ROOT / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def style_figure(fig, axes):
    """Apply dark/neutral publication theme to matplotlib figure and axes."""
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat if hasattr(axes, "flat") else [axes]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
            spine.set_linewidth(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FAILURE MODES & CLINICAL LIMITATIONS FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_failure_modes_figure(df: pd.DataFrame, models: dict):
    print("Generating Figure: Clinical Failure Modes & Limitations...", flush=True)
    cases = [
        ("CANON_1018", "Hand Phalanx: Subtle Non-Displaced Hairline (Concordant Miss)"),
        ("CANON_0064", "Tibia/Fibula: Non-Displaced Cortical Fissure (Concordant Miss)"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9), dpi=300)
    style_figure(fig, axes)

    for row_idx, (cid, desc) in enumerate(cases):
        sub = df[df["canonical_id"] == cid]
        row = sub.iloc[0]
        img_p = Path(row["resolved_primary_path"])
        boxes = parse_boxes(row.get("letterbox_target_boxes"))

        disp_rgb, img_t = load_and_preprocess(img_p)
        fov_mask = get_fov_mask(img_t)

        m_a = models["resnet18_baseline"]
        m_b = models["resnet18_region"]
        m_dense_b = models["densenet121_region"]

        p_a, logit_a = run_inference(m_a, img_t, DEVICE)
        p_b, logit_b = run_inference(m_b, img_t, DEVICE)

        cam_a = generate_attribution_map(m_a, img_t, "final", fov_mask, DEVICE)
        cam_b = generate_attribution_map(m_b, img_t, "final", fov_mask, DEVICE)
        aux_b = generate_aux_head_map(m_b, img_t, fov_mask, DEVICE)
        aux_dense = generate_aux_head_map(m_dense_b, img_t, fov_mask, DEVICE)

        img1 = draw_fov_contour(draw_boxes_on_image(disp_rgb, boxes, color=(0, 255, 0), thickness=2), fov_mask)
        img2 = add_peak_marker(overlay_heatmap(disp_rgb, cam_a, alpha=0.45, fov_mask=fov_mask), cam_a, fov_mask)
        img3 = add_peak_marker(overlay_heatmap(disp_rgb, cam_b, alpha=0.45, fov_mask=fov_mask), cam_b, fov_mask)
        img4 = overlay_heatmap(disp_rgb, aux_b, alpha=0.45, colormap="magma", fov_mask=fov_mask)
        img5 = overlay_heatmap(disp_rgb, aux_dense, alpha=0.45, colormap="magma", fov_mask=fov_mask)

        axes[row_idx, 0].imshow(img1)
        axes[row_idx, 0].set_title(f"Case {cid} (GT: Fracture, y=1)\n{desc}", color="#f0f6fc", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 0].set_xlabel("Green: Subtle Fracture Bounding Box", color="#8b949e", fontsize=8)

        axes[row_idx, 1].imshow(img2)
        axes[row_idx, 1].set_title(f"Model A (Baseline): MISSED\nProb: {p_a:.1%} (Logit: {logit_a:+.2f})", color="#ff7b72", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 1].set_xlabel("Diffuse off-target attention", color="#ff7b72", fontsize=8)

        axes[row_idx, 2].imshow(img3)
        axes[row_idx, 2].set_title(f"Model B (BoneTrace): MISSED\nProb: {p_b:.1%} (Logit: {logit_b:+.2f})", color="#ff7b72", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 2].set_xlabel("Suppressed low-contrast crack", color="#ff7b72", fontsize=8)

        axes[row_idx, 3].imshow(img4)
        axes[row_idx, 3].set_title(f"ResNet-18 σ(M_aux)\nPeak Prob: {float(aux_b.max()):.1%}", color="#d29922", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 3].set_xlabel("Aux head sub-threshold output", color="#8b949e", fontsize=8)

        axes[row_idx, 4].imshow(img5)
        axes[row_idx, 4].set_title(f"DenseNet-121 σ(M_aux)\nPeak Prob: {float(aux_dense.max()):.1%}", color="#58a6ff", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 4].set_xlabel("Dense cross-backbone miss", color="#8b949e", fontsize=8)

        for col_idx in range(5):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    plt.tight_layout()
    out_p = VIS_DIR / "fig_clinical_failure_modes_and_limitations.png"
    plt.savefig(out_p, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_p}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-ANATOMICAL DIVERSITY SPOTLIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_anatomical_diversity_figure(df: pd.DataFrame, models: dict):
    print("Generating Figure: Anatomical Diversity Across Body Regions...", flush=True)
    cases = [
        ("CANON_0044", "Pelvis / Hip Fracture", "Hip"),
        ("CANON_0647", "Clavicle / Shoulder Fracture", "Shoulder"),
        ("CANON_0766", "Hardware / Metal Plate with Fracture", "Leg Hardware"),
        ("CANON_2386", "Hand / Metacarpal Displaced Fracture", "Hand"),
    ]

    fig, axes = plt.subplots(4, 5, figsize=(22, 17), dpi=300)
    style_figure(fig, axes)

    col_headers = [
        "1. Input Radiograph & GT",
        "2. Model A (Baseline Grad-CAM)",
        "3. Model B (BoneTrace Grad-CAM)",
        "4. ResNet-18 Aux Head σ(M_aux)",
        "5. DenseNet-121 Aux Head σ(M_aux)",
    ]

    for row_idx, (cid, desc, region_tag) in enumerate(cases):
        sub = df[df["canonical_id"] == cid]
        row = sub.iloc[0]
        img_p = Path(row["resolved_primary_path"])
        boxes = parse_boxes(row.get("letterbox_target_boxes"))

        disp_rgb, img_t = load_and_preprocess(img_p)
        fov_mask = get_fov_mask(img_t)

        m_a = models["resnet18_baseline"]
        m_b = models["resnet18_region"]
        m_dense_b = models["densenet121_region"]

        p_a, _ = run_inference(m_a, img_t, DEVICE)
        p_b, _ = run_inference(m_b, img_t, DEVICE)

        cam_a = generate_attribution_map(m_a, img_t, "final", fov_mask, DEVICE)
        cam_b = generate_attribution_map(m_b, img_t, "final", fov_mask, DEVICE)
        aux_b = generate_aux_head_map(m_b, img_t, fov_mask, DEVICE)
        aux_dense = generate_aux_head_map(m_dense_b, img_t, fov_mask, DEVICE)

        img1 = draw_fov_contour(draw_boxes_on_image(disp_rgb, boxes, color=(0, 255, 0), thickness=2), fov_mask)
        img2 = add_peak_marker(overlay_heatmap(disp_rgb, cam_a, alpha=0.45, fov_mask=fov_mask), cam_a, fov_mask)
        img3 = add_peak_marker(overlay_heatmap(disp_rgb, cam_b, alpha=0.45, fov_mask=fov_mask), cam_b, fov_mask)
        img4 = overlay_heatmap(disp_rgb, aux_b, alpha=0.45, colormap="magma", fov_mask=fov_mask)
        img5 = overlay_heatmap(disp_rgb, aux_dense, alpha=0.45, colormap="magma", fov_mask=fov_mask)

        row_imgs = [img1, img2, img3, img4, img5]
        row_subs = [
            f"Region: {region_tag} ({cid})",
            f"P(Frac): {p_a:.1%}",
            f"P(Frac): {p_b:.1%} (ΔP: {p_b - p_a:+.1%})",
            f"Peak: {float(aux_b.max()):.1%}",
            f"Peak: {float(aux_dense.max()):.1%}",
        ]

        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            ax.imshow(row_imgs[col_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], color="#f0f6fc", fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(row_subs[col_idx], color="#8b949e", fontsize=9, labelpad=4)

    plt.tight_layout()
    out_p = VIS_DIR / "fig_anatomical_diversity_spotlights.png"
    plt.savefig(out_p, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_p}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HIRESCAM VS STANDARD GRAD-CAM COMPARISON FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hirescam_comparison(df: pd.DataFrame, models: dict):
    print("Generating Figure: HiResCAM vs Standard Grad-CAM Comparison...", flush=True)
    cases = [
        ("CANON_0350", "Hairline Leg Fracture"),
        ("CANON_1617", "Subtle Hand Phalanx Fracture"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9), dpi=300)
    style_figure(fig, axes)

    for row_idx, (cid, desc) in enumerate(cases):
        sub = df[df["canonical_id"] == cid]
        row = sub.iloc[0]
        img_p = Path(row["resolved_primary_path"])
        boxes = parse_boxes(row.get("letterbox_target_boxes"))

        disp_rgb, img_t = load_and_preprocess(img_p)
        fov_mask = get_fov_mask(img_t)

        m_b = models["resnet18_region"]
        engine = MultiScaleAttributionEngine(model=m_b, device=DEVICE)
        img_batch = img_t.unsqueeze(0).to(DEVICE) if img_t.ndim == 3 else img_t.to(DEVICE)
        attrs = engine.compute_all_attributions(img_batch)

        cam_std = attrs["gradcam_layer4"].squeeze().cpu().numpy() * fov_mask
        cam_hi = attrs["hirescam_layer4"].squeeze().cpu().numpy() * fov_mask
        cam_l3 = attrs["gradcam_layer3"].squeeze().cpu().numpy() * fov_mask

        # Aux Head
        aux_map = generate_aux_head_map(m_b, img_t, fov_mask, DEVICE)

        img1 = draw_fov_contour(draw_boxes_on_image(disp_rgb, boxes, color=(0, 255, 0), thickness=2), fov_mask)
        img2 = add_peak_marker(overlay_heatmap(disp_rgb, cam_std, alpha=0.45, fov_mask=fov_mask), cam_std, fov_mask)
        img3 = add_peak_marker(overlay_heatmap(disp_rgb, cam_hi, alpha=0.45, fov_mask=fov_mask), cam_hi, fov_mask)
        img4 = add_peak_marker(overlay_heatmap(disp_rgb, cam_l3, alpha=0.45, fov_mask=fov_mask), cam_l3, fov_mask)
        img5 = overlay_heatmap(disp_rgb, aux_map, alpha=0.45, colormap="magma", fov_mask=fov_mask)

        axes[row_idx, 0].imshow(img1)
        axes[row_idx, 0].set_title(f"Input: {cid}\n{desc}", color="#f0f6fc", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 0].set_xlabel("Green: Ground Truth Lesion Box", color="#8b949e", fontsize=8)

        axes[row_idx, 1].imshow(img2)
        axes[row_idx, 1].set_title("Standard Grad-CAM (layer4)\nSpatial Average Gradient Weighting", color="#ff7b72", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 1].set_xlabel("GAP causes diffuse activation blob", color="#ff7b72", fontsize=8)

        axes[row_idx, 2].imshow(img3)
        axes[row_idx, 2].set_title("HiResCAM (layer4)\nElement-Wise Gradient Weighting", color="#58a6ff", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 2].set_xlabel("Preserves intra-feature high frequency", color="#58a6ff", fontsize=8)

        axes[row_idx, 3].imshow(img4)
        axes[row_idx, 3].set_title("Stage 3 Grad-CAM (layer3)\nStride 16 (32×32 Feature Grid)", color="#d29922", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 3].set_xlabel("Finer spatial resolution before GAP", color="#8b949e", fontsize=8)

        axes[row_idx, 4].imshow(img5)
        axes[row_idx, 4].set_title("Direct Auxiliary Head σ(M_aux)\nUnpooled 512×512 Dense Projection", color="#56d364", fontsize=9, fontweight="bold", pad=6)
        axes[row_idx, 4].set_xlabel("Direct supervised spatial localization", color="#56d364", fontsize=8)

        for col_idx in range(5):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    plt.tight_layout()
    out_p = VIS_DIR / "fig_hirescam_vs_gradcam_comparison.png"
    plt.savefig(out_p, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_p}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TWO ADDITIONAL HIGH-RESOLUTION ANATOMICAL SPOTLIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_case_spotlight(df: pd.DataFrame, models: dict, cid: str, title_desc: str, filename: str):
    sub = df[df["canonical_id"] == cid]
    row = sub.iloc[0]
    img_p = Path(row["resolved_primary_path"])
    boxes = parse_boxes(row.get("letterbox_target_boxes"))

    disp_rgb, img_t = load_and_preprocess(img_p)
    fov_mask = get_fov_mask(img_t)

    m_a = models["resnet18_baseline"]
    m_b = models["resnet18_region"]
    m_dense_b = models["densenet121_region"]

    p_a, logit_a = run_inference(m_a, img_t, DEVICE)
    p_b, logit_b = run_inference(m_b, img_t, DEVICE)

    cam_a = generate_attribution_map(m_a, img_t, "final", fov_mask, DEVICE)
    cam_b = generate_attribution_map(m_b, img_t, "final", fov_mask, DEVICE)
    aux_b = generate_aux_head_map(m_b, img_t, fov_mask, DEVICE)
    aux_dense = generate_aux_head_map(m_dense_b, img_t, fov_mask, DEVICE)

    img1 = draw_fov_contour(draw_boxes_on_image(disp_rgb, boxes, color=(0, 255, 0), thickness=2), fov_mask)
    img2 = add_peak_marker(overlay_heatmap(disp_rgb, cam_a, alpha=0.45, fov_mask=fov_mask), cam_a, fov_mask)
    img3 = add_peak_marker(overlay_heatmap(disp_rgb, cam_b, alpha=0.45, fov_mask=fov_mask), cam_b, fov_mask)
    img4 = overlay_heatmap(disp_rgb, aux_b, alpha=0.45, colormap="magma", fov_mask=fov_mask)
    img5 = overlay_heatmap(disp_rgb, aux_dense, alpha=0.45, colormap="magma", fov_mask=fov_mask)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), dpi=300)
    style_figure(fig, axes)

    axes[0].imshow(img1)
    axes[0].set_title(f"Radiograph: {cid}\n{title_desc}", color="#f0f6fc", fontsize=10, fontweight="bold", pad=8)
    axes[0].set_xlabel("Green: GT Box | Cyan: Ω_FOV", color="#8b949e", fontsize=9)

    axes[1].imshow(img2)
    stat_a = "FRACTURE" if p_a >= 0.5 else "NORMAL"
    axes[1].set_title(f"Model A (Baseline)\nProb: {p_a:.1%} [{stat_a}]", color="#ff7b72" if p_a < 0.5 else "#56d364", fontsize=10, fontweight="bold", pad=8)
    axes[1].set_xlabel(f"Logit: {logit_a:+.2f}", color="#8b949e", fontsize=9)

    axes[2].imshow(img3)
    stat_b = "FRACTURE" if p_b >= 0.5 else "NORMAL"
    axes[2].set_title(f"Model B (BoneTrace Region)\nProb: {p_b:.1%} [{stat_b}]", color="#56d364" if p_b >= 0.5 else "#ff7b72", fontsize=10, fontweight="bold", pad=8)
    axes[2].set_xlabel(f"Logit: {logit_b:+.2f} (ΔP: {p_b - p_a:+.1%})", color="#58a6ff", fontsize=9)

    axes[3].imshow(img4)
    axes[3].set_title("ResNet-18 σ(M_aux)\nDirect Auxiliary Segmentation", color="#d29922", fontsize=10, fontweight="bold", pad=8)
    axes[3].set_xlabel(f"Peak Prob: {float(aux_b.max()):.1%}", color="#8b949e", fontsize=9)

    axes[4].imshow(img5)
    axes[4].set_title("DenseNet-121 σ(M_aux)\nCross-Backbone Aux Head", color="#58a6ff", fontsize=10, fontweight="bold", pad=8)
    axes[4].set_xlabel(f"Peak Prob: {float(aux_dense.max()):.1%}", color="#8b949e", fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_p = VIS_DIR / filename
    plt.savefig(out_p, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved Spotlight: {out_p}", flush=True)


def main():
    print("=== BoneTrace Extended Visual Figure Generation Suite ===", flush=True)
    t0 = time.time()

    p_csv = REPO_ROOT / "data" / "processed" / "test.csv"
    if not p_csv.is_file():
        p_csv = REPO_ROOT / "data" / "frozen" / "test.csv"
    df = pd.read_csv(p_csv)

    print("Loading model checkpoints...", flush=True)
    models = {
        "resnet18_baseline": load_model_for_ui("resnet18", "baseline", 123, DEVICE),
        "resnet18_region": load_model_for_ui("resnet18", "region", 123, DEVICE),
        "densenet121_baseline": load_model_for_ui("densenet121", "baseline", 42, DEVICE),
        "densenet121_region": load_model_for_ui("densenet121", "region", 42, DEVICE),
    }
    print("Models loaded. Rendering extended figures...", flush=True)

    generate_failure_modes_figure(df, models)
    generate_anatomical_diversity_figure(df, models)
    generate_hirescam_comparison(df, models)

    generate_case_spotlight(df, models, "CANON_0044", "Pelvic & Acetabular Fracture (Base 23.0% -> BoneTrace 99.4%)", "case_spotlight_CANON_0044_hip.png")
    generate_case_spotlight(df, models, "CANON_0647", "Clavicle Fracture (Base 43.0% -> BoneTrace 99.9%)", "case_spotlight_CANON_0647_shoulder.png")

    print(f"\nAll extended figures generated successfully in {time.time() - t0:.2f} seconds!", flush=True)


if __name__ == "__main__":
    main()
