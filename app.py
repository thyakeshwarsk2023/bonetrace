"""BoneTrace XAI Workstation — Clinical Radiology Triage & Explainable AI (XAI) Workstation.

Production-grade, dark-themed PACS-style interface for the BoneTrace clinical
fracture classification & explainability study.

Provides:
  - Sidebar: Backbone selector (ResNet-18 vs DenseNet-121), seed selector (42, 123, 2026),
             attribution method/layer (Final Grad-CAM, Intermediate Grad-CAM, MultiScale/HiResCAM),
             curated benchmark cases or upload, overlay controls.
  - Tab 1: Side-by-Side Triage & 4-Panel Synchronized XAI Inspector
  - Tab 2: Live Interactive Causal Perturbation Simulator (Model A vs Model B)
  - Tab 3: Multi-Seed Benchmark & Backbone Ablation Dashboard with Clinical Safety Banner

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ── Ensure src/ is on sys.path ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import streamlit as st

# ── BoneTrace UI Helpers ───────────────────────────────────────────────────────
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
    compute_pointing_game,
    run_perturbation,
    load_markdown_table,
    load_json_safe,
    get_checkpoint_path,
    scan_available_checkpoints,
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="BoneTrace XAI Workstation",
    page_icon="🦴",
    initial_sidebar_state="expanded",
)

# ── Constants & Paths ─────────────────────────────────────────────────────────
EVAL_DIR   = _REPO_ROOT / "outputs" / "evaluation"
TABLES_DIR = _REPO_ROOT / "outputs" / "tables"
VIS_DIR    = _REPO_ROOT / "outputs" / "visualizations"
FROZEN_DIR = _REPO_ROOT / "data" / "frozen"
TEST_CSV   = FROZEN_DIR / "test.csv"
SEEDS      = [42, 123, 2026]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Curated benchmark cases from Table 7 audit (held-out test cases)
CURATED_CORRECTED_IDS = [
    "CANON_0350",  # Leg: Model A p=0.026 -> Model B p=0.993 (+0.967) [y=1]
    "CANON_1617",  # Hand: Model A p=0.050 -> Model B p=0.999 (+0.950) [y=1]
    "CANON_4059",  # Hand: Model A p=0.053 -> Model B p=0.887 (+0.834) [y=1]
    "CANON_1574",  # Leg: Model A p=0.041, Model B p=0.012 [y=0]
    "CANON_3531",  # Hand: Model A p=0.063, Model B p=0.008 [y=0]
    "CANON_2386",  # Hand: Concordant True Positive [y=1]
    "CANON_1268",  # Leg: Concordant True Positive [y=1]
]

# ── Dark PACS Theme CSS ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ─── Dark Theme Radiologic PACS Palette ───────────────────────────── */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* ─── Metric Cards ─────────────────────────────────────────────────── */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 0.72rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f0f6fc;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #8b949e;
    }

    /* ─── Badges ───────────────────────────────────────────────────────── */
    .badge-pos {
        background: rgba(248, 81, 73, 0.15);
        border: 1px solid #f85149;
        color: #ff7b72;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-neg {
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        color: #3fb950;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-hit {
        background: rgba(56, 139, 253, 0.15);
        border: 1px solid #388bfd;
        color: #58a6ff;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-miss {
        background: rgba(110, 118, 129, 0.15);
        border: 1px solid #6e7681;
        color: #8b949e;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }

    /* ─── Panel Labels ─────────────────────────────────────────────────── */
    .pacs-panel-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #58a6ff;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        text-align: center;
        background: #161b22;
        padding: 6px;
        border-radius: 4px 4px 0 0;
        border: 1px solid #30363d;
        border-bottom: none;
    }

    /* ─── Clinical Safety Banner ───────────────────────────────────────── */
    .safety-banner-clinical {
        background: #1c1012;
        border: 2px solid #da3633;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 24px;
        margin-bottom: 24px;
        color: #f85149;
    }
    .safety-banner-title {
        font-weight: 800;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .safety-banner-body {
        font-size: 0.83rem;
        color: #e6edf3;
        line-height: 1.5;
    }

    /* ─── Sidebar Section Headers ──────────────────────────────────────── */
    .sb-section-hdr {
        font-size: 0.75rem;
        font-weight: 800;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 1px solid #30363d;
        padding-bottom: 4px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_cached_model(backbone: str, model_type: str, seed: int):
    """Load and cache model on GPU/CPU."""
    try:
        return load_model_for_ui(backbone, model_type, seed, DEVICE)
    except (FileNotFoundError, Exception):
        return None


@st.cache_data(show_spinner=False)
def get_cached_test_manifest() -> pd.DataFrame:
    """Load frozen test split manifest."""
    if not TEST_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(TEST_CSV)


@st.cache_data(show_spinner=False)
def get_cached_image(image_path: str):
    """Load and letterbox image."""
    return load_and_preprocess(image_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("### 🦴 BoneTrace PACS XAI")
        st.caption(f"Medical XAI Diagnostic Workstation | {DEVICE.type.upper()}")

        # ── 1. Dynamic Backbone Selector ──────────────────────────────────────
        available_ckpts = scan_available_checkpoints()
        backbone_options = [b for b in ["densenet121", "resnet18"] if b in available_ckpts]
        if not backbone_options:
            backbone_options = list(available_ckpts.keys()) or ["densenet121", "resnet18"]

        st.markdown("<div class='sb-section-hdr'>1. Backbone Architecture</div>", unsafe_allow_html=True)
        backbone = st.selectbox(
            "Backbone Model",
            options=backbone_options,
            format_func=lambda x: "DenseNet-121 (Ablation)" if x == "densenet121" else "ResNet-18 (Audited Baseline)",
            key="sb_backbone",
            label_visibility="collapsed",
        )

        # ── 2. Dynamic Seed Selector based on chosen backbone ─────────────────
        available_seeds = available_ckpts.get(backbone, [42, 123, 2026])
        if not available_seeds:
            available_seeds = [42, 123, 2026]

        # Reset session state seed if it is invalid for the newly selected backbone
        if "sb_seed" in st.session_state and st.session_state["sb_seed"] not in available_seeds:
            st.session_state["sb_seed"] = available_seeds[0]

        st.markdown("<div class='sb-section-hdr'>2. Checkpoint Seed</div>", unsafe_allow_html=True)
        seed = st.selectbox(
            "Seed",
            options=available_seeds,
            format_func=lambda x: f"Seed {x}" + (" (Primary Audit)" if x == 42 else ""),
            key="sb_seed",
            label_visibility="collapsed",
        )

        # Status of checkpoints
        col_s1, col_s2 = st.columns(2)
        ckpt_a = get_checkpoint_path(backbone, "baseline", seed)
        ckpt_b = get_checkpoint_path(backbone, "region", seed)
        with col_s1:
            st.caption(f"Model A: {'🟢 Ready' if (ckpt_a and ckpt_a.is_file()) else '🔴 Missing'}")
        with col_s2:
            st.caption(f"Model B: {'🟢 Ready' if (ckpt_b and ckpt_b.is_file()) else '🔴 Missing'}")

        # ── 3. Attribution Method & Layer ─────────────────────────────────────
        st.markdown("<div class='sb-section-hdr'>3. Attribution Method & Layer</div>", unsafe_allow_html=True)
        attr_method = st.radio(
            "Method / Layer",
            options=["final", "intermediate", "multiscale"],
            format_func=lambda x: {
                "final": "Final Stage Grad-CAM (16×16 stride 32)",
                "intermediate": "Intermediate Grad-CAM (32×32 stride 16)",
                "multiscale": "Multi-Scale / HiResCAM (Dynamic)",
            }[x],
            key="sb_attr",
            label_visibility="collapsed",
        )

        # ── 4. Radiograph Input Mode ──────────────────────────────────────────
        st.markdown("<div class='sb-section-hdr'>4. Radiograph Input Mode</div>", unsafe_allow_html=True)
        input_mode = st.radio(
            "Input Mode",
            options=["curated", "upload"],
            format_func=lambda x: "📋 Curated Benchmark Cases" if x == "curated" else "📤 Upload Custom Radiograph",
            key="sb_input_mode",
            label_visibility="collapsed",
        )

        selected_image_path: Optional[str] = None
        selected_record: Optional[Dict[str, Any]] = None

        if input_mode == "curated":
            df_test = get_cached_test_manifest()
            if df_test.empty:
                st.error("Missing `data/frozen/test.csv`")
            else:
                cat = st.selectbox(
                    "Case Category",
                    options=[
                        "⭐ Model B Corrected / Improves Most",
                        "🦴 Fracture-Positive Cases (N=107)",
                        "🛡️ Normal Controls (N=502)",
                        "📋 All Held-Out Test Cases (N=609)",
                    ],
                    key="sb_cat",
                )

                if "Model B Corrected" in cat:
                    filtered = df_test[df_test["canonical_id"].isin(CURATED_CORRECTED_IDS)]
                    if filtered.empty:
                        filtered = df_test[df_test["fractured_label"] == 1]
                elif "Fracture-Positive" in cat:
                    filtered = df_test[df_test["fractured_label"] == 1]
                elif "Normal Controls" in cat:
                    filtered = df_test[df_test["fractured_label"] == 0]
                else:
                    filtered = df_test

                cids = filtered["canonical_id"].tolist()
                cid_choice = st.selectbox("Select Case ID", options=cids, key="sb_cid")
                row = filtered[filtered["canonical_id"] == cid_choice].iloc[0]
                selected_image_path = str(row["resolved_primary_path"])
                selected_record = row.to_dict()
        else:
            up_file = st.file_uploader(
                "Upload DICOM / PNG / JPEG Radiograph",
                type=["jpg", "jpeg", "png"],
                key="sb_up",
            )
            if up_file is not None:
                from PIL import Image as PILImage
                pil_img = PILImage.open(up_file).convert("RGB")
                st.session_state["_pacs_uploaded_img"] = pil_img
                selected_image_path = "__uploaded__"
                selected_record = None

        # ── 5. Overlay Options ────────────────────────────────────────────────
        st.markdown("<div class='sb-section-hdr'>5. Overlay & Guardrails</div>", unsafe_allow_html=True)
        alpha = st.slider("Heatmap Opacity (α)", min_value=0.1, max_value=0.9, value=0.45, step=0.05, key="sb_alpha")
        show_fov = st.checkbox("Show Anatomical FOV Boundary (Cyan)", value=True, key="sb_fov")
        show_box = st.checkbox("Show Radiologist GT Box (Green)", value=True, key="sb_box")

        return {
            "backbone": backbone,
            "seed": seed,
            "attr_method": attr_method,
            "input_mode": input_mode,
            "image_path": selected_image_path,
            "record": selected_record,
            "alpha": alpha,
            "show_fov": show_fov,
            "show_box": show_box,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: SIDE-BY-SIDE TRIAGE & 4-PANEL XAI INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def render_tab_triage(cfg: Dict[str, Any]) -> None:
    st.markdown("### 🔬 Synchronized 4-Panel PACS & Saliency Inspector")

    if not cfg["image_path"]:
        st.info("👈 Please select a benchmark case or upload a radiograph in the sidebar.")
        return

    # Load image
    if cfg["image_path"] == "__uploaded__":
        pil_img = st.session_state.get("_pacs_uploaded_img")
        if pil_img is None:
            st.warning("No image uploaded.")
            return
        disp_rgb, img_t = load_and_preprocess(pil_img)
    else:
        disp_rgb, img_t = get_cached_image(cfg["image_path"])

    boxes = parse_boxes(cfg["record"].get("letterbox_target_boxes")) if cfg["record"] else []
    fov_mask = get_fov_mask(img_t)

    # Load models
    model_a = get_cached_model(cfg["backbone"], "baseline", cfg["seed"])
    model_b = get_cached_model(cfg["backbone"], "region", cfg["seed"])

    if model_a is None or model_b is None:
        st.warning("⚠️ Checkpoint not found on disk. Please select a different seed.")
        return

    # Inference & Saliency
    prob_a, logit_a, cam_a = None, None, None
    prob_b, logit_b, cam_b, aux_b = None, None, None, None

    with st.spinner("Executing inference and calculating attribution maps..."):
        if model_a is not None:
            prob_a, logit_a = run_inference(model_a, img_t, DEVICE)
            cam_a = generate_attribution_map(model_a, img_t, cfg["attr_method"], fov_mask, DEVICE)

        if model_b is not None:
            prob_b, logit_b = run_inference(model_b, img_t, DEVICE)
            cam_b = generate_attribution_map(model_b, img_t, cfg["attr_method"], fov_mask, DEVICE)
            aux_b = generate_aux_head_map(model_b, img_t, fov_mask, DEVICE)

    # Pointing Game evaluation
    hit_a, energy_a = compute_pointing_game(cam_a, boxes, fov_mask) if cam_a is not None else (False, 0.0)
    hit_b, energy_b = compute_pointing_game(cam_b, boxes, fov_mask) if cam_b is not None else (False, 0.0)

    # ── Top Metric Cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if prob_a is not None:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Model A (Baseline)</div>
                    <div class='metric-value'>{prob_a:.1%}</div>
                    <div class='metric-sub'>Logit: {logit_a:+.2f} &nbsp;|&nbsp; {"<span class='badge-pos'>FRACTURE</span>" if prob_a >= 0.5 else "<span class='badge-neg'>NORMAL</span>"}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c2:
        if prob_b is not None:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Model B (Region-Guided)</div>
                    <div class='metric-value'>{prob_b:.1%}</div>
                    <div class='metric-sub'>Logit: {logit_b:+.2f} &nbsp;|&nbsp; {"<span class='badge-pos'>FRACTURE</span>" if prob_b >= 0.5 else "<span class='badge-neg'>NORMAL</span>"}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c3:
        if boxes:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Pointing Game (Model A)</div>
                    <div class='metric-value'>{"<span class='badge-hit'>HIT</span>" if hit_a else "<span class='badge-miss'>MISS</span>"}</div>
                    <div class='metric-sub'>Energy in Box: {energy_a:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class='metric-card'>
                    <div class='metric-label'>Pointing Game (Model A)</div>
                    <div class='metric-value'>N/A</div>
                    <div class='metric-sub'>No GT Box (Normal)</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c4:
        if boxes:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Pointing Game (Model B)</div>
                    <div class='metric-value'>{"<span class='badge-hit'>HIT</span>" if hit_b else "<span class='badge-miss'>MISS</span>"}</div>
                    <div class='metric-sub'>Energy in Box: {energy_b:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class='metric-card'>
                    <div class='metric-label'>Pointing Game (Model B)</div>
                    <div class='metric-value'>N/A</div>
                    <div class='metric-sub'>No GT Box (Normal)</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c5:
        if prob_a is not None and prob_b is not None:
            delta_p = prob_b - prob_a
            dp_color = "#3fb950" if delta_p >= 0 else "#f85149"
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Probability Shift (B − A)</div>
                    <div class='metric-value' style='color:{dp_color}'>{delta_p:+.1%}</div>
                    <div class='metric-sub'>Δ Logit: {(logit_b - logit_a):+.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ── 4-Column Synchronized Visual Grid ─────────────────────────────────────
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown("<div class='pacs-panel-title'>Panel 1: Input Radiograph (512×512)</div>", unsafe_allow_html=True)
        img1 = disp_rgb.copy()
        if cfg["show_box"] and boxes:
            img1 = draw_boxes_on_image(img1, boxes, color=(0, 255, 0), thickness=2)
        if cfg["show_fov"]:
            img1 = draw_fov_contour(img1, fov_mask, color=(0, 255, 255), thickness=1)
        st.image(img1, use_container_width=True)
        st.caption("Standardized canvas with letterbox padding excluded (Ω_FOV). Green: Radiologist GT Box.")

    with p2:
        st.markdown("<div class='pacs-panel-title'>Panel 2: Model A (Baseline) Saliency</div>", unsafe_allow_html=True)
        if cam_a is not None:
            img2 = overlay_heatmap(disp_rgb, cam_a, alpha=cfg["alpha"], fov_mask=fov_mask)
            img2 = add_peak_marker(img2, cam_a, fov_mask=fov_mask, color=(255, 255, 0))
            if cfg["show_box"] and boxes:
                img2 = draw_boxes_on_image(img2, boxes, color=(0, 255, 0), thickness=2)
            st.image(img2, use_container_width=True)
            st.caption("First-order gradient class activation map. Yellow '+': Attribution peak.")
        else:
            st.info("Baseline model checkpoint not loaded.")

    with p3:
        st.markdown("<div class='pacs-panel-title'>Panel 3: Model B (Region) Saliency</div>", unsafe_allow_html=True)
        if cam_b is not None:
            img3 = overlay_heatmap(disp_rgb, cam_b, alpha=cfg["alpha"], fov_mask=fov_mask)
            img3 = add_peak_marker(img3, cam_b, fov_mask=fov_mask, color=(255, 255, 0))
            if cfg["show_box"] and boxes:
                img3 = draw_boxes_on_image(img3, boxes, color=(0, 255, 0), thickness=2)
            st.image(img3, use_container_width=True)
            st.caption("Region-supervised class activation map. Note anatomical focus.")
        else:
            st.info("Region model checkpoint not loaded.")

    with p4:
        st.markdown("<div class='pacs-panel-title'>Panel 4: Model B Auxiliary Head σ(M_aux)</div>", unsafe_allow_html=True)
        if aux_b is not None:
            img4 = overlay_heatmap(disp_rgb, aux_b, alpha=cfg["alpha"], colormap="magma", fov_mask=fov_mask)
            if cfg["show_box"] and boxes:
                img4 = draw_boxes_on_image(img4, boxes, color=(0, 255, 0), thickness=2)
            st.image(img4, use_container_width=True)
            st.caption("Direct spatial prediction mask (512×512) prior to Global Average Pooling.")
        else:
            st.info("Direct auxiliary head only available for Region-Supervised Model B.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: LIVE INTERACTIVE CAUSAL PERTURBATION SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def render_tab_perturbation(cfg: Dict[str, Any]) -> None:
    st.markdown("### ⚗️ Live FOV-Constrained Causal Perturbation Simulator")
    st.caption("Quantifies causal faithfulness by evaluating logit drops under FOV-isolated pixel masking.")

    if not cfg["image_path"]:
        st.info("👈 Please select a benchmark case or upload a radiograph in the sidebar.")
        return

    # Load image
    if cfg["image_path"] == "__uploaded__":
        disp_rgb, img_t = load_and_preprocess(st.session_state["_pacs_uploaded_img"])
    else:
        disp_rgb, img_t = get_cached_image(cfg["image_path"])

    boxes = parse_boxes(cfg["record"].get("letterbox_target_boxes")) if cfg["record"] else []
    fov_mask = get_fov_mask(img_t)

    # Interactive Controls
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)

    with c_ctrl1:
        k_pct = st.slider("Mask Top k% Salient FOV Pixels", min_value=1, max_value=30, value=10, step=1, key="pert_k_pct")

    with c_ctrl2:
        fill_strat = st.selectbox(
            "Fill Strategy",
            options=["zero", "blur"],
            format_func=lambda x: "Zero Fill (ImageNet Mean = 0.0)" if x == "zero" else "Gaussian Blur (Anatomical Smoothing)",
            key="pert_fill",
        )

    with c_ctrl3:
        mode_options = ["top_k_vs_random"]
        mode_labels = {"top_k_vs_random": "Top-k% Attributed vs. Equal-Area Random FOV Control"}
        if boxes:
            mode_options.append("box_vs_outside")
            mode_labels["box_vs_outside"] = "Area-Matched Lesion Box vs. Outside Context"

        pert_mode = st.radio(
            "Perturbation Mode",
            options=mode_options,
            format_func=lambda x: mode_labels[x],
            key="pert_mode_radio",
        )

    # Load models
    model_a = get_cached_model(cfg["backbone"], "baseline", cfg["seed"])
    model_b = get_cached_model(cfg["backbone"], "region", cfg["seed"])

    if model_a is None or model_b is None:
        st.warning("⚠️ Checkpoint not found on disk. Please select a different seed.")
        return

    # Compute saliency maps for perturbation guidance
    cam_a = generate_attribution_map(model_a, img_t, cfg["attr_method"], fov_mask, DEVICE) if model_a else None
    cam_b = generate_attribution_map(model_b, img_t, cfg["attr_method"], fov_mask, DEVICE) if model_b else None

    # Run perturbations for both models
    with st.spinner("Computing live causal perturbation forward passes..."):
        res_a = run_perturbation(
            model=model_a,
            img_tensor=img_t,
            cam=cam_a,
            k_pct=k_pct,
            fill_strategy=fill_strat,
            mode=pert_mode,
            boxes=boxes,
            fov_mask=fov_mask,
            device=DEVICE,
        ) if model_a and cam_a is not None else None

        res_b = run_perturbation(
            model=model_b,
            img_tensor=img_t,
            cam=cam_b,
            k_pct=k_pct,
            fill_strategy=fill_strat,
            mode=pert_mode,
            boxes=boxes,
            fov_mask=fov_mask,
            device=DEVICE,
        ) if model_b and cam_b is not None else None

    # Display side-by-side masked images
    st.markdown("#### 🖼️ Masked Perturbation Inputs")
    m_col1, m_col2, m_col3 = st.columns(3)

    ref_res = res_b if res_b is not None else res_a
    with m_col1:
        st.markdown("<div class='pacs-panel-title'>Original Image (Ω_FOV)</div>", unsafe_allow_html=True)
        st.image(disp_rgb, use_container_width=True)

    with m_col2:
        title_attr = "Masked: Top Attributed Region" if pert_mode == "top_k_vs_random" else "Masked: GT Fracture Box"
        st.markdown(f"<div class='pacs-panel-title'>{title_attr}</div>", unsafe_allow_html=True)
        if ref_res:
            st.image(ref_res["img_masked_attr"], use_container_width=True)

    with m_col3:
        title_ctrl = "Masked: Equal-Area Random FOV" if pert_mode == "top_k_vs_random" else "Masked: Area-Matched Outside Box"
        st.markdown(f"<div class='pacs-panel-title'>{title_ctrl}</div>", unsafe_allow_html=True)
        if ref_res:
            st.image(ref_res["img_masked_ctrl"], use_container_width=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Quantitative Comparison: Model A vs Model B
    st.markdown("#### 📊 Comparative Causal Advantage (Model A vs. Model B)")

    c_met1, c_met2, c_met3, c_met4 = st.columns(4)

    ca_a = res_a["causal_advantage"] if res_a else 0.0
    ca_b = res_b["causal_advantage"] if res_b else 0.0

    with c_met1:
        if res_a:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Model A: Δy_attr Drop</div>
                    <div class='metric-value'>{res_a['dy_attr']:+.3f}</div>
                    <div class='metric-sub'>Orig: {res_a['logit_orig']:+.2f} → Mask: {res_a['logit_attr']:+.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c_met2:
        if res_a:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Model A: Causal Advantage</div>
                    <div class='metric-value' style='color:{"#3fb950" if ca_a > 0 else "#f85149"}'>{ca_a:+.3f}</div>
                    <div class='metric-sub'>Δy_ctrl: {res_a['dy_ctrl']:+.3f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c_met3:
        if res_b:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Model B: Δy_attr Drop</div>
                    <div class='metric-value'>{res_b['dy_attr']:+.3f}</div>
                    <div class='metric-sub'>Orig: {res_b['logit_orig']:+.2f} → Mask: {res_b['logit_attr']:+.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c_met4:
        if res_b:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Model B: Causal Advantage</div>
                    <div class='metric-value' style='color:{"#3fb950" if ca_b > 0 else "#f85149"}'>{ca_b:+.3f}</div>
                    <div class='metric-sub'>Δy_ctrl: {res_b['dy_ctrl']:+.3f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Bar Chart comparison
    import plotly.graph_objects as go
    fig = go.Figure()

    labels = ["Original Logit", "Attributed Mask Logit", "Control Mask Logit", "Causal Advantage (CA)"]
    if res_a:
        vals_a = [res_a["logit_orig"], res_a["logit_attr"], res_a["logit_ctrl"], res_a["causal_advantage"]]
        fig.add_trace(go.Bar(name="Model A (Baseline)", x=labels, y=vals_a, marker_color="#f85149"))

    if res_b:
        vals_b = [res_b["logit_orig"], res_b["logit_attr"], res_b["logit_ctrl"], res_b["causal_advantage"]]
        fig.add_trace(go.Bar(name="Model B (Region-Guided)", x=labels, y=vals_b, marker_color="#388bfd"))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        yaxis=dict(title="Logit Value / Delta", gridcolor="#30363d"),
        xaxis=dict(gridcolor="#30363d"),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: MULTI-SEED BENCHMARK & BACKBONE ABLATION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def render_tab_benchmark(cfg: Dict[str, Any]) -> None:
    st.markdown("### 📊 Multi-Seed Audited Benchmark & Backbone Ablation")
    st.caption("Standardized evaluation across the 4 core clinical AI axes (Discrimination, Calibration, Localization, Faithfulness).")

    sub_t1, sub_t2, sub_t3, sub_t4, sub_t5 = st.tabs([
        "🧬 DenseNet-121 Ablation",
        "🎯 Axis 1 & 4: Discrimination & Calibration",
        "📍 Axis 3: Spatial Localization",
        "⚖️ Axis 2: Causal Faithfulness",
        "🖼️ Publication Figures",
    ])

    # ── Subtab 1: DenseNet-121 Ablation ───────────────────────────────────────
    with sub_t1:
        st.markdown("#### DenseNet-121 Ablation Study (Head-to-Head vs. ResNet-18)")
        densenet_table_path = TABLES_DIR / "T_ablation_densenet.md"
        densenet_json_path = EVAL_DIR / "densenet121_ablation_metrics.json"

        col_b1, col_b2 = st.columns([4, 1])
        with col_b2:
            if st.button("🔄 Refresh Results", key="refresh_densenet_btn"):
                st.rerun()

        if densenet_table_path.is_file():
            md_content = load_markdown_table(densenet_table_path)
            st.markdown(md_content)
        else:
            st.info(
                "⏳ **DenseNet-121 evaluation in progress or pending.**\n\n"
                "Run `scripts/evaluate_densenet_ablation.py` in your PowerShell terminal to generate publication tables and raw metrics.\n\n"
                "Once complete, click the **Refresh Results** button above."
            )

        if densenet_json_path.is_file():
            with st.expander("📄 View Raw DenseNet-121 Metrics JSON", expanded=False):
                data = load_json_safe(densenet_json_path)
                st.json(data)

    # ── Subtab 2: Axis 1 & 4: Discrimination & Calibration ────────────────────
    with sub_t2:
        st.markdown("#### Classification Discrimination & Probability Calibration")
        t1_content = load_markdown_table(TABLES_DIR / "T1.md")
        if t1_content:
            st.markdown(t1_content)
        else:
            st.info("Table T1 not found.")

        t5_content = load_markdown_table(TABLES_DIR / "T5.md")
        if t5_content:
            st.markdown(t5_content)

    # ── Subtab 3: Axis 3: Spatial Localization ────────────────────────────────
    with sub_t3:
        st.markdown("#### Spatial Localization Agreement (N=107 Fracture-Positives)")
        t4_content = load_markdown_table(TABLES_DIR / "T4.md")
        if t4_content:
            st.markdown(t4_content)
        else:
            st.info("Table T4 not found.")

    # ── Subtab 4: Axis 2: Causal Faithfulness ─────────────────────────────────
    with sub_t4:
        st.markdown("#### FOV-Constrained Causal Faithfulness & Area-Matched Lesion Dependence")
        t3_content = load_markdown_table(TABLES_DIR / "T3.md")
        if t3_content:
            st.markdown(t3_content)
        else:
            st.info("Table T3 not found.")

    # ── Subtab 5: Publication Figures ─────────────────────────────────────────
    with sub_t5:
        st.markdown("#### Audited Publication Diagnostic Visualizations")
        f_col1, f_col2 = st.columns(2)

        p_cal = VIS_DIR / "calibration_diagrams.png"
        p_roc = VIS_DIR / "publication_curves.png"
        p_grid = VIS_DIR / "qualitative_4panel_grid.png"
        p_cam = VIS_DIR / "gradcam_qualitative_grid.png"

        with f_col1:
            if p_roc.is_file():
                st.image(str(p_roc), caption="Figure 1: Multi-Seed ROC & PR Discrimination Curves", use_container_width=True)
            if p_cal.is_file():
                st.image(str(p_cal), caption="Figure 2: Reliability Diagrams & Calibration Curves", use_container_width=True)

        with f_col2:
            if p_grid.is_file():
                st.image(str(p_grid), caption="Figure 3: Synchronized 4-Panel Qualitative Cases", use_container_width=True)
            if p_cam.is_file():
                st.image(str(p_cam), caption="Figure 4: Comparative Grad-CAM Saliency Grid", use_container_width=True)

    # ── Clinical Safety Banner ────────────────────────────────────────────────
    st.markdown(
        """
        <div class='safety-banner-clinical'>
            <div class='safety-banner-title'>⚠️ Clinical Safety & Diagnostic Guardrails Notice</div>
            <div class='safety-banner-body'>
                <strong>BoneTrace is an investigational research prototype for triage and worklist prioritization only.</strong><br/>
                On the audited, frozen held-out benchmark (N = 609), model sensitivity is <strong>61.0%</strong> at the default <code>p &ge; 0.50</code> threshold (ResNet-18 Region-Supervised, Seed 42). 
                Consequently, <strong>approx. 39% of fractures are not flagged at this threshold</strong>.<br/>
                <strong>Explainability Constraint:</strong> Post-hoc Grad-CAM heatmaps are first-order gradient approximations and do NOT constitute segmentations or lesion boundaries. They must never be used to delineate surgical margins or establish diagnosis without licensed radiologist review.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "🔬 Side-by-Side Triage & 4-Panel XAI",
        "⚗️ Live Causal Perturbation Simulator",
        "📊 Multi-Seed Benchmark & Ablation Dashboard",
    ])

    with tab1:
        render_tab_triage(cfg)

    with tab2:
        render_tab_perturbation(cfg)

    with tab3:
        render_tab_benchmark(cfg)


if __name__ == "__main__":
    main()
