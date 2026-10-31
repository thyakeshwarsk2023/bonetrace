"""BoneTrace XAI Workstation — Clinical Radiology Triage & Explainable AI (XAI) Workstation.

Production-grade, dark-themed PACS-style interface for the BoneTrace clinical
fracture classification & explainability study.

Streamlined with minimal, intuitive controls:
  - Sidebar:
      1. Image Source: Upload Any X-Ray Image vs. Curated Benchmark Test Cases
      2. Model Architecture: Dynamic discovery dropdown with zero-crash loading
      3. Saliency Method: Grad-CAM (Standard) vs. HiResCAM / Multi-Scale (Sharp)
      4. Overlay Opacity: Clean alpha slider (0.1 to 0.9)
  - Tab 1: 4-Panel Synchronized Diagnostic & XAI Inspector
  - Tab 2: 1-Click FOV-Constrained Causal Faithfulness Simulator
  - Tab 3: Multi-Seed Benchmark & Backbone Ablation Dashboard
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
    compute_unsupervised_metrics,
    run_perturbation,
    load_markdown_table,
    load_json_safe,
    get_checkpoint_path,
    discover_available_model_pairs,
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
VIS_DIR    = _REPO_ROOT / "visualizations"
PROCESSED_TEST_CSV = _REPO_ROOT / "data" / "processed" / "test.csv"
FROZEN_TEST_CSV    = _REPO_ROOT / "data" / "frozen" / "test.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 10 Curated Benchmark Cases (from Table 7 & Test Cohort)
CURATED_BENCHMARK_CASES = [
    {
        "id": "CANON_0350",
        "title": "CANON_0350 — Leg: Hairline Fracture (Baseline: 21.8% → BoneTrace: 98.2%, +76.4%) [GT: Fracture]",
        "body_part": "Leg",
        "gt": 1,
    },
    {
        "id": "CANON_1617",
        "title": "CANON_1617 — Hand: Subtle Fracture (Baseline: 34.1% → BoneTrace: 94.5%, +60.4%) [GT: Fracture]",
        "body_part": "Hand",
        "gt": 1,
    },
    {
        "id": "CANON_4059",
        "title": "CANON_4059 — Hand: Oblique Cortical Break (Baseline: 29.5% → BoneTrace: 88.7%, +59.2%) [GT: Fracture]",
        "body_part": "Hand",
        "gt": 1,
    },
    {
        "id": "CANON_2386",
        "title": "CANON_2386 — Hand: Displaced Fracture (Concordant Positive: Base 94.2%, BoneTrace 98.9%) [GT: Fracture]",
        "body_part": "Hand",
        "gt": 1,
    },
    {
        "id": "CANON_1268",
        "title": "CANON_1268 — Leg: Cortical Disruption (Concordant Positive: Base 91.5%, BoneTrace 97.6%) [GT: Fracture]",
        "body_part": "Leg",
        "gt": 1,
    },
    {
        "id": "CANON_1574",
        "title": "CANON_1574 — Leg: Normal Bone Cortex (Concordant Negative: Base 4.1%, BoneTrace 1.2%) [GT: Normal]",
        "body_part": "Leg",
        "gt": 0,
    },
    {
        "id": "CANON_3531",
        "title": "CANON_3531 — Hand: Normal Joint Space (Concordant Negative: Base 6.3%, BoneTrace 0.8%) [GT: Normal]",
        "body_part": "Hand",
        "gt": 0,
    },
    {
        "id": "CANON_1227",
        "title": "CANON_1227 — Hand: Complex Carpal Margins (Baseline False Alarm 79.5% → Corrected) [GT: Normal]",
        "body_part": "Hand",
        "gt": 0,
    },
    {
        "id": "CANON_1544",
        "title": "CANON_1544 — Shoulder: Normal Clavicle Border (Baseline False Alarm 88.2% → Corrected) [GT: Normal]",
        "body_part": "Shoulder",
        "gt": 0,
    },
    {
        "id": "CANON_0390",
        "title": "CANON_0390 — Hand: Subtle Epiphyseal Joint Fracture [GT: Fracture]",
        "body_part": "Hand",
        "gt": 1,
    },
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
        border-radius: 6px;
        padding: 12px 14px;
        text-align: center;
        height: 100%;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #f0f6fc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #8b949e;
        margin-top: 4px;
    }
    .badge-pos {
        background: rgba(248, 81, 73, 0.15);
        border: 1px solid #f85149;
        color: #ff7b72;
        border-radius: 16px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-neg {
        background: rgba(63, 185, 80, 0.15);
        border: 1px solid #3fb950;
        color: #56d364;
        border-radius: 16px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-hit {
        background: rgba(56, 139, 253, 0.15);
        border: 1px solid #388bfd;
        color: #58a6ff;
        border-radius: 16px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-miss {
        background: rgba(110, 118, 129, 0.15);
        border: 1px solid #6e7681;
        color: #8b949e;
        border-radius: 16px;
        padding: 3px 10px;
        font-size: 0.75rem;
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
    """Load and cache model on GPU/CPU with key harmonization."""
    try:
        return load_model_for_ui(backbone, model_type, seed, DEVICE)
    except Exception as e:
        return None


@st.cache_data(show_spinner=False)
def get_cached_test_manifest() -> pd.DataFrame:
    """Load clean test split manifest (preferring data/processed/test.csv)."""
    if PROCESSED_TEST_CSV.is_file():
        return pd.read_csv(PROCESSED_TEST_CSV)
    if FROZEN_TEST_CSV.is_file():
        return pd.read_csv(FROZEN_TEST_CSV)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# MINIMAL 4-CONTROL SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("### 🦴 BoneTrace PACS XAI")
        st.caption(f"Clinical AI Workstation | {DEVICE.type.upper()}")

        # ── Control 1: Radiograph Image Source ────────────────────────────────
        st.markdown("<div class='sb-section-hdr'>1. Radiograph Source</div>", unsafe_allow_html=True)
        input_mode = st.radio(
            "Image Source",
            options=["upload", "curated"],
            format_func=lambda x: "📤 Upload Any X-Ray Image" if x == "upload" else "📋 Sample Benchmark Cases (10)",
            key="sb_input_mode",
            label_visibility="collapsed",
        )

        selected_image_source: Any = None
        selected_record: Optional[Dict[str, Any]] = None

        if input_mode == "upload":
            up_file = st.file_uploader(
                "Drag & drop any X-Ray image (.png, .jpg, .jpeg, .tif)",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                key="sb_up_file",
                help="Accepts any aspect ratio, resolution, 16-bit medical grayscale, RGB, or RGBA.",
            )
            if up_file is not None:
                selected_image_source = up_file
            else:
                st.caption("No image uploaded. Defaulting to benchmark case CANON_0350 preview.")
                # Fallback to CANON_0350 so the workstation always starts with an active view
                df_test = get_cached_test_manifest()
                sub = df_test[df_test["canonical_id"] == "CANON_0350"]
                if not sub.empty:
                    selected_record = sub.iloc[0].to_dict()
                    selected_image_source = selected_record.get("resolved_primary_path")
        else:
            df_test = get_cached_test_manifest()
            options_labels = [c["title"] for c in CURATED_BENCHMARK_CASES]
            choice_label = st.selectbox(
                "Select Benchmark Case",
                options=options_labels,
                index=0,
                key="sb_curated_choice",
                label_visibility="collapsed",
            )
            choice_item = next(c for c in CURATED_BENCHMARK_CASES if c["title"] == choice_label)
            cid = choice_item["id"]

            sub = df_test[df_test["canonical_id"] == cid] if not df_test.empty else pd.DataFrame()
            if not sub.empty:
                selected_record = sub.iloc[0].to_dict()
                selected_image_source = selected_record.get("resolved_primary_path")
            else:
                selected_image_source = None

        # ── Control 2: Model Architecture Discovery Dropdown ─────────────────
        st.markdown("<div class='sb-section-hdr'>2. Model Architecture</div>", unsafe_allow_html=True)
        available_pairs = discover_available_model_pairs()

        if not available_pairs:
            st.error("No valid checkpoint pairs found in `outputs/checkpoints/`.")
            selected_pair = {"backbone": "densenet121", "seed": 42}
        else:
            pair_map = {p["id"]: p for p in available_pairs}
            pair_choice_id = st.selectbox(
                "Select Model Backbone & Seed",
                options=list(pair_map.keys()),
                format_func=lambda x: pair_map[x]["label"],
                index=0,
                key="sb_pair_select",
                label_visibility="collapsed",
            )
            selected_pair = pair_map[pair_choice_id]

        backbone = selected_pair["backbone"]
        seed = selected_pair["seed"]

        # ── Control 3: Saliency Method ────────────────────────────────────────
        st.markdown("<div class='sb-section-hdr'>3. Saliency Method</div>", unsafe_allow_html=True)
        attr_label = st.radio(
            "Attribution Method",
            options=["Grad-CAM (Standard)", "HiResCAM / Multi-Scale (Sharp)"],
            index=0,
            key="sb_attr_method",
            label_visibility="collapsed",
        )
        attr_method = "multiscale" if "HiResCAM" in attr_label else "final"

        # ── Control 4: Heatmap Opacity Slider ─────────────────────────────────
        st.markdown("<div class='sb-section-hdr'>4. Overlay Opacity</div>", unsafe_allow_html=True)
        alpha = st.slider(
            "Opacity (α)",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            key="sb_alpha",
            label_visibility="collapsed",
        )

        # ── Collapsible Advanced XAI Settings ─────────────────────────────────
        with st.expander("⚙️ Advanced Model & Display Settings", expanded=False):
            st.caption(f"**Selected:** {backbone.upper()} (Seed {seed})")
            ckpt_a = get_checkpoint_path(backbone, "baseline", seed)
            ckpt_b = get_checkpoint_path(backbone, "region", seed)
            st.caption(f"Model A: {'🟢 Ready' if (ckpt_a and ckpt_a.is_file()) else '🔴 Missing'}")
            st.caption(f"Model B: {'🟢 Ready' if (ckpt_b and ckpt_b.is_file()) else '🔴 Missing'}")

            layer_override = st.selectbox(
                "Target Feature Layer",
                options=["Final Stage (Stride 32)", "Intermediate Stage (Stride 16)"],
                index=0,
                key="sb_layer_override",
            )
            if "Intermediate" in layer_override:
                attr_method = "intermediate"

            show_fov = st.checkbox("Show Anatomical FOV Boundary (Cyan)", value=True, key="sb_show_fov")
            show_box = st.checkbox("Show Radiologist GT Box (Green)", value=True, key="sb_show_box")

        return {
            "backbone": backbone,
            "seed": seed,
            "attr_method": attr_method,
            "image_source": selected_image_source,
            "record": selected_record,
            "alpha": alpha,
            "show_fov": show_fov,
            "show_box": show_box,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: 4-PANEL SYNCHRONIZED DIAGNOSTIC & XAI INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def render_tab_triage(cfg: Dict[str, Any]) -> None:
    st.markdown("### 🔬 Synchronized 4-Panel PACS & Saliency Inspector")

    if cfg["image_source"] is None:
        st.info("👈 Please upload an X-Ray image or select a sample benchmark case in the sidebar.")
        return

    # Universal image loading & preprocessing (handles arbitrary inputs safely)
    try:
        disp_rgb, img_t = load_and_preprocess(cfg["image_source"])
    except Exception as e:
        st.error(f"Failed to process image: {e}")
        return

    boxes = parse_boxes(cfg["record"].get("letterbox_target_boxes")) if cfg["record"] else []
    fov_mask = get_fov_mask(img_t)

    # Load models
    model_a = get_cached_model(cfg["backbone"], "baseline", cfg["seed"])
    model_b = get_cached_model(cfg["backbone"], "region", cfg["seed"])

    if model_a is None or model_b is None:
        st.warning("⚠️ Checkpoint file missing for selected backbone/seed. Please select an available pair.")
        return

    # Forward inference & saliency map generation
    with st.spinner("Computing forward passes and attribution maps..."):
        prob_a, logit_a = run_inference(model_a, img_t, DEVICE)
        cam_a = generate_attribution_map(model_a, img_t, cfg["attr_method"], fov_mask, DEVICE)

        prob_b, logit_b = run_inference(model_b, img_t, DEVICE)
        cam_b = generate_attribution_map(model_b, img_t, cfg["attr_method"], fov_mask, DEVICE)
        aux_b = generate_aux_head_map(model_b, img_t, fov_mask, DEVICE)

    # Evaluation metrics
    if boxes:
        hit_a, energy_a = compute_pointing_game(cam_a, boxes, fov_mask)
        hit_b, energy_b = compute_pointing_game(cam_b, boxes, fov_mask)
        unsup_metrics = None
    else:
        hit_a, energy_a = False, 0.0
        hit_b, energy_b = False, 0.0
        unsup_metrics = compute_unsupervised_metrics(cam_b, aux_b, fov_mask)

    # ── Top Summary Metric Cards ──────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Model A (Baseline)</div>
                <div class='metric-value'>{prob_a:.1%}</div>
                <div class='metric-sub'>Logit: {logit_a:+.2f} &nbsp;|&nbsp; {"<span class='badge-pos'>FRACTURE</span>" if prob_a >= 0.5 else "<span class='badge-neg'>NORMAL</span>"}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Model B (BoneTrace Region)</div>
                <div class='metric-value'>{prob_b:.1%}</div>
                <div class='metric-sub'>Logit: {logit_b:+.2f} &nbsp;|&nbsp; {"<span class='badge-pos'>FRACTURE</span>" if prob_b >= 0.5 else "<span class='badge-neg'>NORMAL</span>"}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with c3:
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

    with c4:
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
                f"""<div class='metric-card'>
                    <div class='metric-label'>Attribution Peak Confidence</div>
                    <div class='metric-value'>{unsup_metrics['peak_cam_confidence']:.2f}</div>
                    <div class='metric-sub'>FOV Max Saliency</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with c5:
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
            aux_val = unsup_metrics['aux_peak_activation']
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-label'>Auxiliary Head Peak σ(M_aux)</div>
                    <div class='metric-value'>{aux_val:.1%}</div>
                    <div class='metric-sub'>Direct Dense Lesion Prob</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

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
        cap1 = "Standardized 512×512 canvas. Cyan: Anatomical FOV Boundary (Ω_FOV)."
        if boxes:
            cap1 += " Green: Radiologist GT Box."
        st.caption(cap1)

    with p2:
        st.markdown("<div class='pacs-panel-title'>Panel 2: Model A (Baseline) Saliency</div>", unsafe_allow_html=True)
        img2 = overlay_heatmap(disp_rgb, cam_a, alpha=cfg["alpha"], fov_mask=fov_mask)
        img2 = add_peak_marker(img2, cam_a, fov_mask=fov_mask, color=(255, 255, 0))
        if cfg["show_box"] and boxes:
            img2 = draw_boxes_on_image(img2, boxes, color=(0, 255, 0), thickness=2)
        st.image(img2, use_container_width=True)
        st.caption("Standard classifier Grad-CAM. Yellow '+': Attribution peak.")

    with p3:
        st.markdown("<div class='pacs-panel-title'>Panel 3: Model B (Region) Saliency</div>", unsafe_allow_html=True)
        img3 = overlay_heatmap(disp_rgb, cam_b, alpha=cfg["alpha"], fov_mask=fov_mask)
        img3 = add_peak_marker(img3, cam_b, fov_mask=fov_mask, color=(255, 255, 0))
        if cfg["show_box"] and boxes:
            img3 = draw_boxes_on_image(img3, boxes, color=(0, 255, 0), thickness=2)
        st.image(img3, use_container_width=True)
        st.caption("Region-supervised classification Grad-CAM. Note anatomical focus.")

    with p4:
        st.markdown("<div class='pacs-panel-title'>Panel 4: Model B Auxiliary Head σ(M_aux)</div>", unsafe_allow_html=True)
        if aux_b is not None:
            img4 = overlay_heatmap(disp_rgb, aux_b, alpha=cfg["alpha"], colormap="magma", fov_mask=fov_mask)
            if cfg["show_box"] and boxes:
                img4 = draw_boxes_on_image(img4, boxes, color=(0, 255, 0), thickness=2)
            st.image(img4, use_container_width=True)
            st.caption("Direct unpooled spatial prediction map (512×512) prior to GAP.")
        else:
            st.info("Direct auxiliary head not available for baseline model.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: 1-CLICK LIVE CAUSAL PERTURBATION SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def render_tab_perturbation(cfg: Dict[str, Any]) -> None:
    st.markdown("### ⚗️ 1-Click FOV-Constrained Causal Faithfulness Simulator")
    st.caption("Evaluates causal attribution faithfulness by measuring classification logit drops when salient tissue is masked.")

    if cfg["image_source"] is None:
        st.info("👈 Please upload an X-Ray image or select a sample benchmark case in the sidebar.")
        return

    disp_rgb, img_t = load_and_preprocess(cfg["image_source"])
    boxes = parse_boxes(cfg["record"].get("letterbox_target_boxes")) if cfg["record"] else []
    fov_mask = get_fov_mask(img_t)

    # Interactive slider (single clean control)
    col_k, col_info = st.columns([2, 3])
    with col_k:
        k_pct = st.slider("Mask Top k% Salient FOV Pixels", min_value=1, max_value=30, value=10, step=1, key="pert_k_slider")
    with col_info:
        st.caption(
            f"**Audit Protocol:** Replaces the top-{k_pct}% most salient pixels strictly within the patient's anatomical field of view (Ω_FOV) "
            "with standardized ImageNet mean (0.0). Excludes letterbox padding borders to guarantee authentic causal measurement."
        )

    # Load models
    model_a = get_cached_model(cfg["backbone"], "baseline", cfg["seed"])
    model_b = get_cached_model(cfg["backbone"], "region", cfg["seed"])

    if model_a is None or model_b is None:
        st.warning("⚠️ Checkpoint file missing for selected backbone/seed.")
        return

    cam_a = generate_attribution_map(model_a, img_t, cfg["attr_method"], fov_mask, DEVICE)
    cam_b = generate_attribution_map(model_b, img_t, cfg["attr_method"], fov_mask, DEVICE)

    # Run perturbations
    with st.spinner("Computing causal perturbation forward passes..."):
        res_a = run_perturbation(
            model=model_a,
            img_tensor=img_t,
            cam=cam_a,
            k_pct=k_pct,
            fill_strategy="zero",
            mode="top_k_vs_random",
            boxes=boxes,
            fov_mask=fov_mask,
            device=DEVICE,
        )
        res_b = run_perturbation(
            model=model_b,
            img_tensor=img_t,
            cam=cam_b,
            k_pct=k_pct,
            fill_strategy="zero",
            mode="top_k_vs_random",
            boxes=boxes,
            fov_mask=fov_mask,
            device=DEVICE,
        )

    # ── Display Masked Input Images ───────────────────────────────────────────
    st.markdown("#### 🖼️ Masked Perturbation Inputs")
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown("<div class='pacs-panel-title'>Original Image (Ω_FOV)</div>", unsafe_allow_html=True)
        st.image(disp_rgb, use_container_width=True)

    with m_col2:
        st.markdown(f"<div class='pacs-panel-title'>Masked: Top-{k_pct}% Attributed Pixels</div>", unsafe_allow_html=True)
        st.image(res_b["img_masked_attr"], use_container_width=True)

    with m_col3:
        st.markdown(f"<div class='pacs-panel-title'>Masked: Equal-Area Random FOV Control</div>", unsafe_allow_html=True)
        st.image(res_b["img_masked_ctrl"], use_container_width=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ── Quantitative Metric Cards ─────────────────────────────────────────────
    st.markdown("#### 📊 Causal Advantage Comparison (Model A vs. Model B)")
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)

    ca_a = res_a["causal_advantage"]
    ca_b = res_b["causal_advantage"]

    with c_m1:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Model A: Δy_attr Drop</div>
                <div class='metric-value'>{res_a['dy_attr']:+.3f}</div>
                <div class='metric-sub'>Orig: {res_a['logit_orig']:+.2f} → Mask: {res_a['logit_attr']:+.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with c_m2:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Model A: Causal Advantage</div>
                <div class='metric-value' style='color:{"#3fb950" if ca_a > 0 else "#f85149"}'>{ca_a:+.3f}</div>
                <div class='metric-sub'>Control Drop Δy_ctrl: {res_a['dy_ctrl']:+.3f}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with c_m3:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Model B: Δy_attr Drop</div>
                <div class='metric-value'>{res_b['dy_attr']:+.3f}</div>
                <div class='metric-sub'>Orig: {res_b['logit_orig']:+.2f} → Mask: {res_b['logit_attr']:+.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with c_m4:
        st.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>Model B: Causal Advantage</div>
                <div class='metric-value' style='color:{"#3fb950" if ca_b > 0 else "#f85149"}'>{ca_b:+.3f}</div>
                <div class='metric-sub'>Control Drop Δy_ctrl: {res_b['dy_ctrl']:+.3f}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Comparative Native Chart (Zero-Dependency) ─────────────────────────────
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 📈 Side-by-Side Faithfulness Response")
    df_chart = pd.DataFrame(
        {
            "Model A (Baseline)": [res_a["dy_attr"], res_a["dy_ctrl"], ca_a],
            "Model B (BoneTrace)": [res_b["dy_attr"], res_b["dy_ctrl"], ca_b],
        },
        index=["Attributed Drop (dy_attr)", "Control Drop (dy_ctrl)", "Causal Advantage (CA)"],
    )
    st.bar_chart(df_chart, height=260)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: BENCHMARK & ABLATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def render_tab_benchmark(cfg: Dict[str, Any]) -> None:
    st.markdown("### 📊 Multi-Seed Audited Benchmark & Backbone Ablation")
    st.caption("Standardized evaluation across the 4 core clinical AI axes (Discrimination, Calibration, Localization, Faithfulness).")

    sub_t1, sub_t2, sub_t3, sub_t4, sub_t5 = st.tabs([
        "🧬 DenseNet-121 Ablation",
        "🎯 Primary Discrimination & Calibration",
        "📍 Spatial Localization Audit",
        "⚖️ FOV Causal Faithfulness",
        "🖼️ Publication Figures",
    ])

    with sub_t1:
        st.markdown("#### DenseNet-121 Ablation Study (Head-to-Head vs. ResNet-18)")
        densenet_table_path = TABLES_DIR / "T_ablation_densenet.md"
        densenet_json_path = EVAL_DIR / "densenet121_ablation_metrics.json"

        if densenet_table_path.is_file():
            md_content = load_markdown_table(densenet_table_path)
            st.markdown(md_content)
        else:
            st.info("DenseNet-121 table pending.")

        if densenet_json_path.is_file():
            with st.expander("📄 View Raw DenseNet-121 Metrics JSON", expanded=False):
                data = load_json_safe(densenet_json_path)
                st.json(data)

    with sub_t2:
        st.markdown("#### Classification Discrimination & Probability Calibration")
        t1_content = load_markdown_table(TABLES_DIR / "T1.md")
        if t1_content:
            st.markdown(t1_content)
        t5_content = load_markdown_table(TABLES_DIR / "T5.md")
        if t5_content:
            st.markdown(t5_content)

    with sub_t3:
        st.markdown("#### Spatial Localization Agreement (N=107 Fracture-Positives)")
        t4_content = load_markdown_table(TABLES_DIR / "T4.md")
        if t4_content:
            st.markdown(t4_content)

    with sub_t4:
        st.markdown("#### FOV-Constrained Causal Faithfulness & Area-Matched Lesion Dependence")
        t3_content = load_markdown_table(TABLES_DIR / "T3.md")
        if t3_content:
            st.markdown(t3_content)

    with sub_t5:
        st.markdown("#### Audited Publication Diagnostic Visualizations")
        f_col1, f_col2 = st.columns(2)
        p_roc = VIS_DIR / "publication_curves.png"
        p_cal = VIS_DIR / "calibration_diagrams.png"
        p_grid = VIS_DIR / "qualitative_4panel_grid.png"
        p_cm = VIS_DIR / "fig4_confusion_matrices_comparison.png"

        with f_col1:
            if p_roc.is_file():
                st.image(str(p_roc), caption="Figure 1: Multi-Seed ROC & PR Discrimination Curves", use_container_width=True)
            if p_cal.is_file():
                st.image(str(p_cal), caption="Figure 2: Reliability Diagrams & Calibration Curves", use_container_width=True)

        with f_col2:
            if p_cm.is_file():
                st.image(str(p_cm), caption="Figure 3: Confusion Matrices Comparison", use_container_width=True)
            if p_grid.is_file():
                st.image(str(p_grid), caption="Figure 4: 4-Panel Qualitative Saliency Grid", use_container_width=True)

    # ── Clinical Safety Banner ────────────────────────────────────────────────
    st.markdown(
        """
        <div class='safety-banner-clinical'>
            <div class='safety-banner-title'>⚠️ Clinical Safety & Diagnostic Guardrails Notice</div>
            <div class='safety-banner-body'>
                <strong>BoneTrace is an investigational research prototype for triage and worklist prioritization only (CADt).</strong><br/>
                On the audited held-out test benchmark (N = 609), model sensitivity is <strong>78.8%</strong> (ResNet-18 Region-Supervised) and <strong>74.1%</strong> (DenseNet-121). 
                Consequently, <strong>20% to 26% of fractures are not flagged at the default threshold</strong>.<br/>
                <strong>Explainability Guardrail:</strong> Post-hoc Grad-CAM heatmaps reflect coarse convolutional feature importance governed by a 32-pixel spatial resolution bottleneck. They do NOT outline anatomical fracture margins and must NEVER be used to delineate surgical boundaries or replace expert radiologist review.
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
