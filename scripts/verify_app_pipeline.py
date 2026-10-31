"""
scripts/verify_app_pipeline.py

Fast, automated verification suite for BoneTrace UI pipeline and Streamlit helpers.
Tests:
  1. Curated Benchmark Case with GT box (Inference, 4 panels, Pointing Game, Perturbation)
  2. Arbitrary External Images (odd-dimension grayscale & RGBA, letterbox, FOV, CAM, HiResCAM, Aux Head, Perturbation)
  3. Dynamic Checkpoint Discovery & Harmonized Model Loading (DenseNet-121 & ResNet-18)
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bonetrace.ui.helpers import (
    discover_available_model_pairs,
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
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_1_curated_benchmark_case() -> bool:
    print("\n--- Test 1: Curated Benchmark Case (with GT Box) ---")
    manifest_p = REPO_ROOT / "data" / "processed" / "test.csv"
    if not manifest_p.is_file():
        manifest_p = REPO_ROOT / "data" / "frozen" / "test.csv"
    df = pd.read_csv(manifest_p)

    sub = df[df["canonical_id"] == "CANON_0350"]
    if sub.empty:
        sub = df[df["fractured_label"] == 1]
    row = sub.iloc[0]
    img_path = Path(row["resolved_primary_path"])
    boxes = parse_boxes(row.get("letterbox_target_boxes"))

    print(f"Loading case: {row['canonical_id']} (Path: {img_path.name}, Boxes: {len(boxes)})")
    disp_rgb, img_t = load_and_preprocess(img_path)
    assert disp_rgb.shape == (512, 512, 3), f"Wrong disp shape: {disp_rgb.shape}"
    assert img_t.shape == (3, 512, 512), f"Wrong tensor shape: {img_t.shape}"

    fov_mask = get_fov_mask(img_t)
    assert fov_mask.shape == (512, 512), f"Wrong FOV shape: {fov_mask.shape}"
    assert fov_mask.any(), "FOV mask is completely empty"

    pairs = discover_available_model_pairs()
    pair = pairs[0]
    bb, s = pair["backbone"], pair["seed"]
    print(f"Testing with model pair: {pair['label']}")

    model_a = load_model_for_ui(bb, "baseline", s, DEVICE)
    model_b = load_model_for_ui(bb, "region", s, DEVICE)

    prob_a, logit_a = run_inference(model_a, img_t, DEVICE)
    prob_b, logit_b = run_inference(model_b, img_t, DEVICE)
    print(f"Inference: Model A p={prob_a:.3f} (logit={logit_a:+.2f}) | Model B p={prob_b:.3f} (logit={logit_b:+.2f})")

    cam_a = generate_attribution_map(model_a, img_t, "final", fov_mask, DEVICE)
    cam_b = generate_attribution_map(model_b, img_t, "final", fov_mask, DEVICE)
    aux_b = generate_aux_head_map(model_b, img_t, fov_mask, DEVICE)

    assert cam_a.shape == (512, 512), f"CAM A shape mismatch: {cam_a.shape}"
    assert cam_b.shape == (512, 512), f"CAM B shape mismatch: {cam_b.shape}"
    assert aux_b is not None and aux_b.shape == (512, 512), "Aux head map failed"

    # Pointing Game
    hit_a, energy_a = compute_pointing_game(cam_a, boxes, fov_mask)
    hit_b, energy_b = compute_pointing_game(cam_b, boxes, fov_mask)
    print(f"Pointing Game: Model A Hit={hit_a} (Energy={energy_a:.1f}%) | Model B Hit={hit_b} (Energy={energy_b:.1f}%)")

    # Render overlays for all 4 panels
    img1 = draw_fov_contour(draw_boxes_on_image(disp_rgb, boxes), fov_mask)
    img2 = add_peak_marker(overlay_heatmap(disp_rgb, cam_a, alpha=0.45, fov_mask=fov_mask), cam_a, fov_mask)
    img3 = add_peak_marker(overlay_heatmap(disp_rgb, cam_b, alpha=0.45, fov_mask=fov_mask), cam_b, fov_mask)
    img4 = overlay_heatmap(disp_rgb, aux_b, alpha=0.45, colormap="magma", fov_mask=fov_mask)
    assert all(img.shape == (512, 512, 3) for img in [img1, img2, img3, img4]), "Visual grid shape mismatch"

    # Perturbation
    res_b = run_perturbation(
        model=model_b,
        img_tensor=img_t,
        cam=cam_b,
        k_pct=10.0,
        fill_strategy="zero",
        mode="top_k_vs_random",
        boxes=boxes,
        fov_mask=fov_mask,
        device=DEVICE,
    )
    print(f"Causal Perturbation: dy_attr={res_b['dy_attr']:+.3f}, dy_ctrl={res_b['dy_ctrl']:+.3f}, CA={res_b['causal_advantage']:+.3f}")
    assert "causal_advantage" in res_b, "Perturbation result missing causal advantage"
    return True


def test_2_arbitrary_external_images() -> bool:
    print("\n--- Test 2: Arbitrary Non-Square External RGB/Grayscale Images ---")
    pairs = discover_available_model_pairs()
    pair = pairs[0]
    model_b = load_model_for_ui(pair["backbone"], "region", pair["seed"], DEVICE)

    # Case A: Odd-sized 16-bit medical grayscale (420 x 780)
    print("Testing Case A: 16-bit Grayscale NumPy array (420 x 780)...")
    raw16 = (np.random.rand(420, 780) * 65535).astype(np.uint16)
    disp1, t1 = load_and_preprocess(raw16)
    assert disp1.shape == (512, 512, 3), f"Case A display shape mismatch: {disp1.shape}"
    assert t1.shape == (3, 512, 512), f"Case A tensor shape mismatch: {t1.shape}"
    fov1 = get_fov_mask(t1)
    assert not fov1.all(), "FOV mask should have detected letterbox padding margins"

    prob1, logit1 = run_inference(model_b, t1, DEVICE)
    cam1 = generate_attribution_map(model_b, t1, "final", fov1, DEVICE)
    hires1 = generate_attribution_map(model_b, t1, "multiscale", fov1, DEVICE)
    aux1 = generate_aux_head_map(model_b, t1, fov1, DEVICE)
    unsup1 = compute_unsupervised_metrics(cam1, aux1, fov1)

    print(f"Case A results: Prob={prob1:.3f}, Grad-CAM Peak={unsup1['peak_cam_confidence']:.2f}, Aux Peak={unsup1['aux_peak_activation']:.3f}")
    assert cam1.shape == (512, 512) and hires1.shape == (512, 512), "Saliency map shape error"

    pert1 = run_perturbation(
        model=model_b,
        img_tensor=t1,
        cam=cam1,
        k_pct=10.0,
        fill_strategy="zero",
        mode="top_k_vs_random",
        boxes=[],
        fov_mask=fov1,
        device=DEVICE,
    )
    assert "causal_advantage" in pert1, "Perturbation on arbitrary image failed"

    # Case B: Odd-sized RGBA PIL Image (1024 x 600)
    print("Testing Case B: RGBA PIL Image (1024 x 600)...")
    rgba_arr = (np.random.rand(600, 1024, 4) * 255).astype(np.uint8)
    pil_rgba = Image.fromarray(rgba_arr, mode="RGBA")
    disp2, t2 = load_and_preprocess(pil_rgba)
    assert disp2.shape == (512, 512, 3), f"Case B display shape mismatch: {disp2.shape}"
    assert t2.shape == (3, 512, 512), f"Case B tensor shape mismatch: {t2.shape}"
    fov2 = get_fov_mask(t2)

    cam2 = generate_attribution_map(model_b, t2, "final", fov2, DEVICE)
    aux2 = generate_aux_head_map(model_b, t2, fov2, DEVICE)
    assert cam2.shape == (512, 512) and aux2.shape == (512, 512)
    print("Case B processed smoothly through full pipeline!")
    return True


def test_3_checkpoint_discovery_and_loading() -> bool:
    print("\n--- Test 3: All Discovered Checkpoints Discovery & Zero-Crash Loading ---")
    pairs = discover_available_model_pairs()
    print(f"Discovered {len(pairs)} model pairs:")
    assert len(pairs) >= 2, f"Expected at least 2 pairs, found {len(pairs)}"

    for p in pairs:
        bb, s = p["backbone"], p["seed"]
        print(f"Testing pair: {p['label']} ({bb}, seed={s})...")
        t0 = time.time()
        m_a = load_model_for_ui(bb, "baseline", s, DEVICE)
        m_b = load_model_for_ui(bb, "region", s, DEVICE)
        dt = time.time() - t0
        assert m_a is not None and m_b is not None
        print(f"  [OK] Model A ({m_a.__class__.__name__}) and Model B ({m_b.__class__.__name__}) loaded in {dt:.2f}s")
    return True


def main():
    print("================================================================================")
    print("BONETRACE XAI WORKSTATION PIPELINE VERIFICATION")
    print("================================================================================")
    t_start = time.time()

    pass_1 = False
    pass_2 = False
    pass_3 = False

    try:
        pass_1 = test_1_curated_benchmark_case()
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        pass_2 = test_2_arbitrary_external_images()
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        pass_3 = test_3_checkpoint_discovery_and_loading()
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        import traceback; traceback.print_exc()

    total_time = time.time() - t_start
    print("\n================================================================================")
    print("VERIFICATION SCORECARD")
    print("================================================================================")
    print(f"Test 1 (Curated Benchmark Case with GT Box):         [{'PASS' if pass_1 else 'FAIL'}]")
    print(f"Test 2 (Arbitrary Non-Square Grayscale/RGBA Inputs):  [{'PASS' if pass_2 else 'FAIL'}]")
    print(f"Test 3 (All Discovered Checkpoints Zero-Crash Load): [{'PASS' if pass_3 else 'FAIL'}]")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print("================================================================================")

    if pass_1 and pass_2 and pass_3:
        print("ALL TESTS PASSED SUCCESSFULLY! WORKSTATION READY.")
        sys.exit(0)
    else:
        print("ONE OR MORE TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
