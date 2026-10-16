"""
scripts/run_task_d5_lesion_dependence.py

Implements Task D5:
1. Evaluates fracture-positive test images across seeds and models.
2. Reports logit drop for:
   (a) masking ground-truth box
   (b) masking area-matched random FOV region (100 draws per image)
   (c) top-10% attributed region excluding the box
3. Computes ratio (a)/(b) with 1,000 bootstrap resamples for 95% CIs.
4. Saves structured JSON to outputs/evaluation/lesion_dependence_d5_results.json.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from bonetrace.config import load_config
from bonetrace.data.dataset import FracAtlasDataset
from bonetrace.explainability.gradcam import GradCAM
from bonetrace.models.factory import build_model


def is_letterbox_padding(image_tensor: torch.Tensor) -> np.ndarray:
    return (image_tensor[0] < -2.0).cpu().numpy()


def get_checkpoint_path(model_type: str, seed: int) -> Path:
    p_seed = Path(f"outputs/checkpoints/{model_type}_seed{seed}.pt")
    p_best = Path(f"outputs/checkpoints/{model_type}/best_model.pt")
    if p_seed.exists():
        return p_seed
    elif p_best.exists():
        return p_best
    raise FileNotFoundError(f"No checkpoint found for {model_type} seed {seed}")


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boot_means = []
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    mean = float(np.mean(arr))
    return mean, low, high


def evaluate_lesion_dependence(model_type: str, seed: int, dataset: FracAtlasDataset, device: torch.device, n_draws: int = 100):
    cfg = load_config()
    model = build_model(config=cfg, model_type=model_type)
    ckpt_path = get_checkpoint_path(model_type, seed)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    gradcam = GradCAM(model, target_size=(512, 512))
    rng = np.random.default_rng(seed)

    # Filter to fracture positives with annotations
    pos_indices = [
        i for i in range(len(dataset))
        if dataset[i]["label"].item() == 1.0 and dataset[i]["has_annotation"].item() and len(dataset[i]["boxes"]) > 0
    ]

    print(f"\n--- {model_type} (seed {seed}) on {len(pos_indices)} positive annotated images ---")

    records = []

    for idx in tqdm(pos_indices, desc=f"{model_type}_seed{seed}", ncols=75):
        sample = dataset[idx]
        img_t = sample["image"].to(device)
        cid = sample["canonical_id"]
        boxes = sample["boxes"].cpu().numpy() if isinstance(sample["boxes"], torch.Tensor) else np.array(sample["boxes"])

        with torch.no_grad():
            out_orig = model(img_t.unsqueeze(0))
            l_orig = out_orig["logits"] if isinstance(out_orig, dict) else out_orig
            orig_logit = float(l_orig[0, 0].item())

        cam = gradcam.generate(img_t, class_idx=0)
        pad_mask = is_letterbox_padding(img_t)
        fov_mask = ~pad_mask
        n_fov = int(fov_mask.sum())
        fov_indices = np.where(fov_mask.flatten())[0]

        # Top 10% attribution mask inside FOV
        n_top10 = max(1, int(round(n_fov * 0.10)))
        cam_fov = cam.copy()
        cam_fov[pad_mask] = -np.inf
        top_attr_indices = np.argsort(-cam_fov.flatten())[:n_top10]
        mask_attr = np.zeros_like(cam, dtype=bool)
        mask_attr[np.unravel_index(top_attr_indices, cam.shape)] = True

        # (a) Mask GT Box Region
        m_gt_box = np.zeros_like(cam, dtype=bool)
        for b in boxes:
            if len(b) == 4:
                x1, y1, x2, y2 = [int(round(c)) for c in b]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(512, x2), min(512, y2)
                m_gt_box[y1:y2, x1:x2] = True

        # Restrict GT box to valid FOV
        m_gt_box = m_gt_box & fov_mask
        n_box_pixels = int(m_gt_box.sum())

        if n_box_pixels == 0:
            continue

        img_box = img_t.clone()
        img_box[:, torch.from_numpy(m_gt_box).to(device)] = 0.0
        with torch.no_grad():
            out_box = model(img_box.unsqueeze(0))
            l_box = out_box["logits"] if isinstance(out_box, dict) else out_box
            drop_box = orig_logit - float(l_box[0, 0].item())

        # (b) Mask area-matched random FOV region (100 draws)
        # Vectorized / batched execution across 100 draws for high performance
        rand_drops = []
        if len(fov_indices) >= n_box_pixels:
            # Batch 100 draws in chunks of 25
            chunk_size = 25
            n_chunks = n_draws // chunk_size
            for _ in range(n_chunks):
                batch_imgs = []
                for _ in range(chunk_size):
                    r_idx = rng.choice(fov_indices, size=n_box_pixels, replace=False)
                    m_r = np.zeros_like(cam, dtype=bool)
                    m_r[np.unravel_index(r_idx, cam.shape)] = True
                    i_r = img_t.clone()
                    i_r[:, torch.from_numpy(m_r).to(device)] = 0.0
                    batch_imgs.append(i_r)
                stacked = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    out_r = model(stacked)
                    logits_r = out_r["logits"] if isinstance(out_r, dict) else out_r
                    for r_logit in logits_r[:, 0].cpu().numpy():
                        rand_drops.append(orig_logit - float(r_logit))

        mean_drop_rand = float(np.mean(rand_drops)) if rand_drops else 0.0

        # (c) Mask top-10% attributed region EXCLUDING the box
        m_attr_minus_box = mask_attr & (~m_gt_box)
        img_minus = img_t.clone()
        img_minus[:, torch.from_numpy(m_attr_minus_box).to(device)] = 0.0
        with torch.no_grad():
            out_minus = model(img_minus.unsqueeze(0))
            l_minus = out_minus["logits"] if isinstance(out_minus, dict) else out_minus
            drop_attr_minus = orig_logit - float(l_minus[0, 0].item())

        ratio_a_b = float(drop_box / mean_drop_rand) if abs(mean_drop_rand) > 1e-6 else np.nan

        records.append({
            "canonical_id": cid,
            "orig_logit": orig_logit,
            "drop_gt_box": drop_box,
            "drop_rand_fov_mean": mean_drop_rand,
            "drop_attr_minus_box": drop_attr_minus,
            "ratio_a_over_b": ratio_a_b,
            "box_pixels": n_box_pixels,
            "fov_pixels": n_fov,
        })

    # Summary statistics with bootstrap CIs
    drops_box = [r["drop_gt_box"] for r in records]
    drops_rand = [r["drop_rand_fov_mean"] for r in records]
    drops_minus = [r["drop_attr_minus_box"] for r in records]
    valid_ratios = [r["ratio_a_over_b"] for r in records if not np.isnan(r["ratio_a_over_b"]) and not np.isinf(r["ratio_a_over_b"])]

    box_m, box_l, box_h = bootstrap_ci(drops_box)
    rand_m, rand_l, rand_h = bootstrap_ci(drops_rand)
    minus_m, minus_l, minus_h = bootstrap_ci(drops_minus)
    ratio_m, ratio_l, ratio_h = bootstrap_ci(valid_ratios)

    # Also compute aggregate cohort ratio: mean(a) / mean(b)
    cohort_ratio = float(box_m / rand_m) if abs(rand_m) > 1e-6 else 0.0

    summary = {
        "model_type": model_type,
        "seed": seed,
        "checkpoint_used": str(ckpt_path),
        "n_samples": len(records),
        "draws_per_sample": n_draws,
        "metric_a_drop_gt_box": {
            "mean": box_m,
            "ci_95": [box_l, box_h],
        },
        "metric_b_drop_rand_fov": {
            "mean": rand_m,
            "ci_95": [rand_l, rand_h],
        },
        "metric_c_drop_attr_minus_box": {
            "mean": minus_m,
            "ci_95": [minus_l, minus_h],
        },
        "ratio_a_over_b_individual": {
            "mean": ratio_m,
            "ci_95": [ratio_l, ratio_h],
        },
        "ratio_a_over_b_cohort": cohort_ratio,
        "records": records,
    }

    print(f"  (a) Mask Box:           {box_m:.4f} [95% CI: {box_l:.4f}, {box_h:.4f}]")
    print(f"  (b) Mask Random FOV:    {rand_m:.4f} [95% CI: {rand_l:.4f}, {rand_h:.4f}]")
    print(f"  (c) Top 10% Minus Box:  {minus_m:.4f} [95% CI: {minus_l:.4f}, {minus_h:.4f}]")
    print(f"  Ratio (a)/(b) Cohort:   {cohort_ratio:.4f}")
    print(f"  Ratio (a)/(b) Mean ind: {ratio_m:.4f} [95% CI: {ratio_l:.4f}, {ratio_h:.4f}]")

    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Lesion-Dependence Audit (Task D5) on Device: {device}")

    cfg = load_config()
    test_csv = cfg.paths.processed_data_dir / "test.csv"
    dataset = FracAtlasDataset(manifest=test_csv, split="test", exclude_conflicts=True)

    results = {}
    for mtype in ["baseline", "region_supervised"]:
        results[mtype] = {}
        for s in [42, 123, 2026]:
            summary = evaluate_lesion_dependence(
                model_type=mtype,
                seed=s,
                dataset=dataset,
                device=device,
                n_draws=100,
            )
            # Exclude full records from compact summary
            rec = summary.pop("records")
            results[mtype][str(s)] = summary

    out_file = Path("outputs/evaluation/lesion_dependence_d5_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nTask D5 Results saved to: {out_file}")


if __name__ == "__main__":
    main()
