"""
scripts/aggregate_multiseed_results.py

Aggregates test performance across multi-seed runs.
Computes Mean ± Std Dev for:
- ROC-AUC
- PR-AUC
- Specificity
- Causal Faithfulness Advantage
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

from bonetrace.config import load_config
from bonetrace.data.dataset import FracAtlasDataset
from bonetrace.models.factory import build_model
from bonetrace.explainability.gradcam import GradCAM


def extract_logit(out) -> float:
    """Safely extracts a scalar logit whether model returns dict, tuple, or tensor."""
    if isinstance(out, dict):
        for k in ["logits", "cls_logits", "classification_logits", "pred"]:
            if k in out:
                val = out[k]
                return float(val.squeeze().item() if val.numel() == 1 else val[0].item())
        val = next(iter(out.values()))
        return float(val.squeeze().item() if val.numel() == 1 else val[0].item())
    elif isinstance(out, (tuple, list)):
        val = out[0]
        return float(val.squeeze().item() if hasattr(val, "numel") and val.numel() == 1 else val[0].item())
    return float(out.squeeze().item() if out.numel() == 1 else out[0].item())


def get_checkpoint_path(model_type: str, seed: int) -> Path:
    candidates = [
        Path(f"outputs/checkpoints/{model_type}_seed{seed}.pt"),
        Path(f"outputs/checkpoints/{model_type}_seed_{seed}.pt"),
        Path(f"outputs/checkpoints/{model_type}_best.pt"),
        Path(f"outputs/checkpoints/{model_type}/best_model.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Checkpoint for {model_type} (seed {seed}) not found among: {candidates}")


def evaluate_seed_model(model_type: str, seed: int, dataset, device: torch.device):
    cfg = load_config()
    model = build_model(config=cfg, model_type=model_type)
    ckpt_path = get_checkpoint_path(model_type, seed)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            img = sample["image"].unsqueeze(0).to(device)
            target = int(sample["label"].item())
            out = model(img)
            logit = extract_logit(out)
            prob = torch.sigmoid(torch.tensor(logit)).item()

            all_targets.append(target)
            all_probs.append(prob)

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = float(tn / (tn + fp))

    # Causal faithfulness advantage (positive test cohort)
    gcam = GradCAM(model)
    pos_indices = [i for i, r in enumerate(dataset.records) if int(r.get("fractured_label", 0)) == 1]

    delta_attrs = []
    delta_ctrls = []
    for idx in pos_indices:
        s = dataset[idx]
        img_t = s["image"].to(device)
        orig_out = model(img_t.unsqueeze(0))
        orig_val = extract_logit(orig_out)

        cam = gcam.generate(img_t)
        pad_mask = (img_t[0] < -2.0).cpu().numpy()
        fov_mask = ~pad_mask
        n_fov = int(fov_mask.sum())
        n_mask = max(1, int(round(n_fov * 0.10)))

        # Mask top 10%
        cam_attr = cam.copy()
        cam_attr[pad_mask] = -np.inf
        top_idx = np.argsort(cam_attr.flatten())[-n_mask:]
        img_attr = img_t.clone()
        for ch in range(3):
            flat = img_attr[ch].flatten()
            flat[top_idx] = 0.0
            img_attr[ch] = flat.reshape(512, 512)

        # Mask bottom 10% inside FOV
        cam_ctrl = cam.copy()
        cam_ctrl[pad_mask] = np.inf
        ctrl_idx = np.argsort(cam_ctrl.flatten())[:n_mask]
        img_ctrl = img_t.clone()
        for ch in range(3):
            flat = img_ctrl[ch].flatten()
            flat[ctrl_idx] = 0.0
            img_ctrl[ch] = flat.reshape(512, 512)

        with torch.no_grad():
            l_attr = extract_logit(model(img_attr.unsqueeze(0)))
            l_ctrl = extract_logit(model(img_ctrl.unsqueeze(0)))

        delta_attrs.append(orig_val - l_attr)
        delta_ctrls.append(orig_val - l_ctrl)

    advantage = float(np.mean(delta_attrs) - np.mean(delta_ctrls))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "specificity": specificity,
        "causal_advantage": advantage,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    test_csv = cfg.paths.processed_data_dir / "test.csv"
    dataset = FracAtlasDataset(manifest=test_csv, split="test", exclude_conflicts=True)

    # Automatically identify which seeds have checkpoints
    target_seeds = [42, 123, 2026]
    available_seeds = []
    for s in target_seeds:
        try:
            get_checkpoint_path("baseline", s)
            get_checkpoint_path("region_supervised", s)
            available_seeds.append(s)
        except FileNotFoundError:
            pass

    print(f"Discovered checkpoints for seeds: {available_seeds} (Target: {target_seeds})")
    if not available_seeds:
        print("No paired checkpoints found. Please train models first.")
        return

    results = {"baseline": [], "region_supervised": []}
    for seed in available_seeds:
        print(f"\n--- Evaluating Seed {seed} ---")
        res_b = evaluate_seed_model("baseline", seed, dataset, device)
        res_r = evaluate_seed_model("region_supervised", seed, dataset, device)
        results["baseline"].append(res_b)
        results["region_supervised"].append(res_r)
        print(f"  Baseline (Seed {seed}): ROC-AUC = {res_b['roc_auc']:.4f} | Causal Adv = {res_b['causal_advantage']:.4f}")
        print(f"  Region-Sup (Seed {seed}): ROC-AUC = {res_r['roc_auc']:.4f} | Causal Adv = {res_r['causal_advantage']:.4f}")

    # Aggregate Mean and Std
    summary = {}
    for model_type in ["baseline", "region_supervised"]:
        summary[model_type] = {}
        for metric in ["roc_auc", "pr_auc", "specificity", "causal_advantage"]:
            vals = [r[metric] for r in results[model_type]]
            summary[model_type][metric] = (float(np.mean(vals)), float(np.std(vals)))

    report_md = f"""# BoneTrace: Multi-Seed Robustness Evaluation (N = {len(available_seeds)} Runs)

**Seeds Evaluated:** `{available_seeds}`  
**Held-out Test Cohort:** $N = 610$ ($100$ fracture-positive, $510$ negative controls)

---

### Comparative Summary (Mean ± Std Dev)

| Metric | Baseline (Model A) | Region-Supervised (Model B) | $\\Delta$ Shift |
| :--- | :---: | :---: | :---: |
| **Test ROC-AUC** | {summary['baseline']['roc_auc'][0]:.4f} ± {summary['baseline']['roc_auc'][1]:.4f} | **{summary['region_supervised']['roc_auc'][0]:.4f} ± {summary['region_supervised']['roc_auc'][1]:.4f}** | +{summary['region_supervised']['roc_auc'][0] - summary['baseline']['roc_auc'][0]:.4f} |
| **Test PR-AUC** | {summary['baseline']['pr_auc'][0]:.4f} ± {summary['baseline']['pr_auc'][1]:.4f} | **{summary['region_supervised']['pr_auc'][0]:.4f} ± {summary['region_supervised']['pr_auc'][1]:.4f}** | +{summary['region_supervised']['pr_auc'][0] - summary['baseline']['pr_auc'][0]:.4f} |
| **Specificity** | {summary['baseline']['specificity'][0]*100:.2f}% ± {summary['baseline']['specificity'][1]*100:.2f}% | **{summary['region_supervised']['specificity'][0]*100:.2f}% ± {summary['region_supervised']['specificity'][1]*100:.2f}%** | +{(summary['region_supervised']['specificity'][0] - summary['baseline']['specificity'][0])*100:.2f}% |
| **Causal Advantage** | {summary['baseline']['causal_advantage'][0]:.4f} ± {summary['baseline']['causal_advantage'][1]:.4f} | **{summary['region_supervised']['causal_advantage'][0]:.4f} ± {summary['region_supervised']['causal_advantage'][1]:.4f}** | +{summary['region_supervised']['causal_advantage'][0] - summary['baseline']['causal_advantage'][0]:.4f} |
"""
    out_path = Path("outputs/evaluation/MULTISEED_REPORT.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")
    print(f"\nReport saved to: {out_path}\n")
    print(report_md)


if __name__ == "__main__":
    main()