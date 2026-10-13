"""
scripts/generate_publication_curves.py

Generates vector-quality (PDF and 300 DPI PNG) ROC and PR curves
comparing Baseline vs Region-Supervised models.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from bonetrace.config import load_config
from bonetrace.data.dataset import FracAtlasDataset
from bonetrace.models.factory import build_model


def extract_logit(out) -> float:
    """Safely extracts a scalar logit whether model returns dict, tuple, or tensor."""
    if isinstance(out, dict):
        for k in ["logits", "cls_logits", "classification_logits", "pred"]:
            if k in out:
                v = out[k]
                return float(v.squeeze().item() if v.numel() == 1 else v[0].item())
        v = next(iter(out.values()))
        return float(v.squeeze().item() if v.numel() == 1 else v[0].item())
    elif isinstance(out, (tuple, list)):
        v = out[0]
        return float(v.squeeze().item() if hasattr(v, "numel") and v.numel() == 1 else v[0].item())
    return float(out.squeeze().item() if out.numel() == 1 else out[0].item())


def get_probabilities(model, dataset, device):
    probs, targets = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            img = dataset[i]["image"].unsqueeze(0).to(device)
            target = int(dataset[i]["label"].item())
            out = model(img)
            logit = extract_logit(out)
            probs.append(torch.sigmoid(torch.tensor(logit)).item())
            targets.append(target)
    return np.array(targets), np.array(probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    test_csv = cfg.paths.processed_data_dir / "test.csv"
    dataset = FracAtlasDataset(manifest=test_csv, split="test", exclude_conflicts=True)

    # Load baseline
    m_base = build_model(config=cfg, model_type="baseline")
    ckpt_b = torch.load("outputs/checkpoints/baseline/best_model.pt", map_location=device, weights_only=False)
    m_base.load_state_dict(ckpt_b["model_state_dict"] if "model_state_dict" in ckpt_b else ckpt_b)
    m_base.to(device).eval()

    # Load region-supervised
    m_reg = build_model(config=cfg, model_type="region_supervised")
    ckpt_r = torch.load("outputs/checkpoints/region_supervised/best_model.pt", map_location=device, weights_only=False)
    m_reg.load_state_dict(ckpt_r["model_state_dict"] if "model_state_dict" in ckpt_r else ckpt_r)
    m_reg.to(device).eval()

    y_true, p_base = get_probabilities(m_base, dataset, device)
    _, p_reg = get_probabilities(m_reg, dataset, device)

    # 1. ROC Curve
    fpr_b, tpr_b, _ = roc_curve(y_true, p_base)
    fpr_r, tpr_r, _ = roc_curve(y_true, p_reg)
    auc_b, auc_r = auc(fpr_b, tpr_b), auc(fpr_r, tpr_r)

    # 2. PR Curve
    prec_b, rec_b, _ = precision_recall_curve(y_true, p_base)
    prec_r, rec_r, _ = precision_recall_curve(y_true, p_reg)
    pr_auc_b, pr_auc_r = auc(rec_b, prec_b), auc(rec_r, prec_r)
    prevalence = np.mean(y_true)

    # Plot Setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # ROC Subplot
    ax1.plot(fpr_b, tpr_b, label=f"Baseline (AUC = {auc_b:.3f})", color="#2b5c8f", lw=2)
    ax1.plot(fpr_r, tpr_r, label=f"Region-Supervised (AUC = {auc_r:.3f})", color="#c23b22", lw=2.5)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # PR Subplot
    ax2.plot(rec_b, prec_b, label=f"Baseline (PR-AUC = {pr_auc_b:.3f})", color="#2b5c8f", lw=2)
    ax2.plot(rec_r, prec_r, label=f"Region-Supervised (PR-AUC = {pr_auc_r:.3f})", color="#c23b22", lw=2.5)
    ax2.axhline(prevalence, color="gray", linestyle="--", label=f"Prevalence ({prevalence*100:.1f}%)")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax2.set_ylabel("Precision (PPV)", fontsize=11)
    ax2.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "publication_curves.png", dpi=300)
    fig.savefig(out_dir / "publication_curves.pdf")
    print(f"Publication curves exported to {out_dir}/publication_curves.png and .pdf")


if __name__ == "__main__":
    main()