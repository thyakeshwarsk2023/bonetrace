"""
scripts/generate_calibration_diagrams.py

Generates publication-quality Reliability Diagrams (Calibration Curves)
comparing Baseline (ECE = 0.226) vs Region-Supervised (ECE = 0.089)
on the held-out test cohort (N = 610).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from bonetrace.config import load_config
from bonetrace.data.dataset import FracAtlasDataset
from bonetrace.models.factory import build_model


def extract_logit(out) -> float:
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


def get_predictions(model, dataset, device):
    probs, targets = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            img = dataset[i]["image"].unsqueeze(0).to(device)
            target = int(dataset[i]["label"].item())
            out = model(img)
            logit = extract_logit(out)
            prob = torch.sigmoid(torch.tensor(logit)).item()
            probs.append(prob)
            targets.append(target)
    return np.array(targets), np.array(probs)


def compute_calibration(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high) if i < n_bins - 1 else (y_prob >= low) & (y_prob <= high)
        n_in_bin = np.sum(mask)

        if n_in_bin > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(n_in_bin)
            ece += (n_in_bin / total_samples) * np.abs(acc - conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((low + high) / 2.0)
            bin_counts.append(0)

    return bin_edges, np.array(bin_accs), np.array(bin_confs), np.array(bin_counts), ece


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    test_csv = cfg.paths.processed_data_dir / "test.csv"
    dataset = FracAtlasDataset(manifest=test_csv, split="test", exclude_conflicts=True)

    # 1. Load Baseline Checkpoint
    m_base = build_model(config=cfg, model_type="baseline")
    ckpt_b = torch.load("outputs/checkpoints/baseline/best_model.pt", map_location=device, weights_only=False)
    m_base.load_state_dict(ckpt_b["model_state_dict"] if "model_state_dict" in ckpt_b else ckpt_b)
    m_base.to(device).eval()

    # 2. Load Region-Supervised Checkpoint
    m_reg = build_model(config=cfg, model_type="region_supervised")
    ckpt_r = torch.load("outputs/checkpoints/region_supervised/best_model.pt", map_location=device, weights_only=False)
    m_reg.load_state_dict(ckpt_r["model_state_dict"] if "model_state_dict" in ckpt_r else ckpt_r)
    m_reg.to(device).eval()

    y_true, p_base = get_predictions(m_base, dataset, device)
    _, p_reg = get_predictions(m_reg, dataset, device)

    edges, acc_b, conf_b, counts_b, ece_b = compute_calibration(y_true, p_base, n_bins=10)
    _, acc_r, conf_r, counts_r, ece_r = compute_calibration(y_true, p_reg, n_bins=10)

    # Plot Settings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    width = 0.085

    # Panel 1: Baseline
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    ax1.bar(
        bin_centers,
        acc_b,
        width=width,
        color="#2b5c8f",
        alpha=0.75,
        edgecolor="black",
        label=f"Outputs (Acc)",
    )
    # Highlight Calibration Gap
    for i in range(len(bin_centers)):
        if counts_b[i] > 0 and acc_b[i] < conf_b[i]:
            ax1.bar(
                bin_centers[i],
                conf_b[i] - acc_b[i],
                bottom=acc_b[i],
                width=width,
                color="#e74c3c",
                alpha=0.4,
                edgecolor="#c0392b",
                linestyle=":",
                label="Calibration Gap" if i == 0 else "",
            )

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("Confidence (Predicted Probability)", fontsize=11)
    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.set_title(f"Baseline ResNet-18\nECE = {ece_b:.3f} (Severe Overconfidence)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Region-Supervised
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    ax2.bar(
        bin_centers,
        acc_r,
        width=width,
        color="#27ae60",
        alpha=0.75,
        edgecolor="black",
        label=f"Outputs (Acc)",
    )
    for i in range(len(bin_centers)):
        if counts_r[i] > 0 and acc_r[i] < conf_r[i]:
            ax2.bar(
                bin_centers[i],
                conf_r[i] - acc_r[i],
                bottom=acc_r[i],
                width=width,
                color="#e74c3c",
                alpha=0.4,
                edgecolor="#c0392b",
                linestyle=":",
                label="Calibration Gap" if i == 0 else "",
            )

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Confidence (Predicted Probability)", fontsize=11)
    ax2.set_ylabel("Empirical Accuracy", fontsize=11)
    ax2.set_title(f"Region-Supervised ResNet-18\nECE = {ece_r:.3f} (Well-Calibrated)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "calibration_diagrams.png", dpi=300)
    fig.savefig(out_dir / "calibration_diagrams.pdf")
    print(f"Calibration diagrams exported to {out_dir}/calibration_diagrams.png and .pdf")
    print(f"Baseline ECE: {ece_b:.4f} | Region-Supervised ECE: {ece_r:.4f}")


if __name__ == "__main__":
    main()