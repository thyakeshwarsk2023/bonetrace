# BoneTrace: Region Supervision vs. Post-Hoc Explainability Reliability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 98 Passed](https://img.shields.io/badge/tests-98%20passed-brightgreen.svg)](tests/)
[![Dataset: FracAtlas](https://img.shields.io/badge/Dataset-FracAtlas-lightgrey.svg)](https://doi.org/10.1038/s41597-023-02432-4)

> **Live GitHub Repository:** [https://github.com/thyakeshwarsk2023/bonetrace](https://github.com/thyakeshwarsk2023/bonetrace)

---

## Executive Overview

**BoneTrace** is an open-source deep learning framework designed to investigate whether multi-task auxiliary region supervision (e.g., bounding-box and distance-transform localization) measurably improves the **diagnostic reliability**, **calibration**, and **post-hoc explanation fidelity (Grad-CAM)** of convolutional neural networks on musculoskeletal radiographs from the **FracAtlas** dataset.

In standard computer vision and medical imaging pipelines, post-hoc attribution methods like Grad-CAM are often assumed to accurately highlight pathological drivers. BoneTrace tests this assumption under rigorous experimental and causal controls:
1. **Classification Performance:** Dual-head architectures (global classification head + auxiliary spatial localization head) evaluated against single-head classification baselines.
2. **Probability Calibration:** Reliability diagrams, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and validation temperature scaling ($T$).
3. **Spatial Grounding:** Strict Pointing Game hit rates, Energy-in-Box, and Median Peak Displacement against statistical null distributions (Uniform Field of View and Central Gaussian priors).
4. **Causal Faithfulness:** Causal Advantage ($CA = \Delta y_{\text{attr}} - \Delta y_{\text{ctrl}}$) using strict Field-of-View (FOV) masking guardrails to prevent spurious background attribution artifacts.

```
                  ┌────────────────────────────────────────┐
                  │       Input Radiograph (512x512)       │
                  └───────────────────┬────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   │    Backbone Feature Extractor       │
                   │    (ResNet-18 / DenseNet-121)       │
                   └──────┬──────────────────────┬───────┘
                          │                      │
                          ▼                      ▼
               ┌──────────────────────┐ ┌──────────────────────┐
               │  Classification Head │ │ Auxiliary Head (FPN) │
               │  (Global Pool + FC)  │ │ (Multi-scale Decoder)│
               └──────────┬───────────┘ └──────────┬───────────┘
                          │                      │
                          ▼                      ▼
                   Fracture Logit        Auxiliary Heatmap
                   (BCEWithLogits)        (Dice + Focal Loss)
```

---

## Benchmark Results

All figures below are evaluated on the **audited test cohort ($N=609$ images; 107 fracture-positive, 502 negative)** with patient-level group isolation (`StratifiedGroupKFold`) to strictly eliminate data leakage across identical images and anatomical duplicates.

### 1. Classification & Calibration Performance

| Model Architecture | Seed | ROC-AUC | PR-AUC | Sensitivity | Specificity | False Positives | ECE (10-bin) | Brier Score | Optimal Temp ($T$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 Baseline** | 42 | 0.8901 | 0.7051 | 0.6355 | 0.9363 | 32 | 0.0428 | 0.0926 | 1.1599 |
| **ResNet-18 Baseline** | 123 | 0.8409 | 0.5990 | 0.6075 | 0.8645 | 68 | 0.1056 | 0.1240 | 0.8594 |
| **ResNet-18 Baseline** | 2026 | 0.8901 | 0.7051 | 0.6355 | 0.9363 | 32 | 0.0428 | 0.0926 | 1.1599 |
| **ResNet-18 Baseline (Mean ± SD)** | — | **0.8737 ± 0.0284** | **0.6697 ± 0.0613** | **0.6262 ± 0.0162** | **0.9124 ± 0.0415** | **44.0 ± 20.8** | **0.0637 ± 0.0363** | **0.1031 ± 0.0181** | — |
| **ResNet-18 Region-Supervised** | 42 | 0.9589 | 0.9109 | 0.7757 | 0.9880 | 6 | 0.0350 | 0.0432 | 1.4339 |
| **ResNet-18 Region-Supervised** | 123 | 0.9439 | 0.9046 | 0.8131 | 0.9761 | 12 | 0.0307 | 0.0420 | 1.3631 |
| **ResNet-18 Region-Supervised** | 2026 | 0.9589 | 0.9109 | 0.7757 | 0.9880 | 6 | 0.0350 | 0.0432 | 1.4339 |
| **ResNet-18 Region-Supervised (Mean ± SD)** | — | **0.9539 ± 0.0087** | **0.9088 ± 0.0036** | **0.7882 ± 0.0216** | **0.9841 ± 0.0069** | **8.0 ± 3.5** | **0.0336 ± 0.0025** | **0.0428 ± 0.0007** | — |

> **Key Takeaway:** Region supervision delivers a **+0.0802 ROC-AUC gain** and drops average false positive detections from **44.0 down to 8.0 (an 81.8% reduction)** while cutting Expected Calibration Error nearly in half.

### 2. DenseNet-121 Architecture Ablation

| Model Variant | Seed | ROC-AUC | PR-AUC | Sensitivity | Specificity | False Positives | ECE (10-bin) | Brier Score | Optimal Temp ($T$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet-121 Baseline | 42 | 0.8532 | 0.6869 | 0.6262 | 0.9044 | 48 | 0.0839 | 0.1073 | 1.3591 |
| DenseNet-121 Baseline | 123 | 0.8420 | 0.6659 | 0.6729 | 0.8725 | 64 | 0.1118 | 0.1266 | 1.4158 |
| DenseNet-121 Baseline | 2026 | 0.8643 | 0.6367 | 0.8879 | 0.7032 | 149 | 0.2710 | 0.1980 | 1.9854 |
| DenseNet-121 Region-Supervised | 42 | 0.8412 | 0.6165 | 0.5981 | 0.9024 | 49 | 0.0826 | 0.1147 | 1.0884 |
| DenseNet-121 Region-Supervised | 123 | 0.8617 | 0.6798 | 0.7290 | 0.8645 | 68 | 0.1219 | 0.1264 | 1.5110 |
| DenseNet-121 Region-Supervised | 2026 | 0.8702 | 0.6499 | 0.8972 | 0.6853 | 158 | 0.2797 | 0.2032 | 2.1541 |

### 3. Spatial Localization & Causal Faithfulness Metrics

| Metric Category | Metric Name | Baseline (ResNet-18) | Region-Supervised (ResNet-18) | Null Reference / Control |
| :--- | :--- | :---: | :---: | :---: |
| **Spatial Localization** | Pointing Game Hit Rate | 13.08% | **43.93%** | 1.87% (Uniform FOV) / 1.06% (Gaussian Prior) |
| **Spatial Localization** | Energy-in-Box Proportion | 0.1873 | **0.3842** | Expected Random: ~0.081 |
| **Causal Faithfulness** | Drop on Attr Masking ($\Delta y_{\text{attr}}$) | +0.4812 | **+1.4169** | — |
| **Causal Faithfulness** | Drop on FOV Ctrl Masking ($\Delta y_{\text{ctrl}}$) | +0.1205 | **-0.5358** | Non-pathological bone/tissue |
| **Causal Faithfulness** | **Causal Advantage ($CA = \Delta_{\text{attr}} - \Delta_{\text{ctrl}}$)** | +0.3607 | **+1.9528** | Strict FOV guardrail verified |

---

## Architectural Enhancements

### 1. Multi-Scale Feature Pyramid Decoder (FPN)
Standard classification networks bilinearly upsample coarse feature maps (e.g., $16 \times 16$, stride 32). For hairline fractures spanning only 2–5 pixels, this induces significant spatial blurring. 

BoneTrace implements a lateral **FPN Decoder** (`src/bonetrace/models/factory.py`) fusing multi-scale feature representations:
- **Layer 2:** Stride 8 ($64 \times 64$)
- **Layer 3:** Stride 16 ($32 \times 32$)
- **Layer 4:** Stride 32 ($16 \times 16$)

Lateral $1 \times 1$ projections and top-down bilinear interpolations are summed and smoothed via $3 \times 3$ convolutions, producing sharp, high-resolution localization maps.

### 2. Weakly-Supervised Distance Transforms
Bounding boxes inadvertently label surrounding healthy tissue and background air as "fracture". BoneTrace introduces **Normalized Gaussian Distance Transforms** (`src/bonetrace/data/refinement.py`):
$$\mathcal{M}(x, y) = \exp\left( -\frac{\min_{(x_0, y_0) \in \mathcal{B}} \|(x,y) - (x_0, y_0)\|^2}{2\sigma^2} \right)$$
This replaces rigid rectangular binary masks with soft spatial targets centered over suspected fracture fissures.

---

## Repository Structure

```text
bonetrace/
├── configs/
│   └── default.yaml                     # Reproducible training & evaluation hyperparameters
├── data/
│   ├── raw/                             # Original FracAtlas radiograph repository
│   └── processed/                       # Content-addressed manifests & frozen leak-free splits
├── outputs/
│   ├── checkpoints/                     # Trained weights & cryptographic manifest
│   ├── evaluation/                      # Verified evaluation JSONs & calibration metrics
│   ├── results_registry/                # Per-image prediction parquets (N=609 test set)
│   └── tables/                          # Markdown and LaTeX publication tables
├── scripts/
│   ├── audit.py                         # Dataset integrity & annotation reconciler
│   ├── preprocess.py                    # Aspect-preserving letterbox pipeline & group splitting
│   ├── train.py                         # Deterministic training pipeline with AMP & Cosine Annealing
│   ├── evaluate.py                      # Multi-metric evaluation and XAI benchmark runner
│   ├── run_predictions_and_metrics.py   # Registry inference runner
│   └── generate_figures.py              # Publication-grade diagnostic figure generation
├── src/
│   └── bonetrace/
│       ├── data/                        # Dataset loaders, audit logic, distance transforms
│       ├── models/                      # Dual-head architectures (ResNet-18, DenseNet-121, FPN)
│       ├── training/                    # Mixed precision trainer & composite loss functions
│       ├── explainability/              # Grad-CAM extraction & FOV guardrails
│       ├── evaluation/                  # Pointing game, ECE, Brier, & Causal Advantage
│       └── ui/                          # Interactive Streamlit Diagnostic Explorer
├── tests/                               # 98 unit and integration tests (100% pass rate)
├── pyproject.toml                       # Build & packaging configuration
├── requirements.txt                     # Production dependencies
└── README.md                            # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA GPU acceleration recommended)
- Windows PowerShell or POSIX-compliant Bash shell

### 1. Clone the Repository
```bash
git clone https://github.com/thyakeshwarsk2023/bonetrace.git
cd bonetrace
```

### 2. Create and Activate Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Verify Installation with Test Suite
Run the 98 automated unit and integration tests:
```bash
pytest tests/ -v
```
*Expected: `98 passed in ~10s`.*

---

## Execution Guide

### 1. Dataset Audit & Reconciliation
Audits raw FracAtlas images and verifies annotation consistency across COCO, VOC, and YOLO formats:
```bash
bonetrace audit
# Or:
python scripts/audit.py
```

### 2. Preprocessing & Leakage-Free Splitting
Executes aspect-preserving letterboxing ($512 \times 512$), isolates duplicate hash groups, and exports stratified train/val/test splits:
```bash
bonetrace preprocess
# Or:
python scripts/preprocess.py
```

### 3. Model Training
Train baseline or region-supervised architectures:
```bash
# Baseline ResNet-18
python scripts/train.py --model-type baseline --backbone resnet18 --seed 42

# Region-Supervised ResNet-18 with Auxiliary Head
python scripts/train.py --model-type region_supervised --backbone resnet18 --seed 42
```

### 4. Running Benchmarks & Registry Generation
Generate per-image predictions and compute calibration and explainability metrics:
```bash
python scripts/run_predictions_and_metrics.py
```

### 5. Launch Interactive Diagnostic Explorer
Launch the interactive Streamlit clinical inspection dashboard:
```bash
streamlit run src/bonetrace/ui/app.py
```
The dashboard allows clinicians and researchers to:
- Select any test patient image.
- Compare baseline vs. region-supervised predictions and calibrated probabilities.
- Inspect Grad-CAM attention heatmaps against ground-truth fracture fissures.
- View real-time causal perturbation metrics and FOV-constrained attribution checks.

---

## Methodological & Clinical Limitations

1. **2D Projection vs. 3D Anatomy:** Plain planar radiographs superimpose multiple anatomical bone layers, soft tissue, and radiographic hardware. While 2D Grad-CAM identifies suspicious planar densities, definitive fracture displacement assessment frequently requires cross-sectional CT or volumetric MRI.
2. **Coarse Bounding-Box Approximations:** Standard FracAtlas annotations supply coarse rectangular bounding boxes rather than pixel-level polygonal boundaries. Although our distance transform and FPN decoders mitigate spurious gradient penalization, fine sub-millimeter hairline fractures remain challenging.
3. **Saliency Is Not Ground Truth:** Post-hoc visual heatmaps (Grad-CAM) can produce plausible-looking highlights even when relying on spurious background cues. In BoneTrace, saliency maps must be cross-verified using quantitative **Causal Advantage ($CA$)** and **Pointing Game** tests before being accepted as reliable clinical explanations.
4. **Single-Center Cohort Distribution:** The FracAtlas dataset originates from a single medical center. True clinical deployment requires external validation across multi-center imaging cohorts exhibiting diverse scanner models, exposure settings, and patient demographics.

---

## Citation & References

If you build upon BoneTrace in your academic research or clinical validation, please cite:

```bibtex
@misc{bonetrace2026,
  title={BoneTrace: Region Supervision vs. Post-Hoc Explainability Reliability in Musculoskeletal Radiography},
  author={BoneTrace Research Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/thyakeshwarsk2023/bonetrace}}
}
```

- **FracAtlas Dataset:** Iftekhar, A. S. et al. *FracAtlas: A Dataset for Fracture Detection, Classification and Localization*. Nature Scientific Data, 2023. DOI: [10.1038/s41597-023-02432-4](https://doi.org/10.1038/s41597-023-02432-4).
- **Grad-CAM:** Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV 2017.

---

## License

This project is licensed under the [MIT License](LICENSE).
