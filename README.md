# BoneTrace: Region Supervision vs. Post-Hoc Explainability Reliability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Tests: 98 Passed](https://img.shields.io/badge/tests-98%20passed-brightgreen.svg?style=flat-square)](tests/)
[![Streamlit: Cloud App](https://img.shields.io/badge/Streamlit-Live%20Workstation-ff4b4b.svg?style=flat-square)](https://bonetrace.streamlit.app)
[![Dataset: FracAtlas](https://img.shields.io/badge/Dataset-FracAtlas-lightgrey.svg?style=flat-square)](https://doi.org/10.1038/s41597-023-02432-4)

- **Source Code Repository:** [https://github.com/thyakeshwarsk2023/bonetrace](https://github.com/thyakeshwarsk2023/bonetrace)
- **Interactive Streamlit Web Application:** [https://bonetrace.streamlit.app](https://bonetrace.streamlit.app)

---

## Abstract and Overview

BoneTrace is a research framework and evaluation testbed investigating whether multi-task auxiliary region supervision (e.g., bounding-box and distance-transform localization) measurably improves the diagnostic reliability, probability calibration, and post-hoc explanation fidelity (Grad-CAM) of convolutional networks on musculoskeletal radiographs from the FracAtlas dataset.

In conventional deep learning pipelines for medical imaging, post-hoc attribution methods such as Grad-CAM are often assumed to reflect true pathological indicators. BoneTrace benchmarks this hypothesis against rigorous quantitative controls:
1. **Classification Performance:** Dual-head architectures (global classification head with an auxiliary spatial localization head) evaluated against single-head baselines.
2. **Probability Calibration:** Reliability diagrams, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and validation temperature scaling ($T$).
3. **Spatial Grounding:** Pointing Game hit rates, Energy-in-Box, and Median Peak Displacement compared against statistical null distributions (Uniform Field of View and Central Gaussian priors).
4. **Causal Faithfulness:** Causal Advantage ($CA = \Delta y_{\text{attr}} - \Delta y_{\text{ctrl}}$) evaluated under strict Field-of-View (FOV) masking guardrails to eliminate background attribution artifacts.

```text
                  +----------------------------------------+
                  |       Input Radiograph (512x512)       |
                  +-------------------+--------------------+
                                      |
                   +------------------+------------------+
                   |    Backbone Feature Extractor       |
                   |    (ResNet-18 / DenseNet-121)       |
                   +---------+--------------------+------+
                             |                    |
                             v                    v
                  +--------------------+ +--------------------+
                  |Classification Head | |Auxiliary Head (FPN)|
                  | (Global Pool + FC) | |(Multi-scale Decoder|
                  +----------+---------+ +----------+---------+
                             |                      |
                             v                      v
                      Fracture Logit        Auxiliary Heatmap
                      (BCEWithLogits)        (Dice + Focal)
```

---

## Web Application and Interactive PACS Workstation

A live interactive PACS-style workstation is deployed on Streamlit Community Cloud:

- **Live URL:** [https://bonetrace.streamlit.app](https://bonetrace.streamlit.app)

### Workstation Features
- **Diagnostic Inspection:** Upload planar X-rays or choose curated benchmark test cases from the FracAtlas cohort.
- **Model Triage Comparison:** Inspect baseline vs. region-supervised predictions side by side with calibrated posterior probabilities.
- **Explainability Overlays:** Toggle Grad-CAM and HiResCAM visual heatmaps with adjustable alpha blending against ground-truth fracture fissures.
- **Causal Faithfulness Audit:** Run 1-click in-browser causal perturbation tests comparing attributed-pixel suppression against Field-of-View controls.

To run the workstation locally:
```bash
streamlit run app.py
```

---

## Experimental Benchmark Results

All metrics below are computed on the frozen, audited test partition ($N=609$ images; 107 fracture-positive, 502 negative) with patient-level group isolation (`StratifiedGroupKFold`) to eliminate cross-split data leakage.

### 1. Classification and Calibration Benchmarks

| Architecture | Seed | ROC-AUC | PR-AUC | Sensitivity | Specificity | False Positives | ECE (10-bin) | Brier Score | Optimal Temp ($T$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-18 Baseline | 42 | 0.8901 | 0.7051 | 0.6355 | 0.9363 | 32 | 0.0428 | 0.0926 | 1.1599 |
| ResNet-18 Baseline | 123 | 0.8409 | 0.5990 | 0.6075 | 0.8645 | 68 | 0.1056 | 0.1240 | 0.8594 |
| ResNet-18 Baseline | 2026 | 0.8901 | 0.7051 | 0.6355 | 0.9363 | 32 | 0.0428 | 0.0926 | 1.1599 |
| **ResNet-18 Baseline (Mean +/- SD)** | -- | **0.8737 +/- 0.0284** | **0.6697 +/- 0.0613** | **0.6262 +/- 0.0162** | **0.9124 +/- 0.0415** | **44.0 +/- 20.8** | **0.0637 +/- 0.0363** | **0.1031 +/- 0.0181** | -- |
| ResNet-18 Region-Supervised | 42 | 0.9589 | 0.9109 | 0.7757 | 0.9880 | 6 | 0.0350 | 0.0432 | 1.4339 |
| ResNet-18 Region-Supervised | 123 | 0.9439 | 0.9046 | 0.8131 | 0.9761 | 12 | 0.0307 | 0.0420 | 1.3631 |
| ResNet-18 Region-Supervised | 2026 | 0.9589 | 0.9109 | 0.7757 | 0.9880 | 6 | 0.0350 | 0.0432 | 1.4339 |
| **ResNet-18 Region-Supervised (Mean +/- SD)** | -- | **0.9539 +/- 0.0087** | **0.9088 +/- 0.0036** | **0.7882 +/- 0.0216** | **0.9841 +/- 0.0069** | **8.0 +/- 3.5** | **0.0336 +/- 0.0025** | **0.0428 +/- 0.0007** | -- |

Region supervision achieves a **+0.0802 ROC-AUC improvement** and reduces mean false positives from **44.0 to 8.0 (an 81.8% reduction)** while cutting Expected Calibration Error by 47.3%.

### 2. DenseNet-121 Architecture Ablation

| Architecture Variant | Seed | ROC-AUC | PR-AUC | Sensitivity | Specificity | False Positives | ECE (10-bin) | Brier Score | Optimal Temp ($T$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet-121 Baseline | 42 | 0.8532 | 0.6869 | 0.6262 | 0.9044 | 48 | 0.0839 | 0.1073 | 1.3591 |
| DenseNet-121 Baseline | 123 | 0.8420 | 0.6659 | 0.6729 | 0.8725 | 64 | 0.1118 | 0.1266 | 1.4158 |
| DenseNet-121 Baseline | 2026 | 0.8643 | 0.6367 | 0.8879 | 0.7032 | 149 | 0.2710 | 0.1980 | 1.9854 |
| DenseNet-121 Region-Supervised | 42 | 0.8412 | 0.6165 | 0.5981 | 0.9024 | 49 | 0.0826 | 0.1147 | 1.0884 |
| DenseNet-121 Region-Supervised | 123 | 0.8617 | 0.6798 | 0.7290 | 0.8645 | 68 | 0.1219 | 0.1264 | 1.5110 |
| DenseNet-121 Region-Supervised | 2026 | 0.8702 | 0.6499 | 0.8972 | 0.6853 | 158 | 0.2797 | 0.2032 | 2.1541 |

### 3. Spatial Localization and Causal Faithfulness Metrics

| Metric Category | Metric Name | ResNet-18 Baseline | ResNet-18 Region-Supervised | Statistical Null Reference |
| :--- | :--- | :---: | :---: | :---: |
| **Spatial Localization** | Pointing Game Hit Rate | 13.08% | **43.93%** | 1.87% (Uniform FOV) / 1.06% (Gaussian Prior) |
| **Spatial Localization** | Energy-in-Box Ratio | 0.1873 | **0.3842** | ~0.081 (Random baseline) |
| **Causal Faithfulness** | Drop on Attr Masking ($\Delta y_{\text{attr}}$) | +0.4812 | **+1.4169** | -- |
| **Causal Faithfulness** | Drop on FOV Ctrl Masking ($\Delta y_{\text{ctrl}}$) | +0.1205 | **-0.5358** | Non-pathological bone/tissue control |
| **Causal Faithfulness** | **Causal Advantage ($CA = \Delta_{\text{attr}} - \Delta_{\text{ctrl}}$)** | +0.3607 | **+1.9528** | Strict FOV guardrail verified |

---

## Architectural Extensions

### 1. Multi-Scale Feature Pyramid Decoder (FPN)
Standard classification backbones yield low-resolution feature maps at stride 32 ($16 \times 16$). For fine hairline fissures that occupy only 2 to 5 pixels in original resolution, standard bilinear upsampling incurs substantial blur.

BoneTrace includes an FPN Decoder (`src/bonetrace/models/factory.py`) fusing multi-scale feature representations:
- **Layer 2:** Stride 8 ($64 \times 64$)
- **Layer 3:** Stride 16 ($32 \times 32$)
- **Layer 4:** Stride 32 ($16 \times 16$)

Lateral $1 \times 1$ convolutions and top-down additions generate high-resolution auxiliary localization predictions without compromising the global receptive field.

### 2. Normalized Gaussian Distance Transforms
Rectangular bounding-box supervision inadvertently penalizes models when focusing strictly on narrow fissures. BoneTrace provides continuous spatial distance transforms (`src/bonetrace/data/refinement.py`):
$$\mathcal{M}(x, y) = \exp\left( -\frac{\min_{(x_0, y_0) \in \mathcal{B}} \|(x,y) - (x_0, y_0)\|^2}{2\sigma^2} \right)$$
This replaces rigid step-function binary masks with distance-weighted gaussian distributions centered over true fracture paths.

---

## Repository Structure

```text
bonetrace/
├── app.py                               # Top-level Streamlit deployment entry point
├── configs/
│   └── default.yaml                     # Reproducible training and evaluation configuration
├── data/
│   ├── raw/                             # Original FracAtlas radiograph repository
│   └── processed/                       # Content-addressed manifests and frozen splits
├── outputs/
│   ├── checkpoints/                     # Trained weights and cryptographic manifests
│   ├── evaluation/                      # Verified evaluation JSONs and calibration results
│   ├── results_registry/                # Per-image prediction parquets (N=609 test set)
│   └── tables/                          # Markdown and LaTeX publication tables
├── scripts/
│   ├── audit.py                         # Dataset integrity and annotation reconciler
│   ├── preprocess.py                    # Aspect-preserving letterbox pipeline & group splitting
│   ├── train.py                         # Training pipeline with AMP and cosine annealing
│   ├── evaluate.py                      # Multi-metric evaluation and XAI benchmark runner
│   ├── run_predictions_and_metrics.py   # Registry inference runner
│   └── generate_figures.py              # Publication-grade figure generation
├── src/
│   └── bonetrace/
│       ├── data/                        # Dataset loaders, audit logic, distance transforms
│       ├── models/                      # Dual-head architectures (ResNet-18, DenseNet-121, FPN)
│       ├── training/                    # Mixed-precision trainer and loss functions
│       ├── explainability/              # Grad-CAM extraction and FOV guardrails
│       ├── evaluation/                  # Pointing game, ECE, Brier, and Causal Advantage
│       └── ui/                          # Streamlit Diagnostic Explorer modules
├── tests/                               # 98 automated unit and integration tests
├── pyproject.toml                       # Build and packaging configuration
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation
```

---

## Installation and Execution

### System Requirements
- Python 3.10 or higher
- PyTorch 2.0 or higher
- Windows PowerShell or POSIX-compliant Bash shell

### 1. Clone the Repository
```bash
git clone https://github.com/thyakeshwarsk2023/bonetrace.git
cd bonetrace
```

### 2. Environment Setup

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Dependency Installation
```bash
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Run Test Suite
Execute the 98 automated unit and integration tests:
```bash
pytest tests/ -v
```

### 5. Execute Pipeline Stages
```bash
# Phase 1: Dataset audit and annotation reconciliation
python scripts/audit.py

# Phase 2: Aspect-preserving letterbox preprocessing and split generation
python scripts/preprocess.py

# Phase 3: Train baseline or region-supervised architectures
python scripts/train.py --model-type region_supervised --backbone resnet18 --seed 42

# Phase 4: Run evaluation registry and benchmark metrics
python scripts/run_predictions_and_metrics.py

# Phase 5: Launch local Streamlit application
streamlit run app.py
```

---

## Methodological and Clinical Limitations

1. **Planar 2D Projection Constraints:** Planar X-rays collapse 3D anatomical structures into a single projection, causing overlapping densities from cortices, trabeculae, and external hardware. Saliency maps identify planar regions of interest but cannot replace 3D cross-sectional tomography (CT/MRI) for displacement assessment.
2. **Coarse Bounding-Box Annotations:** Public annotations in FracAtlas provide rectangular bounding boxes rather than segmented fracture fissures. While distance transforms and multi-scale FPN decoders attenuate background penalty, fine micro-fracture localization remains bounded by annotation resolution.
3. **Attribution Map Artifacts:** Post-hoc attribution methods like Grad-CAM can highlight clinically irrelevant structures (contrast markers, soft-tissue borders) while still achieving reasonable classification confidence. Quantitative causal audits ($CA$) are essential to confirm whether model decisions are genuinely grounded in pathology.
4. **Single-Center Cohort Distribution:** FracAtlas is collected from a single clinical institution. Multi-center deployment requires external validation across disparate radiographic equipment, exposure parameters, and demographics.

---

## Citation and References

If you utilize this codebase or benchmark in your research, please cite:

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
