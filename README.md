# BoneTrace — Region Supervision vs Explanation Reliability

## Overview

BoneTrace investigates whether region supervision (e.g., fracture bounding boxes and region-guided attention) improves the reliability and fidelity of post-hoc explainability methods (e.g., Grad-CAM) on musculoskeletal radiographs from the FracAtlas dataset.

> **Project Status: Phase 1, Phase 1.5, Phase 2 & Phase 2.5 Completed**  
> Phase 1 data auditing, Phase 1.5 reconciliation, Phase 2 data preparation, and Phase 2.5 training-readiness validation have been fully implemented, unit-tested (48 passed tests), and validated on the real FracAtlas dataset. Downstream model training, Grad-CAM extraction, and reliability benchmarks remain deferred to subsequent phases.

---

## Repository Structure

```text
bonetrace/
├── configs/
│   └── default.yaml             # Configurable paths, hyperparams, and stage settings
├── data/
│   ├── raw/                     # Untracked directory for raw FracAtlas data (.gitkeep)
│   └── processed/               # Untracked directory for processed tensors/manifests (.gitkeep)
├── outputs/                     # Untracked directory for logs, checkpoints, metrics (.gitkeep)
├── scripts/
│   ├── audit.py                 # Runner for dataset audit stage
│   ├── preprocess.py            # Runner for preprocessing stage
│   ├── train.py                 # Runner for model training stage
│   └── evaluate.py              # Runner for evaluation and XAI benchmark stage
├── src/
│   └── bonetrace/
│       ├── __init__.py          # Package initialization
│       ├── config.py            # YAML configuration loader & path resolution
│       ├── logging_utils.py     # Unified console and file logging setup
│       ├── cli.py               # Top-level CLI entry point (`bonetrace`)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── audit.py         # FracAtlas integrity audit placeholder
│       │   ├── dataset.py       # FracAtlas PyTorch Dataset placeholder
│       │   └── preprocess.py    # Preprocessing and split placeholder
│       ├── models/
│       │   ├── __init__.py
│       │   └── factory.py       # Baseline and region-supervised model factory
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py       # Training loop and optimization placeholder
│       ├── explainability/
│       │   ├── __init__.py
│       │   └── gradcam.py       # Grad-CAM heatmap extraction placeholder
│       └── evaluation/
│           ├── __init__.py
│           └── metrics.py       # Classification & XAI reliability metrics placeholder
├── tests/
│   ├── __init__.py
│   ├── test_cli.py              # Tests for CLI parser, subcommands, and exits
│   ├── test_config.py           # Tests for YAML config loading & path overrides
│   ├── test_logging.py          # Tests for console and file loggers
│   └── test_placeholders.py     # Tests verifying NotImplementedError guards
├── .gitignore                   # Excludes raw data, processed data, and outputs
├── pyproject.toml               # PEP 517/621 packaging metadata and CLI console script
├── requirements.txt             # Project dependencies (core and target ML stack)
├── requirements-dev.txt         # Development and testing dependencies
└── README.md                    # Setup and execution guide
```

---

## Windows PowerShell Setup Guide

All commands below are designed to be run in **Windows PowerShell**.

### 1. Navigate to Project Directory

```powershell
Set-Location "C:\Users\welcome\.gemini\antigravity-ide\scratch\bonetrace"
```

### 2. Create and Activate Virtual Environment

```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

*Note: If PowerShell execution policy restricts script execution, run:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

### 3. Install Package and Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install project in editable mode with development dependencies
python -m pip install -e .
python -m pip install -r requirements-dev.txt

# (Optional / Future ML dependencies):
# python -m pip install -r requirements.txt
```

---

## Verifying the Scaffold

### Run Infrastructure Tests

```powershell
pytest tests/ -v
```

All infrastructure tests should pass, confirming that:
- The configuration loader parses `configs/default.yaml` and resolves paths relative to the repository root.
- Environment variables (`BONETRACE_RAW_DATA_DIR`, `BONETRACE_OUTPUT_DIR`, `BONETRACE_LOG_LEVEL`) and CLI flags override default paths.
- The logging system properly formats and outputs both console logs and log files.
- The CLI parses all subcommands (`audit`, `preprocess`, `train`, `evaluate`).
- Research logic placeholders raise clear `NotImplementedError` exceptions until implemented.

---

## CLI Entry Points and Execution

Once installed in editable mode (`pip install -e .`), the `bonetrace` command is available. Alternatively, you can run stages using direct Python module execution or the helper scripts in `scripts/`.

### 1. Audit & Reconciliation Stage (Phase 1 & 1.5 Implemented)

Performs comprehensive, read-only scientific integrity audit and reconciliation of the FracAtlas dataset. Investigates published vs actual counts, normalizes and compares COCO, VOC, YOLO, and VGG annotations, resolves cross-folder duplicates, generates a content-addressed canonical manifest, and produces visual diagnostic overlays.

```powershell
# Via CLI entry point (runs Phase 1 audit + Phase 1.5 reconciliation)
bonetrace audit

# Or via Python module
python -m bonetrace.cli audit

# Or via helper runners
python scripts/audit.py
python scripts/reconcile.py

# With custom paths or overrides:
python -m bonetrace.cli audit --raw-data-dir data/raw --output-dir outputs
```

**Outputs Generated Under `outputs/stage1/`:**
- `audit_summary.json`: Complete machine-readable audit report with metadata, distributions, and finding IDs.
- `reconciliation_summary.json`: Detailed reconciliation report resolving the 717/922 vs 719/924 discrepancy and cross-format metrics.
- `canonical_manifest.json` & `canonical_manifest.csv`: Canonical content-addressed manifest (4,068 unique image contents) with `future_split_group_id` for leak-free splitting.
- `annotation_comparison.csv`: Bounding box and polygon IoU comparisons across COCO, VOC, and YOLO using actual image-header dimensions.
- `annotation_discrepancies.csv`: Catalog of metadata discrepancies (e.g. transposed width/height in 22 VOC XMLs and 1 COCO entry).
- `duplicate_images.csv`: Catalog of all 14 duplicate hash groups (31 total files).
- `class_distribution.csv`: Distribution metrics for fractures, anatomical regions, views, hardware, and multiscans.
- `AUDIT_REPORT.md`: Formatted Phase 1 scientific data audit report.
- `RECONCILIATION_REPORT.md`: Formatted Phase 1.5 reconciliation and annotation verification report.
- `annotation_review/`: Stratified sample of 14 visual annotation overlays (`*.png`) comparing COCO, VOC, YOLO, and VGG polygons.

### 2. Preprocess Stage (Implemented)

Generates canonical data preparation artifacts and leakage-aware splits without modifying the raw dataset:
1. Audits exact duplicates (14 groups) and perceptual near-duplicates (112 candidates, including 69 confirmed exact pixel clones).
2. Partitions data into deterministic 70/15/15 train/val/test splits (`StratifiedGroupKFold`) isolating duplicate groups to prevent cross-split leakage.
3. Implements aspect-ratio preserving letterbox scaling to 512x512 with neutral black padding, with identical transform application to COCO absolute bounding boxes and segmentation masks.
4. Generates split CSVs (`train.csv`, `val.csv`, `test.csv`), file inventory, canonical manifests, and comprehensive reports.

```powershell
bonetrace preprocess
# Or:
python scripts/preprocess.py
```

Generated Preprocessing & Readiness Artifacts:
- `data/processed/file_inventory.csv`: Physical file-level inventory of all 4,085 files.
- `data/processed/canonical_manifest.csv` & `.json`: Content-level manifest (4,068 unique contents) with annotations and group IDs.
- `data/processed/train.csv` (2,843 samples, 69.89%, 17.80% fractures)
- `data/processed/val.csv` (613 samples, 15.07%, 17.62% fractures)
- `data/processed/test.csv` (612 samples, 15.04%, 16.50% fractures)
- `data/processed/near_duplicate_review.csv`: Complete audit of all 112 candidate pairs with decisions and rationales.
- `data/processed/truncated_jpegs_inventory.csv`: Catalog of all 59 truncated JPEGs and their 100% verified decoding.
- `data/processed/frozen_splits_checksums.json`: Cryptographic SHA-256 hashes of all frozen manifests and splits.
- `data/processed/splits_summary.json`: Summary of split counts, distributions, and disjointness verification.
- `outputs/stage2/PREPROCESSING_REPORT.md` & `preprocessing_summary.json`: Preprocessing report and split comparison.
- `outputs/stage2/TRAINING_READINESS_REPORT.md` & `training_readiness.json`: Comprehensive Phase 2.5 training-readiness report.

### 3. Train Stage

Intended to train baseline or region-supervised bone fracture models.

```powershell
bonetrace train
# Or:
python scripts/train.py
```

### 4. Evaluate Stage

Intended to run classification metrics and evaluate Grad-CAM explanation reliability against ground-truth bounding boxes.

```powershell
bonetrace evaluate
# Or:
python scripts/evaluate.py
```

---

## Overriding Configuration and Paths

Default settings are defined in `configs/default.yaml`. You can override paths via:

### 1. CLI Arguments

```powershell
bonetrace audit --config configs/default.yaml --raw-data-dir D:\Data\FracAtlas\raw --output-dir D:\Data\FracAtlas\outputs --log-level DEBUG
```

### 2. Environment Variables

```powershell
$env:BONETRACE_CONFIG = "configs/default.yaml"
$env:BONETRACE_RAW_DATA_DIR = "data/raw"
$env:BONETRACE_OUTPUT_DIR = "outputs"
$env:BONETRACE_LOG_LEVEL = "INFO"
```

---

## Implementation Roadmap (Next Steps)

1. **Obtain and Inspect Raw FracAtlas Dataset**:
   Place raw dataset files in `data/raw/` (or specify an external folder via `--raw-data-dir`). Inspect the exact directory hierarchy, image file formats, annotation files (e.g. CSV/JSON/VOC/COCO), and fracture class distributions.
2. **Implement `bonetrace/data/audit.py`**:
   Write automated validation for file existence, image integrity, bounding box coordinates, and summary statistics.
3. **Implement Preprocessing & Splitting in `bonetrace/data/preprocess.py`**:
   Build patient-aware train/val/test splits, image resizing, and bounding box normalization.
4. **Implement Models & Training (`bonetrace/models/` and `bonetrace/training/`)**:
   Build baseline classifier and region-supervised model variant (e.g., attention-guided or multi-task loss).
5. **Implement Explainability & Evaluation (`bonetrace/explainability/` and `bonetrace/evaluation/`)**:
   Generate Grad-CAM heatmaps and compute pointing-game hit rates, energy-inside-box IoU, and comparative reliability scores.

---

## Technical Notes & Known Issues

- **Dataset Isolation**: Git tracks only directory anchors (`.gitkeep`) in `data/raw`, `data/processed`, and `outputs`. All raw dataset images, manifests, checkpoints, and logs are excluded by `.gitignore`.
- **Windows PyTorch Environment**: On Windows, PyTorch requires compatible Visual C++ redistributables and CUDA drivers. Ensure your PyTorch installation matches your local GPU drivers before proceeding to the model training stage.
