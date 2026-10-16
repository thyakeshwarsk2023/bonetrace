# BoneTrace Reproduction Audit: Evaluation Against Locked Benchmarks

**Audit Date:** 2026-10-16  
**Auditor:** Antigravity Forensic Diagnostic Engine  
**Execution Mode:** Pure Inference-Only (Zero Retraining)  
**Primary Finding:** **NONE** of the locked benchmark metrics can be reproduced using the current repository artifacts. The locked numbers were produced on a defunct initial split cohort ($N=610$, 100 positives), whereas existing checkpoints evaluated on the current mutated split ($N=609$, 107 positives) suffer from severe cross-split data leakage.

---

## 1. Reproduction Scorecard: Locked Table vs. Current Cohort Evaluation

Per evaluation directive D4, a metric is marked `REPRODUCED` only if the recomputed value falls within the reported seed standard deviation of the locked benchmark. Otherwise, it is marked `NOT REPRODUCED` with the signed delta and deviation multiple.

| Metric | Model | Locked Benchmark (N=610, 100 Pos) | Recomputed Inference (N=609, 107 Pos) | Absolute Shift ($\Delta$) | Shift in Seed SDs | Status | Root Cause Category |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Test ROC-AUC** | Baseline (Model A) | $0.7996 \pm 0.0047$ | $0.8737 \pm 0.0232$ | $+0.0741$ | $+15.77\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `TRAIN LEAKAGE` |
| **Test ROC-AUC** | Region-Sup (Model B) | $0.8100 \pm 0.0042$ | $0.9539 \pm 0.0049$ | $+0.1439$ | $+34.26\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `TRAIN LEAKAGE` |
| **Test PR-AUC** | Baseline (Model A) | $0.4753 \pm 0.0228$ | $0.6558 \pm 0.0202$ | $+0.1805$ | $+7.92\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `TRAIN LEAKAGE` |
| **Test PR-AUC** | Region-Sup (Model B) | $0.5088 \pm 0.0007$ | $0.7601 \pm 0.0210$ | $+0.2513$ | $+359.0\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `TRAIN LEAKAGE` |
| **Specificity** | Baseline (Model A) | $89.54\% \pm 1.48\%$ | $96.65\% \pm 0.86\%$ | $+7.11\%$ | $+4.80\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `TRAIN LEAKAGE` |
| **Specificity** | Region-Sup (Model B) | $94.12\% \pm 1.11\%$ | $97.58\% \pm 0.35\%$ | $+3.46\%$ | $+3.12\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `TRAIN LEAKAGE` |
| **Causal Advantage**| Baseline (Model A) | $2.8312 \pm 1.2087$ | $1.1111 \pm 0.0000$ | $-1.7201$ | $-1.42\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `FOV SAMPLING` |
| **Causal Advantage**| Region-Sup (Model B) | $3.5484 \pm 0.2182$ | $3.8293 \pm 0.0000$ | $+0.2809$ | $+1.29\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `FOV SAMPLING` |
| **ECE (Uncalibrated)**| Baseline (Model A) | $0.0858$ | $0.0519$ | $-0.0339$ | $-4.46\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `TRAIN LEAKAGE` |
| **ECE (Uncalibrated)**| Region-Sup (Model B) | $0.1051$ | $0.0350$ | $-0.0701$ | $-9.22\,\sigma$ | **NOT REPRODUCED** | `DIFFERENT COHORT` & `TRAIN LEAKAGE` |

---

## 2. Declaration of Non-Reproducibility and Artifact Recoverability

### 2.1 Artifact Recoverability Status
1. **Original Split Manifests:** `NON-RECOVERABLE`
   - The initial `train.csv`, `val.csv`, and `test.csv` (which had 2,842 train, 612 val, 610 test with 100 positives) were overwritten in-place on 2026-10-14 02:40 AM by `scripts/preprocess.py`.
   - Because `data/processed/*` is listed in `.gitignore` (line 26), the file contents were never committed to git history.
2. **Original Seed-42 Checkpoints:** `NON-RECOVERABLE`
   - The original `baseline_seed42.pt` and `region_supervised_seed42.pt` checkpoints were overwritten by the training of Seed 2026 into `outputs/checkpoints/{model}/best_model.pt`.
   - Hashes prove `outputs/checkpoints/baseline/best_model.pt` is bit-for-bit identical to `baseline_seed2026.pt`.
3. **Preserved Checkpoints:** `PARTIALLY RECOVERABLE`
   - `baseline_seed123.pt`, `baseline_seed2026.pt`, `region_supervised_seed123.pt`, and `region_supervised_seed2026.pt` exist on disk.
   - However, they were trained on the defunct initial split and cannot be evaluated on the current split without cross-split leakage.

### 2.2 Authoritative Scientific Baseline
- **Which set of numbers is scientifically valid?**
  - **Neither set can be published in its current state as an honest test evaluation.**
  - The **locked numbers** (ROC-AUC 0.7996 vs 0.8100) were produced on an unquarantined split where 4 disputed clone pairs and 69 pixel clones were distributed across folds. While the test cohort had 610 images (100 positives), the split suffered from clone leakage across train and test.
  - The **recomputed numbers** (ROC-AUC 0.8737 vs 0.9539) are invalid because the checkpoints were trained on the *old* split, meaning current test images were present in the model's training data.
  - **Path to True Validity:** The clean 20-fold grouped split currently in `data/processed/` ($N=4,068$, train 2,846, val 609, test 609) is structurally correct and leakage-free. To establish valid, publishable numbers, models must be retrained on this clean split from scratch.

---

## 3. Retraining Runtime Estimate (RTX 3050 Empirical Log Benchmarks)

Retraining was **not started** in accordance with prompt instructions. The runtime below is calculated directly from empirical epoch timestamps logged in `outputs/logs/train_baseline.log` and `outputs/logs/train_region_supervised.log`:

| Training Phase | Batch Size | Hardware & Precision | Mean Duration per Epoch | Total Epochs (3 Seeds × 20 Epochs) | Estimated Total Duration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline ResNet-18** | 16 | NVIDIA RTX 3050 (FP32, `num_workers=0`) | $110.7\text{ seconds}$ ($1.85\text{ min}$) | 60 | **$110.7\text{ minutes}$ ($1.85\text{ hours}$)** |
| **Region-Supervised ResNet-18** | 32 | NVIDIA RTX 3050 (`torch.amp`, `num_workers=2`) | $64.2\text{ seconds}$ ($1.07\text{ min}$) | 60 | **$64.2\text{ minutes}$ ($1.07\text{ hours}$)** |
| **Complete Multi-Seed Retraining** | — | **Single RTX 3050 Laptop GPU** | — | **120** | **$174.9\text{ minutes}$ ($2.91\text{ hours}$)** |

*Retraining Note:* On this machine, full retraining of both architectures across 3 seeds (seeds 42, 123, 2026) for 20 epochs will take **2 hours 55 minutes**.
