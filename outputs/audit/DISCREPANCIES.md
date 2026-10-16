# Forensic Discrepancy Log: Locked Benchmark Numbers vs. Current Recomputations

**Audit Date:** 2026-10-16  
**Auditor:** Antigravity Forensic Diagnostic Engine  
**Objective Status Classifications:**
- `REPRODUCED`: Recomputed value falls within reported seed standard deviation ($\le 1.0\,\sigma$).
- `NOT REPRODUCED`: Evaluated on equivalent setup but differs by $> 1.0\,\sigma$.
- `DIFFERENT COHORT`: Evaluated on a non-identical dataset split (e.g., initial 610 split vs. post-audit 609 split).
- `DIFFERENT CHECKPOINT`: Evaluated using distinct checkpoint weights or different training seeds (e.g., fallback to seed 2026 due to missing seed 42).

---

## 1. Metric-by-Metric Forensic Discrepancy Table

| Quantity | Baseline (A) Locked | Baseline (A) Recomputed | Reg-Sup (B) Locked | Reg-Sup (B) Recomputed | Objective Status | Evidence & Forensic Provenance |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Test ROC-AUC** (3-Seed Mean) | $0.7996 \pm 0.0047$ | $0.8737 \pm 0.0232$ | $0.8100 \pm 0.0042$ | $0.9539 \pm 0.0049$ | **DIFFERENT COHORT** & **NOT REPRODUCED** | **Locked:** Evaluated on initial frozen split ($N=610$, 100 positives, 510 negatives) generated on Oct 12.<br>**Recomputed:** Evaluated on mutated split ($N=609$, 107 positives, 502 negatives) generated on Oct 14 at 02:40 AM.<br>**Difference:** Baseline $+0.0741$ ($+15.8\,\sigma$); Model B $+0.1439$ ($+34.3\,\sigma$). Difference caused by test-set contamination (clones and training samples from the initial training set were reassigned into the new test set). |
| **Test PR-AUC** (3-Seed Mean) | $0.4753 \pm 0.0228$ | $0.6558 \pm 0.0202$ | $0.5088 \pm 0.0007$ | $0.7601 \pm 0.0210$ | **DIFFERENT COHORT** & **NOT REPRODUCED** | **Locked:** Initial 610 test cohort ($P = 16.39\%$).<br>**Recomputed:** Mutated 609 test cohort ($P = 17.57\%$).<br>**Difference:** Baseline $+0.1805$ ($+7.9\,\sigma$); Model B $+0.2513$ ($+359\,\sigma$). Test contamination inflates precision scores. |
| **Specificity** ($p \ge 0.5$) | $89.54\% \pm 1.48\%$ | $96.65\% \pm 0.86\%$ | $94.12\% \pm 1.11\%$ | $97.58\% \pm 0.35\%$ | **DIFFERENT COHORT** & **NOT REPRODUCED** | **Locked:** Initial 610 test cohort (510 negative controls).<br>**Recomputed:** Mutated 609 test cohort (502 negative controls).<br>**Difference:** Baseline $+7.11\%$ ($+4.8\,\sigma$); Model B $+3.46\%$ ($+3.1\,\sigma$). |
| **Causal Advantage** (Multi-Seed Mean) | $2.8312 \pm 1.2087$ | $1.1111 \pm 0.0000$ | $3.5484 \pm 0.2182$ | $3.8293 \pm 0.0000$ | **DIFFERENT COHORT** & **NOT REPRODUCED** | **Locked:** Computed across seeds exclusively on the $N=100$ fracture-positive subset of the initial split.<br>**Recomputed:** Evaluated on the $N=107$ fracture-positive subset of the mutated split using single checkpoint (`best_model.pt`). |
| **Fracture-Positive Policy B** ($\Delta y_{\text{attr}} / \Delta y_{\text{ctrl}} / \text{CA}$) | $1.3428 / +0.2317 / 1.1111$ | $2.7212 / -0.7891 / 3.5103$ | $3.1328 / -0.6965 / 3.8293$ | $4.7431 / -0.8415 / 5.5846$ | **DIFFERENT COHORT** | **Locked:** Single checkpoint `best_model.pt` on 100 fracture positives from initial split (`outputs/evaluation/FAITHFULNESS_AUDIT.md`).<br>**Recomputed:** Single checkpoint on 107 fracture positives from mutated split. |
| **Pointing Game Hit Rate** | $3.00\%$ | $0.93\%$ ($1/107$) | $1.00\%$ | $0.93\%$ ($1/107$) | **DIFFERENT COHORT** | **Locked:** Evaluated on initial 100 positive test cases ($3/100$ Baseline, $1/100$ Model B).<br>**Recomputed:** Evaluated on current 107 positive test cases ($1/107$ for both models). |
| **Energy Inside Box** ($E_{\text{in}}$) | $1.89\% \pm 3.99\%$ | $1.74\% \pm 2.94\%$ | $1.79\% \pm 2.92\%$ | $2.15\% \pm 3.12\%$ | **DIFFERENT COHORT** | Evaluated on initial 100 cases vs mutated 107 cases. Numerical magnitude (~1.8–2.1%) is consistent across cohorts. |
| **Heatmap IoU** ($\tau=0.3$) | $0.0194 \pm 0.0382$ | $0.0185 \pm 0.0229$ | $0.0187 \pm 0.0327$ | $0.0189 \pm 0.0232$ | **DIFFERENT COHORT** | Spatial overlap remains bounded at ~0.018–0.019 across both cohorts. |
| **Heatmap IoU** ($\tau=0.5$) | $0.0182 \pm 0.0401$ | $0.0229 \pm 0.0315$ | $0.0184 \pm 0.0393$ | $0.0232 \pm 0.0318$ | **DIFFERENT COHORT** | Numerical magnitude (~0.018–0.023) is consistent across cohorts. |
| **Expected Calibration Error** (ECE) | $0.0858$ ($8.58\%$) | $0.0519$ ($5.19\%$) | $0.1051$ ($10.51\%$) | $0.0350$ ($3.50\%$) | **DIFFERENT COHORT** & **NOT REPRODUCED** | **Locked:** Evaluated on initial 610 test cohort ($0.0858$ vs $0.1051$, $+0.0193$ for Model B).<br>**Recomputed:** Evaluated on mutated 609 cohort ($0.0519$ vs $0.0350$, $-0.0169$ for Model B). The direction of calibration difference reverses between cohorts. |

---

## 2. Checkpoint Availability and Overwrite Audit

| Checkpoint Identifier | Intended Run | Actual Checkpoint on Disk | State Dict SHA-256 | Distinctness Status |
| :--- | :--- | :--- | :--- | :--- |
| `baseline_seed42` | Seed 42 baseline | `outputs/checkpoints/baseline/best_model.pt` | `5f9f00da...` | **DIFFERENT CHECKPOINT** (Overwritten by Seed 2026) |
| `baseline_seed123` | Seed 123 baseline | `outputs/checkpoints/baseline_seed123.pt` | `5cafe589...` | **DISTINCT CHECKPOINT** |
| `baseline_seed2026` | Seed 2026 baseline | `outputs/checkpoints/baseline_seed2026.pt` | `5f9f00da...` | **DISTINCT CHECKPOINT** |
| `region_supervised_seed42` | Seed 42 region-supervised | `outputs/checkpoints/region_supervised/best_model.pt` | `369b152e...` | **DIFFERENT CHECKPOINT** (Overwritten by Seed 2026) |
| `region_supervised_seed123` | Seed 123 region-supervised | `outputs/checkpoints/region_supervised_seed123.pt` | `f4ee501f...` | **DISTINCT CHECKPOINT** |
| `region_supervised_seed2026` | Seed 2026 region-supervised | `outputs/checkpoints/region_supervised_seed2026.pt` | `369b152e...` | **DISTINCT CHECKPOINT** |

---

## 3. Methodological Audit of Claims

1. **"Cohort Shift" Description Removed:**
   - The numerical delta between locked and recomputed metrics is **not** a statistical cohort shift within a valid data distribution.
   - It is an experimental mismatch: the models were trained on Split 1 and tested on Split 2, causing test contamination.
2. **"Hypothesis Falsified" Language Removed:**
   - For calibration (ECE $0.0858$ vs $0.1051$ in the locked report, and $0.0519$ vs $0.0350$ in the recomputed run), the difference does not exceed seed-to-seed variance ($\pm 0.0076$) or bootstrap uncertainty. Neither claim of superiority or inferiority is supported with statistical significance ($p > 0.05$).
3. **Qualitative Case Provenance Corrected:**
   - 6 of the 7 cases in the draft report (`CANON_0005`, `CANON_0594`, `CANON_1418`, `CANON_1482`, `CANON_2747`, `CANON_3325`) belong to `data/processed/train.csv`. They are marked as `TRAIN: not valid as a held-out example`.
   - The 14 balanced cases in `outputs/tables/T7.md` belong strictly to `data/processed/test.csv`.
