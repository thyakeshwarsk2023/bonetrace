# Table 7: Audited Qualitative Cases Provenance and Held-Out Validity

**Audit Date:** 2026-10-16  
**Auditor:** Antigravity Forensic Diagnostic Engine  
**Objective:** Resolve the split membership, producing checkpoints, and held-out scientific validity for both the 7 original draft cases and the 14 rule-based balanced cases.

---

## 1. Original 7 Qualitative Cases (Draft Study Report)

| Case ID | Radiograph File | GT Label | Producing Checkpoint | Checkpoint Seed | Initial Split (Producing Run) | Current Split (`data/processed/`) | Held-Out Validity Status | Diagnostic Finding |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **`CANON_0005`** | `IMG0000007.jpg` | Fractured ($y=1$) | `baseline/best_model.pt` | 2026 | Train | **`train.csv`** | **`TRAIN: not valid as a held-out example`** | Flagship Case 1 was in the model's training data. Evaluates training memorization, not generalization. |
| **`CANON_1418`** | `IMG0001815.jpg` | Fractured ($y=1$) | `baseline/best_model.pt` | 2026 | Train | **`train.csv`** | **`TRAIN: not valid as a held-out example`** | Training set image. |
| **`CANON_3325`** | `IMG0004265.jpg` | Fractured ($y=1$) | `baseline/best_model.pt` | 2026 | Train | **`train.csv`** | **`TRAIN: not valid as a held-out example`** | Training set image. |
| **`CANON_2747`** | `IMG0003517.jpg` | Fractured ($y=1$) | `baseline/best_model.pt` | 2026 | Train | **`train.csv`** | **`TRAIN: not valid as a held-out example`** | Training set image. |
| **`CANON_0594`** | `IMG0000771.jpg` | Fractured ($y=1$) | `baseline/best_model.pt` | 2026 | Train | **`train.csv`** | **`TRAIN: not valid as a held-out example`** | Training set image. |
| **`CANON_1482`** | `IMG0001897.jpg` | Non-Fractured ($y=0$) | `baseline/best_model.pt` | 2026 | Train | **`train.csv`** | **`TRAIN: not valid as a held-out example`** | Control case was sampled from the training split. |
| **`CANON_1416`** | `IMG0001813.jpg` | Non-Fractured ($y=0$) | `baseline/best_model.pt` | 2026 | Test | **`test.csv`** | **`VALID HELD-OUT TEST CASE`** | The only genuine test sample among the 7 original draft cases. |

*Summary:* **6 of the 7 original cases (85.7%) were drawn from the training partition**, completely invalidating claims of "held-out test interpretability."

---

## 2. Rule-Based Balanced 14 Cases (Audited Test Cohort)

All 14 cases were selected strictly from the held-out test split (`data/processed/test.csv`, $N=609$) using explicit quantitative criteria across baseline (Model A) and region-supervised (Model B) models evaluated at Seed 2026 (`best_model.pt`).

| Case ID | Anatomical Site & View | GT Label | Selection Category | $P(\text{Frac})_A$ (Base) | $P(\text{Frac})_B$ (Reg-Sup) | Probability Shift ($\Delta P$) | Current Split | Diagnostic Held-Out Status |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **`CANON_0350`** | Leg (Frontal) | Fractured ($y=1$) | Model B Improves Most | $0.218$ | **$0.982$** | $+0.764$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_1617`** | Hand (Frontal) | Fractured ($y=1$) | Model B Improves Most | $0.341$ | **$0.945$** | $+0.604$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_4059`** | Hand (Oblique) | Fractured ($y=1$) | Model B Improves Most | $0.295$ | **$0.887$** | $+0.592$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_0390`** | Hand (Frontal) | Fractured ($y=1$) | Model B Worse | **$0.864$** | $0.312$ | $-0.552$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_0451`** | Leg (Frontal) | Fractured ($y=1$) | Model B Worse | **$0.791$** | $0.285$ | $-0.506$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_1126`** | Shoulder (Frontal) | Fractured ($y=1$) | Model B Worse | **$0.814$** | $0.349$ | $-0.465$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_2386`** | Hand (Frontal) | Fractured ($y=1$) | Concordant True Positive | $0.942$ | $0.989$ | $+0.047$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_1268`** | Leg (Lateral) | Fractured ($y=1$) | Concordant True Positive | $0.915$ | $0.976$ | $+0.061$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_1574`** | Leg (Frontal) | Non-Fractured ($y=0$) | Concordant True Negative | $0.041$ | $0.012$ | $-0.029$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_3531`** | Hand (Frontal) | Non-Fractured ($y=0$) | Concordant True Negative | $0.063$ | $0.008$ | $-0.055$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_1544`** | Shoulder (Frontal) | Non-Fractured ($y=0$) | Persistent False Positive | $0.882$ | $0.841$ | $-0.041$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_1227`** | Hand (Oblique) | Non-Fractured ($y=0$) | Persistent False Positive | $0.795$ | $0.764$ | $-0.031$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_0610`** | Hand (Frontal) | Fractured ($y=1$) | Persistent False Negative | $0.114$ | $0.142$ | $+0.028$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |
| **`CANON_1857`** | Leg (Frontal) | Fractured ($y=1$) | Persistent False Negative | $0.187$ | $0.165$ | $-0.022$ | `test.csv` | **`VALID HELD-OUT TEST CASE`** |

*Verification:* 100% of the 14 balanced cases belong strictly to the held-out test split (`test.csv`). None appear in the training partition.
