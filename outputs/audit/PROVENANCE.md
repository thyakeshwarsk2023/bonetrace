# BoneTrace Forensic Audit: Provenance, Checkpoint Integrity, and Training Configuration

**Audit Date:** 2026-10-16  
**Auditor:** Antigravity Forensic Diagnostic Engine  
**Target Repository:** `bonetrace` (HEAD commit: `2cc966c`)  
**Core Finding:** The numerical divergence between the locked benchmark numbers (e.g. Model B ROC-AUC 0.8100 on $N=610$, 100 positives) and the recomputed results (ROC-AUC 0.9539 on $N=609$, 107 positives) is **not** an experimental replication variance or "cohort shift." It is the direct consequence of evaluating checkpoints on a **completely different, mutated dataset split** created *after* the models had already finished training.

---

## D1. Provenance of the Locked Numbers

### 1.1 Forensic String Search & Source Mapping
Every target metric string was traced across repository files, logs, and commit metadata.

| Locked String | Reported Metric & Model | File Location | Line | Generation Timestamp (Local) | Producing Script | Associated Commit |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **`0.7996`** | Test ROC-AUC Baseline ($0.7996 \pm 0.0047$) | `outputs/evaluation/MULTISEED_REPORT.md` | 12 | 2026-10-13 22:39:38 | `scripts/aggregate_multiseed_results.py` | `2cc966c` (script) / untracked output |
| **`0.8100`** | Test ROC-AUC Region-Sup ($0.8100 \pm 0.0042$) | `outputs/evaluation/MULTISEED_REPORT.md` | 12 | 2026-10-13 22:39:38 | `scripts/aggregate_multiseed_results.py` | `2cc966c` (script) / untracked output |
| **`0.4753`** | Test PR-AUC Baseline ($0.4753 \pm 0.0228$) | `outputs/evaluation/MULTISEED_REPORT.md` | 13 | 2026-10-13 22:39:38 | `scripts/aggregate_multiseed_results.py` | `2cc966c` (script) / untracked output |
| **`0.5088`** | Test PR-AUC Region-Sup ($0.5088 \pm 0.0007$) | `outputs/evaluation/MULTISEED_REPORT.md` | 13 | 2026-10-13 22:39:38 | `scripts/aggregate_multiseed_results.py` | `2cc966c` (script) / untracked output |
| **`89.54`** | Specificity Baseline ($89.54\% \pm 1.48\%$) | `outputs/evaluation/MULTISEED_REPORT.md` | 14 | 2026-10-13 22:39:38 | `scripts/aggregate_multiseed_results.py` | `2cc966c` (script) / untracked output |
| **`94.12`** | Specificity Region-Sup ($94.12\% \pm 1.11\%$) | `outputs/evaluation/MULTISEED_REPORT.md` | 14 | 2026-10-13 22:39:38 | `scripts/aggregate_multiseed_results.py` | `2cc966c` (script) / untracked output |
| **`2.8312`** | Causal Adv Baseline ($2.8312 \pm 1.2087$) | `outputs/evaluation/MULTISEED_REPORT.md` | 15 | 2026-10-13 22:39:38 | `scripts/aggregate_multiseed_results.py` | `2cc966c` (script) / untracked output |
| **`3.5484`** | Causal Adv Region-Sup ($3.5484 \pm 0.2182$) | `outputs/evaluation/MULTISEED_REPORT.md` | 15 | 2026-10-13 22:39:38 | `scripts/aggregate_multiseed_results.py` | `2cc966c` (script) / untracked output |
| **`3.8293`** | FOV-Constrained Causal Adv Positives | `outputs/evaluation/FAITHFULNESS_AUDIT.md` | 36 | 2026-10-13 16:31:21 | `scripts/verify_faithfulness_audit.py` | `2cc966c` (script) / untracked output |
| **`0.0858`** | Expected Calibration Error Baseline | `outputs/REVIEW2_DRAFT_REPORT.md` | 35, 94 | 2026-10-14 02:45:00 | `scripts/generate_calibration_diagrams.py` | `2cc966c` (script) / untracked output |
| **`0.1051`** | Expected Calibration Error Region-Sup | `outputs/REVIEW2_DRAFT_REPORT.md` | 35, 94 | 2026-10-14 02:45:00 | `scripts/generate_calibration_diagrams.py` | `2cc966c` (script) / untracked output |

*Note on git commits:* The scripts that produced these outputs (`aggregate_multiseed_results.py`, `verify_faithfulness_audit.py`, `generate_calibration_diagrams.py`) were committed in `2cc966c` ("feat(evaluation): add localization, faithfulness audit, calibration, and demo scripts"). The generated markdown reports in `outputs/` were excluded from commit by `.gitignore` (line 30: `outputs/*`).

---

### 1.2 The Original Test Cohort Behind the Locked Numbers
The locked numbers were evaluated on a held-out test split of **$N = 610$ images**, composed of:
- **100 fracture-positive images**
- **510 non-fracture control images**

**Evidence from `tests/test_dataset.py` (Commit `3743b79`, Mon Oct 12 23:17:12 2026):**
```python
def test_frozen_split_manifests_exclusion_counts():
    """Verify that the primary benchmark protocol yields 2,842 / 612 / 610 across real split CSVs."""
    proc_dir = Path("data/processed")
    train_ds = FracAtlasDataset(proc_dir / "train.csv", split="train", exclude_conflicts=True)
    val_ds   = FracAtlasDataset(proc_dir / "val.csv",   split="val",   exclude_conflicts=True)
    test_ds  = FracAtlasDataset(proc_dir / "test.csv",  split="test",  exclude_conflicts=True)

    assert len(train_ds) == 2842  # Excluded: CANON_3339 (1 disputed clone)
    assert len(val_ds)   == 612   # Excluded: CANON_1668 (1 disputed clone)
    assert len(test_ds)  == 610   # Excluded: CANON_0206, CANON_0678 (2 disputed clones)
```
- In that original frozen split:
  - Total canonical records: $2,843 + 613 + 612 = 4,068$.
  - After disputed clone exclusions (`exclude_conflicts=True`): **$2,842$ Train / $612$ Val / $610$ Test**.
  - In `test.csv`, exactly 100 images were fracture-positive and 510 were negative controls ($100 / 610 = 16.39\%$).
  - `outputs/evaluation/comparative_evaluation_results.json` line 5 confirms: `"sample_count": 610.0, "pos_count": 100.0, "neg_count": 510.0`.
  - `outputs/evaluation/faithfulness_audit.json` lines 3, 21, 39 confirm: `"Full Test Cohort": 610, "Fracture-Positive Subset": 100, "Non-Fracture Controls": 510`.
  - `outputs/evaluation/LOCALIZATION_REPORT.md` line 3 confirms: `"Held-Out Fracture-Positive Test Cohort (N = 100)"`.

---

## D2. Checkpoint and Split Integrity Audit

### 2.1 Complete Inventory and Cryptographic Hashes of Checkpoints

Every checkpoint file in `outputs/checkpoints/` was inspected for internal dictionary keys, epoch, seed, metrics, and state dict SHA-256:

| Checkpoint Path | File MTime | Size (Bytes) | Internal Seed | Internal Epoch | Logged Val ROC-AUC | Model State Dict SHA-256 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `outputs/checkpoints/baseline/best_model.pt` | 2026-10-13 19:36:49 | 134,260,274 | 2026 | 10 | 0.82130101 | `5f9f00da7ca96c9f420856e21c2801e41bd6ea2b99239da64b4f2432b703c7de` |
| `outputs/checkpoints/baseline/latest_checkpoint.pt` | 2026-10-13 19:58:15 | 134,262,458 | 2026 | 20 | 0.79425372 | `06e9e5adf740feaa9a330ee60104f36ad6d02942d1c675677bc76dc0bef13e50` |
| `outputs/checkpoints/baseline_seed123.pt` | 2026-10-13 04:50:21 | 134,262,146 | 123 | 7 | 0.81288979 | `5cafe58994c2b8061cad35fe5a56a79d60cda3573c04c21d4f43714dd337c0ac` |
| `outputs/checkpoints/baseline_seed2026.pt` | 2026-10-13 19:36:49 | 134,262,458 | 2026 | 10 | 0.82130101 | `5f9f00da7ca96c9f420856e21c2801e41bd6ea2b99239da64b4f2432b703c7de` |
| `outputs/checkpoints/region_supervised/best_model.pt` | 2026-10-13 20:08:34 | 141,350,793 | 2026 | 15 | 0.81729435 | `369b152ef905fb36255a50b1884fe96e56ea23a7de3df4e191e32d3d3f9a330b` |
| `outputs/checkpoints/region_supervised/latest_checkpoint.pt`| 2026-10-13 20:11:56 | 141,353,138 | 2026 | 20 | 0.81226983 | `797ec0ce262a4ef6c62fc27ba7d6093d694734c50b790ce5aa073b5543e46785` |
| `outputs/checkpoints/region_supervised_seed123.pt` | 2026-10-13 09:33:20 | 141,355,882 | 123 | 14 | 0.83560655 | `f4ee501f12ce15d3daa230112c33205ca71672b7ab64700d9d46b8b8f8511389` |
| `outputs/checkpoints/region_supervised_seed2026.pt` | 2026-10-13 20:08:34 | 141,356,217 | 2026 | 15 | 0.81729435 | `369b152ef905fb36255a50b1884fe96e56ea23a7de3df4e191e32d3d3f9a330b` |

#### Critical Findings on Checkpoint Overwriting:
1. **`best_model.pt` is NOT seed 42:**
   - `outputs/checkpoints/baseline/best_model.pt` has internal `seed: 2026` and its state dict hash matches `baseline_seed2026.pt` bit-for-bit (`5f9f00da...`).
   - `outputs/checkpoints/region_supervised/best_model.pt` has internal `seed: 2026` and its state dict hash matches `region_supervised_seed2026.pt` bit-for-bit (`369b152e...`).
2. **Distinctness of Seed Checkpoints:**
   - Seed 123 and Seed 2026 checkpoints are **distinct** and have different state dict hashes.
   - However, **seed-42 checkpoints do not exist as separate files**.
   - In earlier evaluation scripts (`get_checkpoint_path`), when seed 42 was requested, the fallback returned `best_model.pt`, which resulted in evaluating **Seed 2026 twice** under the label of Seed 42.

---

### 2.2 Split Mutation Timeline and Cryptographic Hashes

| File | Current Disk SHA-256 | Frozen Checksum in `stage2/preprocessing_summary.json` | Status |
| :--- | :--- | :--- | :--- |
| `data/processed/train.csv` | `b39823265094e3da63a2a5bfe3cd07253a83577eb03ae7d907892770e56f0d99` | `b39823265094e3da63a2a5bfe3cd07253a83577eb03ae7d907892770e56f0d99` | Match (Frozen at 2026-10-13T21:10:37 UTC) |
| `data/processed/val.csv` | `235c709c3aa2a1eede4dc7ed6a0f54043d094dec18940de24c30d0b775e5faa6` | `235c709c3aa2a1eede4dc7ed6a0f54043d094dec18940de24c30d0b775e5faa6` | Match (Frozen at 2026-10-13T21:10:37 UTC) |
| `data/processed/test.csv` | `887404835d26f59ae0707acb32ffaab2f52f413ae89f4d291a481abb954a7ff6` | `887404835d26f59ae0707acb32ffaab2f52f413ae89f4d291a481abb954a7ff6` | Match (Frozen at 2026-10-13T21:10:37 UTC) |

#### The Event Timeline:
1. **2026-10-12 23:17 (Local):** Commit `3743b79` establishes initial split files with test assertion `assert len(test_ds) == 610` (100 pos / 510 neg).
2. **2026-10-13 04:50 - 20:11 (Local):** Models trained on the initial split. Checkpoints saved. Validation metrics logged on $N=612$ validation cohort.
3. **2026-10-13 22:39 (Local):** Multi-seed evaluation runs on the initial 610 test cohort and writes `outputs/evaluation/MULTISEED_REPORT.md` (locked numbers).
4. **2026-10-14 02:40 (Local) / 2026-10-13 21:10 (UTC):** The script `scripts/preprocess.py` is re-run (`outputs/stage2/PREPROCESSING_REPORT.md`). It re-partitions the entire dataset with `StratifiedGroupKFold(n_splits=20, random_state=42)` and quarantines all 4 disputed clone pairs into `train.csv`. This overwrites `data/processed/test.csv` with a new $N=609$ cohort (107 positives, 502 negatives).
5. **No Retraining Occurred:** Checkpoints were left untouched with timestamps prior to 20:11 on Oct 13.

---

### 2.3 Cross-Split Contamination & Test Leakage
- **Internal Disjointness of Current Splits:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 canonical IDs).
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 canonical IDs).
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 canonical IDs).
- **Contamination between Current Test Set and the Checkpoints' Training Data:**
  - The checkpoints were trained on the **Oct 12 initial train split** ($N = 2,842$).
  - When stage 2 re-preprocessing ran at 02:40 AM on Oct 14, 75 pixel clones were newly grouped and quarantined into `train.csv`, causing the entire 20-fold assignment to reshuffle.
  - As proven by `outputs/explanations/explanations_manifest.json` (evaluated on the initial test split), **7 out of 10 test samples** (`CANON_0005`, `CANON_0012`, `CANON_0015`, `CANON_0037`, `CANON_0038`, `CANON_0039`, `CANON_0041`) were shifted into the current `train.csv`.
  - Conversely, samples from the old training split shifted into the new `test.csv`.
  - Consequently, the existing checkpoints had already seen images in the new test set during their training. **This explains why recomputed ROC-AUC artificially inflated from 0.8100 to 0.9539.**

---

### 2.4 Empirical Validation ROC-AUC Recomputation (Proof of Split Mismatch)
We recomputed validation ROC-AUC on the current `val.csv` ($N = 609$ and $N = 608$ conflict-excluded) using each checkpoint and compared it against the exact validation metric logged during training:

| Checkpoint File | Epoch & Seed | Logged Best Val ROC-AUC | Recomputed Val ROC-AUC on Current `val.csv` | Delta ($\Delta$) | Mismatch Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `baseline/best_model.pt` | Ep 10, Seed 2026 | **0.82130101** | **0.91015378** | **+0.08885277** | **MISMATCH** |
| `baseline_seed123.pt` | Ep 7, Seed 123 | **0.81288979** | **0.88511375** | **+0.07222396** | **MISMATCH** |
| `region_supervised/best_model.pt` | Ep 15, Seed 2026 | **0.81729435** | **0.97726850** | **+0.15997415** | **MISMATCH** |
| `region_supervised_seed123.pt` | Ep 14, Seed 123 | **0.83560655** | **0.98354247** | **+0.14793591** | **MISMATCH** |

**Conclusion:** The massive positive shifts in validation ROC-AUC (+0.07 to +0.16) prove beyond doubt that the checkpoints in `outputs/checkpoints/` do **not** belong to the current `val.csv` or `test.csv`.

---

### 2.5 Diff of Test-Set Membership
- **Locked Test Set (Initial Cohort):**
  - Total Samples: **$610$** (after excluding `CANON_0206` and `CANON_0678` from raw $612$).
  - Fracture Positives: **$100$** ($16.39\%$).
  - Negative Controls: **$510$** ($83.61\%$).
- **Current `data/processed/test.csv`:**
  - Total Samples: **$609$** (0 clones excluded, as all 4 disputed pairs were quarantined in `train.csv`).
  - Fracture Positives: **$107$** ($17.57\%$).
  - Negative Controls: **$502$** ($82.43\%$).
- **Net Cohort Shift:**
  - Total change: $-1$ sample.
  - Positives: **$+7$** ($100 \rightarrow 107$).
  - Negatives: **$-8$** ($510 \rightarrow 502$).

---

### 2.6 Final Split Counts Currently Produced by the Code

From `data/processed/splits_summary.json` and `data/processed/{train,val,test}.csv`:

| Split Name | Total Samples | Fractured Positives ($y=1$) | Non-Fractured Negatives ($y=0$) | Positive Prevalence | Group Leakage Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Split (`train.csv`)** | **2,850** | 501 | 2,349 | 17.58% | 0 groups |
| *(Train after clone quarantine)* | *2,846* | *499* | *2,347* | *17.53%* | *4 clones excluded* |
| **Validation Split (`val.csv`)** | **609** | 107 | 502 | 17.57% | 0 groups |
| **Test Split (`test.csv`)** | **609** | 107 | 502 | 17.57% | 0 groups |
| **Full Clean Dataset** | **4,068** | **715** | **3,353** | **17.58%** | **Strictly Disjoint** |

---

## D3. Training Configuration Actually Used

From internal checkpoint dictionaries (`ckpt['config']`) and training history CSVs (`train_history.csv`):

| Parameter | Saved Value in Checkpoints | Code Implementation | Status / Clarification |
| :--- | :--- | :--- | :--- |
| **Learning Rate Scheduler** | Active | `CosineAnnealingLR(T_max=20, eta_min=1e-6)` | **CONFIRMED:** LR was NOT fixed. It started at $1.0 \times 10^{-4}$ and decayed down to $1.61 \times 10^{-6}$ at epoch 20 (verified in `train_history.csv` column `lr`). |
| **Positive Class Weight (`pos_weight`)** | `4.6166` | `BCEWithLogitsLoss(pos_weight=torch.tensor([4.6166]))` | **CONFIRMED:** Applied to **both** Baseline and Region-Supervised models. Calculated as $N_{neg} / N_{pos} = 2342 / 500 = 4.6166$. |
| **Backbone Initialization** | `pretrained=False` | `weights=None` in `torchvision.models.resnet18` | **CONFIRMED:** Fully randomly initialized from scratch for all runs. Pretrained ImageNet weights were NOT used. |
| **Auxiliary Loss Formulation** | `BCEWithLogitsLoss()` | `nn.BCEWithLogitsLoss()` | **CONFIRMED:** Auxiliary loss is spatial BCE only. **Dice loss does not exist** in `trainer.py`. Reports claiming "Dice + BCE" were factually incorrect. |
| **Model Architecture Distinction** | Single class | `src/bonetrace/models/region_guided.py` | **CONFIRMED:** `region_guided.py` and `region_supervised.py` are the **same model**. `RegionSupervisedModel` is defined in `region_guided.py` and instantiated by `factory.py` under the key `region_supervised`. |
| **Batch Size & AMP** | Baseline: 16, Reg: 32 | `torch.amp.autocast('cuda')` in Region-Sup | Baseline trained with BS=16 (FP32); Region-Supervised trained with BS=32 and mixed precision (AMP). |

---

## D3.1 Code Changes Between Locked Results and HEAD

From `git log -p b10f99e..HEAD`:
1. **Commit `7029f9c` (Tue Oct 13 20:30):**
   - Added automatic saving of seed checkpoints (`outputs/checkpoints/{model_type}_seed{seed}.pt`). Prior to this commit, all runs overwrote `best_model.pt`.
   - Enabled AMP (`torch.amp.autocast`) and `GradScaler` in `trainer.py`.
   - Added multi-worker prefetching (`num_workers=2, persistent_workers=True`).
2. **Commit `2cc966c` (Tue Oct 13 22:45):**
   - Added `scripts/aggregate_multiseed_results.py`, `scripts/verify_faithfulness_audit.py`, `scripts/generate_calibration_diagrams.py`.
   - Added Grad-CAM FOV padding constraint to eliminate letterbox bias.
