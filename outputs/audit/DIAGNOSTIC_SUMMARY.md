# BoneTrace Diagnostic Summary: Provenance, Leakage, and Scientific Validity

**Date:** 2026-10-16  
**Auditor:** Antigravity Forensic Diagnostic Engine  
**Subject:** Resolution of Benchmark Discrepancies (Locked: ROC-AUC 0.7996/0.8100 on $N=610$ vs. Recomputed: 0.8737/0.9539 on $N=609$)  
**Target Repository:** `bonetrace` (HEAD commit: `2cc966c`)

---

### Question 1: Which checkpoints and splits produced the locked numbers?

**Answer:**
The locked benchmark numbers (e.g. Model B ROC-AUC $0.8100 \pm 0.0042$, PR-AUC $0.5088 \pm 0.0007$, Specificity $94.12\% \pm 1.11\%$, and Causal Advantage $3.5484 \pm 0.2182$) were produced on **2026-10-13 22:39:38** by `scripts/aggregate_multiseed_results.py` using an **initial frozen split of $N = 610$ held-out test images** (100 fracture-positive, 510 non-fracture controls).

- **Evidence from Code:** The split definition is frozen in `tests/test_dataset.py` (Commit `3743b79`, Mon Oct 12 23:17:12 2026, lines 230–245), asserting exact sample counts: Train $N=2,842$, Val $N=612$, Test $N=610$.
- **Checkpoints Used:** 
  - Seeds 123 and 2026 were evaluated from their respective checkpoint files (`baseline_seed123.pt`, `baseline_seed2026.pt`, `region_supervised_seed123.pt`, `region_supervised_seed2026.pt`).
  - For Seed 42, no separate `seed42.pt` file existed; the script fell back to `outputs/checkpoints/{model}/best_model.pt`. Forensic state-dict hashing proves `best_model.pt` was actually **Seed 2026** (state dict SHA-256 `5f9f00da...` for baseline, `369b152e...` for region-supervised), meaning Seed 2026 was evaluated twice under different seed labels.
- **Reporting Inaccuracy:** The locked Causal Advantage ($2.8312$ vs $3.5484$) was evaluated **strictly on the 100 fracture-positive test cases**, not the full test set as claimed in early draft text.

---

### Question 2: Does the current test set overlap any checkpoint's training data?

**Answer: YES, EXTENSIVELY.**
While the current splits (`train.csv`, `val.csv`, `test.csv`) are internally disjoint ($0$ canonical ID overlap), **the current test set severely overlaps the data used to train the existing checkpoints.**

- **Root Cause:** The models in `outputs/checkpoints/` were trained on Oct 13 (between 04:50 and 20:11). Four hours later, at **02:40 AM on Oct 14** (2026-10-13T21:10:37 UTC), `scripts/preprocess.py` was executed (`outputs/stage2/PREPROCESSING_REPORT.md`). This run introduced grouped clone isolation for 75 pixel clones and quarantined all 4 disputed clone pairs into `train.csv`, repartitioning the dataset into $2,850$ Train / $609$ Val / $609$ Test.
- **Direct Empirical Proof:**
  - In `outputs/explanations/explanations_manifest.json` (evaluated on the initial test set), **7 of the 10 test samples** (`CANON_0005`, `CANON_0012`, `CANON_0015`, `CANON_0037`, `CANON_0038`, `CANON_0039`, `CANON_0041`) were shifted into the current `train.csv`.
  - Conversely, samples that the existing checkpoints trained on were shifted into the new `test.csv`.
  - Validation ROC-AUC recomputation on the current `val.csv` yields massive positive mismatches: Baseline best model logged **0.8213** during training but achieves **0.9102** on current `val.csv` (+0.0888); Region-Supervised best model logged **0.8173** but jumps to **0.9773** (+0.1600).
- **Impact:** Evaluating the older checkpoints against the new `test.csv` yields test-set data leakage, explaining why recomputed ROC-AUC artificially inflated from 0.8100 to 0.9539.

---

### Question 3: Which of the two sets of results (locked vs recomputed) is valid?

**Answer: NEITHER SET IS SCIENTIFICALLY VALID FOR PUBLICATION AS AN HONEST TEST BENCHMARK.**

1. **The Locked Numbers ($N=610$, ROC-AUC 0.7996 vs 0.8100):**
   - Evaluated on an unquarantined split where 4 duplicate pairs with conflicting labels and 69 exact pixel clone pairs were distributed across folds without group isolation.
   - Seed 42 was an unintended duplicate evaluation of Seed 2026 due to checkpoint overwriting.
2. **The Recomputed Numbers ($N=609$, ROC-AUC 0.8737 vs 0.9539):**
   - Invalid because the checkpoints were trained on Split 1 while evaluated on Split 2, causing direct cross-split contamination (training images in the test set).
3. **The Path to True Scientific Validity:**
   - The current dataset partition in `data/processed/` ($2,846$ Train, $609$ Val, $609$ Test) is structurally sound, leakage-free, and correctly isolates all duplicate clones into groups.
   - **Both model architectures must be cleanly retrained from scratch on this split.**
   - Retraining 3 seeds × 2 models × 20 epochs will take **2 hours 55 minutes** on the current RTX 3050 hardware.

---

### Question 4: What differs between the code that produced the locked results and the current code?

**Answer:**
Inspection of `git log -p b10f99e..HEAD` reveals the exact differences:

1. **Checkpoint Persistence:** Commit `7029f9c` added `torch.save(state, root_chk_dir / f"{self.model_type}_seed{self.seed}.pt")`. When the original locked seed-42 runs occurred under commit `b10f99e`, this logic did not exist, causing `outputs/checkpoints/{model}/best_model.pt` to be overwritten by subsequent seed runs.
2. **Learning Rate Scheduler:** The code at all commits uses `CosineAnnealingLR(T_max=20, eta_min=1e-6)`. Training logs prove LR actively decayed from $1.0 \times 10^{-4}$ down to $1.61 \times 10^{-6}$ at epoch 20. The author's belief that LR was fixed is contradicted by the active scheduler.
3. **Loss Function:** The code uses spatial `BCEWithLogitsLoss()`. **No Dice loss exists in the codebase.** Prior report claims of "Dice + BCE" were factually inaccurate.
4. **Model Architecture:** `src/bonetrace/models/region_guided.py` defines `RegionSupervisedModel`, which is instantiated under the alias `region_supervised`. There is no separate `region_supervised.py` file; both names refer to the same architecture.
5. **Class Weight:** `pos_weight = 4.6166` was passed to BCE loss for **both** Baseline and Region-Supervised models.
6. **XAI Letterbox Padding Mask:** Commit `2cc966c` added Policy B (FOV-constrained control masks) in `scripts/verify_faithfulness_audit.py`, eliminating the letterbox padding bias that previously allowed 40–50% of control perturbation masks to sample inert black border canvas.
