# BoneTrace Track B: Pre-Registration & Statistical Analysis Plan

**Pre-Registration Date:** 2026-10-19  
**Status:** LOCKED & COMMITTED PRIOR TO TEST PARTITION EVALUATION  
**Repository State:** Frozen dataset manifests established at `data/frozen/`  
**Frozen Manifest SHA-256 Checksums:**
- `data/frozen/train.csv`: `197c36528d975db3cf0ae91b157e84992797e8bc6ff809b4c053c8c738e4dfad` (2,846 images: 499 pos / 2,347 neg)
- `data/frozen/val.csv`: `0ca5a7146522f25a74ef6f4b66df825b42d1e2e92c00a649646b9a2a7aa8936d` (609 images: 107 pos / 502 neg)
- `data/frozen/test.csv`: `94924c5e3f4ea24dfc08f4c1bb5723f66c07153cb4c28f644b1c2b8c56fa90d9` (609 images: 107 pos / 502 neg)
- `pos_weight` (computed from frozen train split): `4.70340681` (2,347 / 499)

---

## 1. Study Overview & Scientific Objective

Track B conducts a confound-free, fully reproducible retraining and evaluation of all BoneTrace model arms on ONE frozen, cryptographically checksummed dataset partition. The goal is to determine whether auxiliary region supervision (or direct CAM alignment) provides genuine anatomical localization and causal faithfulness beyond general multi-task regularization.

Crucially:
- **Phase 0** establishes the frozen split, validates dataset integrity via automated unit tests, and commits this pre-registration file.
- **Zero test partition evaluation** is permitted until this document is committed to git history.
- The evaluation script (`scripts/run_trackb_eval.py`) validates the SHA-256 of `outputs/trackb/PREREGISTRATION.md` and aborts if it is absent or modified.

---

## 2. Experimental Arms and Architecture

All models share a ResNet-18 convolutional backbone operating on $512 \times 512$ letterboxed radiographs with ImageNet normalization and no data augmentation.

1. **Arm A (Baseline Scratch):**
   - Standard ResNet-18 initialized from scratch (random seed per run).
   - Global average pooling + linear classification projection.
   - Loss: Weighted binary cross-entropy ($\text{pos\_weight} = 4.70340681$).
2. **Arm B (Region-Supervised):**
   - Dual-head ResNet-18 sharing identical backbone with Arm A.
   - Auxiliary convolutional head operating on `layer4` feature maps ($16 \times 16$): $\text{Conv}(512 \to 128, 3\times 3, \text{pad } 1) \to \text{BatchNorm} \to \text{ReLU} \to \text{Conv}(128 \to 1, 1\times 1) \to \text{Bilinear Upsample}(512 \times 512)$.
   - Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$, with $\lambda_{\text{aux}} = 1.0$ and pixel-wise BCE against rectangular bounding-box rasterized masks $M_{\text{box}}$.
3. **Arm B_placebo (Placebo Control):**
   - Identical architecture and loss as Arm B.
   - **Intervention:** For positive training images ($y = 1$), the ground-truth mask is replaced by the bounding-box mask of a *different* positive training image via a pre-computed derangement (fixed permutation per seed, strictly zero fixed points $\sigma(i) \neq i$). Negative images ($y = 0$) retain empty zero masks.
   - **Control Rationale:** Preserves mask frequency, size, and spatial location distributions while completely breaking image-to-mask anatomical correspondence. Disentangles spatial multi-task regularization from true anatomical guidance.
4. **Arm C (Direct CAM Alignment):**
   - Single-head architecture with NO auxiliary convolutional head.
   - Evaluates the differentiable CAM logit map at `layer4`: $s_{u,v} = \sum_{k=1}^{512} W_k A_{k,u,v}$, where $W_k$ are the linear classifier weights.
   - Upsamples $s$ bilinearly to $512 \times 512$ ($s_{\text{up}}$).
   - Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_{\text{cam}} \text{BCEWithLogits}(s_{\text{up}}, M_{\text{box}})$, with $\lambda_{\text{cam}} = 1.0$.
   - **Theoretical property:** $\frac{1}{HW} \sum_{u,v} s_{u,v} + b = \text{logit}_{\text{cls}}$ holds analytically to within numerical precision ($< 10^{-5}$).
5. **Arm A_pre & Arm B_pre (Pretrained Arms):**
   - Identical to Arms A and B, except the ResNet-18 backbone is initialized from torchvision ImageNet-pretrained weights (`ResNet18_Weights.DEFAULT`).

---

## 3. Matched Training Protocol

- **Optimizer:** AdamW ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 10^{-4}$).
- **Schedule:** `CosineAnnealingLR` ($T_{\max} = 20$, $\eta_{\min} = 10^{-6}$).
- **Epochs:** 20 epochs, no early stopping.
- **Batch Size:** Effective batch size 32 under AMP (`torch.amp.autocast('cuda')` + `GradScaler`). Implemented as batch size 16 with 2-step gradient accumulation for guaranteed safety on 4 GB VRAM GPUs.
- **Model Selection:** Best validation ROC-AUC checkpoint saved as `best_model.pt`; final epoch saved as `last_model.pt`.
- **Determinism:** Fixed `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic = True`, and deterministic DataLoader worker initialization.
- **Core Seeds:** `42`, `123`, `2026`.
- **Run ID Convention:** `{arm}_seed{seed}_{git_sha7}`. Every checkpoint embeds its full configuration dictionary and writes an adjacent `config.json`.

---

## 4. Pre-Specified Analysis Plan

### Primary Family (Holm-Bonferroni Corrected, $\alpha = 0.05$)

The primary hypothesis family evaluates whether region supervision yields genuine discrimination, localization, and causal attribution advantages on the locked test partition:

- **P1: Discrimination Difference:** ROC-AUC contrast between Model B and Model A ($B - A$).
- **P2: Specificity Contrast:** Specificity evaluated at the operating threshold chosen on the validation split where sensitivity equals 70%, contrast $B - A$.
- **P3: Placebo Contrast:** Contrast between Model B and Model B_placebo on ROC-AUC and P2-specificity ($B - B_{\text{placebo}}$). Tests whether improvements stem from anatomical localization or multi-task regularization.
- **P4: Layer4 Grad-CAM Pointing Game vs. Box-Prior Null:** Evaluates whether the FOV-constrained maximum attribution peak of `layer4[-1]` Grad-CAM hits the ground-truth fracture box significantly above the training-set box-density spatial mode (box-prior null), for Arms A, B, and C.
- **P5: Direct-Map Pointing Game vs. Box-Prior Null:** Evaluates whether direct model outputs ($\sigma(M_{\text{aux}})$ for Model B; CAM logit map $s_{\text{up}}$ for Model C) hit the ground-truth fracture box above the box-prior null.
- **P6: Area-Matched Lesion Dependence on Positives:** Probability drop for masking the ground-truth lesion box minus the probability drop for an equal-area shape-matched random box (control c), using ImageNet mean fill, evaluated for Model B and Model A.
- **P7: Calibration Error:** Difference in 10-bin expected calibration error (ECE) post-affine recalibration (fitted on validation set) between Model B and Model A ($B - A$).

### Exploratory Family
All other analyses are explicitly pre-registered as **EXPLORATORY**:
- Intermediate layer attributions: `layer3[-1]` and `layer2[-1]` Grad-CAM.
- Alternative XAI methods: `layer3[-1]` HiResCAM, multi-scale LayerCAM (`layers 2+3+4`).
- Alternative perturbation fills: Gaussian blur ($\sigma = 11$, kernel 23).
- Fine-grained calibration: 15-bin equal-mass ECE, maximum calibration error (MCE), Brier score, raw and temperature-only scaling.
- Pretrained arms (A_pre, B_pre).
- Optional extra seeds (`7`, `2027`).

---

## 5. Pre-Specified Decision Rules

1. **Evidence Standard:** An effect is declared **"supported"** if and only if:
   - The stratified paired bootstrap 95% confidence interval (10,000 resamples over test images) excludes zero, AND
   - The direction of the difference agrees across every individual evaluated seed.
   Otherwise, the result is reported as **"not shown"** or **"negative"**.
2. **Placebo Attribution Rule:** If Model B demonstrates a statistically significant gain in ROC-AUC or specificity over Model A, but Model B_placebo demonstrates a comparable gain over Model A (bootstrap CI of $B - B_{\text{placebo}}$ includes zero), the gain MUST be attributed to **multi-task regularization**, NOT to region supervision or anatomical localization.
3. **Transparency Commitment:** All negative results, null baseline comparisons, and failed hypotheses are reported without post-hoc rationalization.

---

## 6. Execution Safeguards

- Checksum validation at initialization of training and evaluation scripts.
- Single per-image parquet file (`outputs/trackb/per_image_results.parquet`) as the single source of truth for all tables and figures.
- Automatic verification that `outputs/trackb/PREREGISTRATION.md` matches its committed SHA-256 hash before test evaluations can proceed.
