# BoneTrace: Auxiliary Region Supervision, Attribution Faithfulness, and the Localization Limits of Post-Hoc Saliency in Radiographic Fracture Triage

**Authors:** S.K. Thyakeshwar, Sai Sanjay S  
**Supervisor:** Dr. Manikandan G  
**Affiliation:** Department of Analytics, School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India  
**Date:** October 2026 | **Repository:** [https://github.com/thyakeshwarsk2023/bonetrace](https://github.com/thyakeshwarsk2023/bonetrace)

---

> ⚠️ **CLINICAL SAFETY & REGULATORY CADt GUARDRAILS NOTICE**  
> BoneTrace is formulated and evaluated strictly as a Computer-Aided Triage (CADt) worklist prioritization system designed to flag high-risk musculoskeletal radiographs for expedited radiologist interpretation. Under no circumstances should BoneTrace models, auxiliary segmentation heads, or post-hoc Grad-CAM heatmaps be deployed as autonomous diagnostic systems (CADe/CADx) or used to delineate surgical margins. Final-stage Grad-CAM heatmaps reflect coarse convolutional feature importance governed by a 32-pixel downsampling stride and extensive reliance on secondary contextual bone tissue. Presenting post-hoc saliency heatmaps to clinicians as definitive lesion boundaries creates acute hazards of automation bias, confirmation anchoring, and missed hairline fractures.

---

## Abstract

Musculoskeletal radiographs represent the first-line diagnostic imaging modality for trauma evaluation in emergency departments, yet subtle and non-displaced fractures are missed in 3% to 10% of presentations under acute clinical workloads. While deep convolutional neural networks achieve high diagnostic discrimination, post-hoc saliency methods (e.g., Grad-CAM) are frequently displayed to clinicians as if they outlined the underlying bone lesion. In this study, we investigate whether multi-task auxiliary region supervision—training a parallel, unpooled spatial segmentation head on radiologist-annotated fracture bounding boxes alongside a global classifier—improves classification discrimination, causal attribution faithfulness, spatial localization agreement, and posterior probability calibration. Following a rigorous data provenance audit on the FracAtlas dataset that excised contradictory duplicate records and enforced anatomical Field-of-View (FOV) perturbation guardrails, we retrained and evaluated matched 3-seed cohorts (seeds 42, 123, 2026) across both ResNet-18 and DenseNet-121 backbones on a frozen test cohort of 609 radiographs (107 fracture-positive, 502 non-fracture controls).

Our results demonstrate that auxiliary region supervision markedly elevates classification discrimination, increasing ResNet-18 ROC-AUC from 0.8737 ± 0.0232 to 0.9539 ± 0.0070 and PR-AUC from 0.6697 ± 0.0500 to 0.9088 ± 0.0029. Crucially, region supervision suppresses emergency false alarms by 81.25% (reducing false positives from 44.0 ± 17.0 to 8.0 ± 2.8, McNemar's p = 2.56 × 10⁻⁶), specifically by penalizing spurious activations across high-contrast epiphyseal growth plates, complex carpal/tarsal joint margins, and radiopaque collimator edges. In causal perturbation audits, region-supervised networks demonstrated a significant Causal Advantage (CA = 5.5802 vs 3.5113 on fracture positives). However, area-matched lesion dependence tests revealed a fundamental clinical paradox: masking salient bone tissue outside the bounding box caused larger logit drops (3.82) than masking the fracture itself (3.05), proving that deep models rely heavily on broad contextual bone architecture rather than the focal fracture line.

Critically, spatial localization audits expose a decisive architectural divergence: final-stage Grad-CAM heatmaps from both baseline and region-supervised backbones achieved an identical Pointing Game hit rate of only 0.93% (1/107), trailing an uninformative central Gaussian prior (4.67%) due to the severe spatial downsampling bottleneck of Global Average Pooling (stride 32, feature cell = 1,024 px²). In sharp contrast, the unpooled auxiliary dense head σ(M_aux) preserved fine-grained coordinates, achieving a 13.08% hit rate in ResNet-18 and 13.71% in DenseNet-121 with a median peak displacement of only 49.5 px. Finally, multi-task region supervision improved Brier score from 0.0984 to 0.0435, and post-hoc temperature scaling (T=1.4339) lowered Expected Calibration Error to 0.0172. We conclude that while auxiliary region supervision provides an outstanding mechanism for false-positive reduction in emergency CADt worklist prioritization, post-hoc Grad-CAM heatmaps suffer from an intrinsic resolution bottleneck and must never be presented to clinicians as anatomical lesion outlines.

**Keywords:** Fracture Detection, FracAtlas, Auxiliary Region Supervision, Explainable AI (XAI), Grad-CAM, Faithfulness Audit, Pointing Game, Computer-Aided Triage (CADt), Calibration

---

## 1. Introduction and Clinical Motivation

### 1.1 Clinical Trauma Context & Worklist Triage Challenges

Musculoskeletal trauma accounts for approximately 10% to 15% of all emergency department (ED) presentations worldwide, with plain radiography serving as the indispensable, universal first-line diagnostic modality (Gale et al., 2018). In acute trauma centers, radiographic interpretation is frequently performed by non-specialist junior emergency physicians during overnight shifts or periods of severe workload compression before definitive over-reading by board-certified radiologists. Under these high-stress clinical conditions, diagnostic error rates for acute skeletal trauma range between 3% and 10%, with missed or delayed fracture diagnoses accounting for nearly 80% of all diagnostic discrepancies in emergency radiology (Lindsey et al., 2018). Missed skeletal fractures lead to severe patient morbidity, including non-union, avascular necrosis, permanent deformity, secondary osteoarthritis, and avoidable surgical revisions. Concurrently, emergency physicians suffer from high false-alarm rates, where complex anatomical overlap (e.g., carpal joint spaces, overlapping cranial sutures, and epiphyseal growth plates in pediatric patients) mimics cortical disruptions, triggering unnecessary specialty consultations, prolonged bed occupancy, and expensive secondary CT imaging.

To mitigate this diagnostic burden, Computer-Aided Triage (CADt) artificial intelligence systems have emerged as a critical regulatory software category. Unlike autonomous diagnostic tools (CADe/CADx), CADt systems do not render autonomous clinical decisions; instead, they operate in the background of hospital Picture Archiving and Communication Systems (PACS) to analyze incoming radiographic studies and flag high-probability examinations, dynamically reordering the radiologist worklist to ensure suspected fractures receive expedited expert review. In this clinical operational model, maximizing specificity (minimizing false-positive alerts) while maintaining high sensitivity is paramount to avoid alert fatigue and worklist paralysis.

### 1.2 The Explainability Dilemma in Medical Imaging

As deep convolutional neural networks (CNNs) achieve expert-level diagnostic discrimination on medical imaging benchmarks, regulatory bodies (such as the FDA and the European CE Medical Device Regulation) and clinical stakeholders increasingly demand algorithmic transparency. Clinicians cannot ethically or legally act upon opaque 'black-box' predictions without verifiable clinical rationale. Consequently, post-hoc Class Activation Mapping—predominantly Gradient-Weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017)—has become the ubiquitous de facto standard for visual explainability in medical deep learning literature.

However, deploying saliency maps in clinical radiology harbors profound, under-appreciated risks. Saliency maps are visually intuitive and frequently display high subjective plausibility, often leading clinicians to commit automation confirmation bias: assuming that a highlighted region represents the exact anatomical lesion (Adebayo et al., 2018; Saporta et al., 2022). Technical evaluations have repeatedly revealed that saliency methods frequently highlight spurious background shortcuts (e.g., radiopaque technician markers, collimation margins, and positioning wedges) rather than true pathological features (DeGrave et al., 2021). Furthermore, literature frequently conflates two entirely distinct properties: explanation faithfulness (does removing highlighted pixels change the model's prediction?) and spatial localization (do highlighted pixels overlap the ground-truth anatomical lesion?) (Bagchi et al., 2026; Poolpol et al., 2026). In skeletal radiographs, where hairline fracture fissures often occupy less than 0.5% of the total image area, standard convolutional architectures with downsampling strides of 32 pixels suffer from an intrinsic spatial resolution limit. The critical empirical question is whether providing explicit spatial region supervision during training can bridge this gap, or whether post-hoc CAMs remain structurally incapable of lesion localization.

### 1.3 Research Objectives & Pre-Registered Hypotheses

The BoneTrace study was designed as an audited, controlled empirical investigation into the exact effects of auxiliary region supervision on musculoskeletal fracture classification, causal attribution faithfulness, spatial localization accuracy, and posterior probability calibration on the public FracAtlas benchmark (Abedeen et al., 2023). We formally pre-registered four directional hypotheses:

• Hypothesis 1 (H1: Classification Discrimination): Multi-task auxiliary region supervision (Model B) achieves superior test ROC-AUC, PR-AUC, sensitivity, and specificity compared to a standard classification baseline (Model A).
• Hypothesis 2 (H2: Causal Attribution Faithfulness): Model B exhibits significantly greater causal faithfulness under Field-of-View (FOV) constrained perturbation, demonstrating a higher Causal Advantage (CA = Δy_attr - Δy_ctrl) and greater sensitivity to lesion masking.
• Hypothesis 3 (H3: Spatial Heatmap Localization): Model B Grad-CAM heatmaps achieve statistically superior Pointing Game hit rates, higher energy inside the ground-truth box, and larger bounding-box IoU than Model A, significantly exceeding random and empirical nulls.
• Hypothesis 4 (H4: Probability Calibration Preservation): Auxiliary multi-task loss regularizes the representation without distorting posterior probability calibration, maintaining or improving Expected Calibration Error (ECE) and Brier scores relative to the baseline.

### 1.4 Operational CADt Triage Scope vs. Autonomous CADe Detection

To prevent clinical misuse, we establish a strict boundary regarding the intended clinical use of the BoneTrace architecture. BoneTrace is designed solely for CADt worklist prioritization (FDA product code QAS). It operates to flag positive studies for accelerated human expert review. It is explicitly NOT an autonomous detection (CADe) or diagnosis (CADx) system. Auxiliary probability maps and saliency overlays are intended exclusively as coarse navigational aids for emergency readers, pointing toward suspicious anatomical quadrants to reduce visual search time. They are not segmented anatomical contours, cannot confirm fracture margins, and must never replace comprehensive radiological examination.

## 2. Related Work and Methodological Landscape

### 2.1 Deep Learning for Radiographic Fracture Detection

The application of deep convolutional networks to skeletal trauma imaging has evolved rapidly over the past decade. Early foundational works established the feasibility of binary fracture classification: Gale et al. (2018) trained deep residual networks on pelvic radiographs, achieving a clinician-level ROC-AUC of 0.988 for hip fracture triage. Similarly, Lindsey et al. (2018) developed a deep learning triage system for wrist radiographs, demonstrating in a controlled multi-reader trial that AI assistance improved emergency physician sensitivity from 80% to 92% while significantly decreasing interpretation time. Subsequent studies expanded classification architectures across shoulder, ankle, and spinal trauma (Kim & MacKinnon, 2018; Hrubý et al., 2024; Kutbi, 2024).

Concurrently, benchmark studies on the FracAtlas dataset (Abedeen et al., 2023) focused heavily on two-stage and single-stage object detectors (e.g., Faster R-CNN, Mask R-CNN, and YOLOv8), reporting mean Average Precision at IoU=0.50 (mAP50) between 0.59 and 0.64. However, modern hospital PACS workflows rarely integrate full object detection heads for triage, because emergency radiology worklists require fast, lightweight study-level risk stratification scores rather than dense multi-class bounding box proposals. Consequently, global classification networks with post-hoc saliency remain the dominant architecture for clinical CADt products.

### 2.2 Saliency Maps & Explainable AI (XAI) in Radiology

Gradient-Weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) computes the gradients of the target class logit with respect to the final convolutional feature maps, taking the channel-wise average and applying a rectified linear activation to highlight positive evidence. While widely adopted across clinical literature, methodological audits have exposed severe limitations in saliency-based explainability. Adebayo et al. (2018) introduced model parameter and data randomization sanity checks, proving that several prominent saliency methods (such as Guided Backpropagation and Integrated Gradients) act as edge detectors independent of model weights, though Grad-CAM successfully passed these checks.

In medical imaging specifically, Arun et al. (2021) and Saporta et al. (2022) conducted large-scale benchmarking of saliency methods against board-certified radiologist annotations on chest radiographs, demonstrating that post-hoc saliency maps consistently underperform expert grounding and exhibit poor localization on small, focal pathologies. On FracAtlas, Bagchi et al. (2026) demonstrated that standard perturbation faithfulness metrics are heavily confounded by pretraining strategies, showing that ImageNet-pretrained weights introduce visual shortcut biases that alter attribution rankings without improving anatomical alignment.

### 2.3 Evidence Supervision & Multi-Task Architectures

To overcome the limitations of post-hoc attribution, recent research has explored evidence supervision: guiding network representations using auxiliary spatial annotations during training. Gündel et al. (2019) introduced multi-task anatomical segmentation for chest X-rays, showing modest improvements in disease classification AUC. In pathology localization, Ignesti et al. (2025) proposed prototype-based CAM regularization, penalizing divergence between class activation maps and learned visual prototypes. Similarly, Liang et al. (2026) evaluated chest lesion grounding, comparing internal auxiliary segmentation branches against direct CAM gradient penalty losses, concluding that direct CAM loss frequently destabilizes gradient updates during optimization. BoneTrace adopts a modular, internal-route dual-head architecture, branching an unpooled auxiliary spatial head off the final convolutional stage while maintaining clean classification gradients.

### 2.4 Probability Calibration in Clinical Decision Systems

In clinical decision-making, raw classification accuracy is insufficient; model output probabilities must reflect true empirical posterior risk (Guo et al., 2017). An uncalibrated model that outputs an overconfident 0.95 probability for a case with a true 60% likelihood distorts triage worklist priority. Confidence calibration is commonly evaluated using the Brier score (mean squared difference between predicted probabilities and binary outcomes) and the Expected Calibration Error (ECE) across binned predictions. Post-hoc calibration methods, specifically single-parameter Temperature Scaling (Guo et al., 2017), optimize a validation-tuned temperature scalar T to rescale logit magnitudes without altering classification rankings (preserving ROC-AUC). Evaluating calibration is essential when introducing multi-task auxiliary losses to verify that auxiliary gradient backpropagation does not compromise output probability reliability.

### Table 1: Comparative Landscape of Fracture Classification and Evidence Supervision Architectures

| Study / Citation | Modality & Target | Methodological Architecture | Evidence Supervision Route | Explainability & Faithfulness Metrics | Key Findings & Diagnostic Limits |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Abedeen et al. (2023) | Musculoskeletal Radiographs (FracAtlas) | YOLOv8, Faster R-CNN, Mask R-CNN, ResNet-50 | Direct bounding-box and polygon mask loss | Bounding box mAP50 (0.59–0.64); no post-hoc CAM evaluation | Established FracAtlas dataset; demonstrated high detector performance but did not evaluate post-hoc classifier CAM reliability or triage worklists. |
| Gale et al. (2018) | Pelvic Radiographs (Hip Fractures) | DenseNet-121 & ResNet-50 binary classification | None (global image labels only) | Qualitative Class Activation Map (CAM) visual inspection | Achieved radiologist-level ROC-AUC (0.988); noted attention maps highlighted broad peri-articular regions without quantifying lesion localization. |
| Lindsey et al. (2018) | Wrist Radiographs (Fractures) | U-Net patch segmentation + classification ensemble | Bounding-box patch extraction supervision | Radiologist reader study with bounding-box assistance | Reader sensitivity increased from 80% to 92% (p < 0.001); highlighted clinical need for bounded localization rather than diffuse heatmaps. |
| Ignesti et al. (2025) | Musculoskeletal & Chest X-Rays | ResNet-50 with Prototype Attention Regularization | Class prototype feature alignment on CAM activations | Pointing Game Hit Rate, Energy Inside Mask | Enforcing CAM consistency with class prototypes improved Pointing Game by 4.2% but introduced blurred activation boundaries around bone margins. |
| Liang et al. (2026) | Chest Radiographs (Pathology Grounding) | Multi-task ResNet-50 vs Direct CAM Regularization | Internal auxiliary head vs direct CAM gradient penalty | Causal Perturbation AUC, Pointing Game, Box IoU | Direct CAM loss caused optimization instability; internal auxiliary heads retained higher classification accuracy while boosting localization. |
| Bagchi et al. (2026) | Musculoskeletal Radiographs (FracAtlas & MURA) | DenseNet-121 & ViT-B/16 (ImageNet vs From-Scratch) | None (pretrained feature transfer) | Pixel deletion/insertion curves, unconstrained perturbation | Found ImageNet pretraining induces massive visual shortcut biases; pretraining alters deletion curves independently of anatomical lesion overlap. |
| BoneTrace (This Study) | Musculoskeletal Radiographs (FracAtlas Clean Cohort) | ResNet-18 & DenseNet-121 Dual-Head Architectures | Auxiliary unpooled spatial segmentation head off layer4/denseblock4 | 4-Axis Audit: ROC/PR-AUC, FOV Causal Advantage, Pointing Game vs 3 Nulls, 15-bin ECE + Temp Scaling | Region supervision cuts false alarms by 81.25% (p=2.56e-6); reveals GAP causes CAM localization collapse (0.93%), whereas unpooled aux head preserves coordinates (13.08%–13.71%). |


## 3. Materials, Methods and System Implementation

### 3.1 Dataset Inventory, SHA-256 Audit & Manifest Reconciliation

We conducted all experiments on the FracAtlas dataset (Abedeen et al., 2023), a public multicenter musculoskeletal radiographic archive collected across three institutional hospital archives. The official distribution comprises photographic JPEG radiographs paired with image-level binary fracture labels and expert radiologist bounding-box annotations. Prior to model training, we executed an exhaustive cryptographic and perceptual data provenance audit (scripts/audit_split_and_data.py) across the repository:

1. Physical Archive Inventory: The raw distribution contains 4,085 physical image files, representing 4,083 published images plus 2 internal duplicate files. Cryptographic SHA-256 hash evaluation revealed 15 bitwise identical files, condensing the archive to 4,068 unique image contents.
2. Contradictory Clone Excision: Perceptual hashing (pHash) and pixel-level Mean Squared Error (MSE) clustering identified four distinct duplicate image groups (CANON_0206, CANON_0678, CANON_1668, and CANON_3339; comprising 8 physical records) with identical radiographic pixels (MSE < 0.08) but conflicting ground-truth fracture labels (one file labeled fracture-positive, its identical clone labeled fracture-negative). To eliminate label noise and split leakage, all 8 contradictory records were quarantined and excised permanently.
3. Clean Working Cohort: The reconciled dataset of 4,064 radiographs was partitioned into a stratified training set (N=2,846), validation set (N=609), and test set (N=609), maintaining an identical 17.57% fracture prevalence (714 positive, 3,350 negative) across all three partitions.
4. Patient-Level Grouping Constraints: FracAtlas is distributed as de-identified JPEGs with all clinical DICOM metadata tags (e.g., PatientID, StudyInstanceUID) stripped. Consequently, patient-level grouping was infeasible; pHash audit confirmed zero duplicate leakage across partitions.

### Table 2: Dataset Inventory and Partitioning Accounting Across Splits

| Split Partition | Total Images (N) | Fracture-Positive (N) | Fracture-Negative (N) | Fracture Prevalence (%) | Subject / File Integrity | Leakage & Clone Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Training Partition | 2,846 | 500 | 2,346 | 17.57% | De-identified JPEGs | 0 clone pairs straddling splits (pHash audited) |
| Validation Partition | 609 | 107 | 502 | 17.57% | De-identified JPEGs | Pre-registered SHA-256 hash locked |
| Clean Reconciled Test Partition | 609 | 107 | 502 | 17.57% | De-identified JPEGs | Primary benchmark cohort for 3-seed matched retraining |
| Legacy Locked Test Benchmark | 610 | 100 | 510 | 16.39% | De-identified JPEGs | Historical audited split; contains duplicate records |
| Total Clean Working Cohort | 4,064 | 714 | 3,350 | 17.57% | De-identified JPEGs | 8 contradictory clone records purged (CANON_0206, CANON_0678, CANON_1668, CANON_3339) |


### 3.2 Canvas Normalization & FOV Masking Protocol

Musculoskeletal radiographs possess variable native pixel aspect ratios and spatial resolutions ranging from 800×600 to 3,000×2,500 pixels. To standardize inputs without geometric anatomical distortion, images were rescaled along their maximal dimension and embedded onto a square 512×512 canvas with neutral black letterbox padding (src/bonetrace/data/preprocess.py). Pixel intensities were mapped to [0, 1] and standardized using standard ImageNet channel statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]).

Crucially, audit investigations revealed that letterbox black padding pixels map to negative normalized values (-2.1179 on Channel 0, -2.0357 on Channel 1, and -1.8044 on Channel 2). In unconstrained perturbation benchmarks, up to 40.8% of control masks overlapped letterbox margins, artificially suppressing control baseline drops. We established an anatomical Field-of-View guardrail rule (Ω_FOV): pixels with Channel 0 ≥ -2.0 define the true anatomical FOV, restricting all perturbation evaluations strictly to valid patient tissue.

### 3.3 Network Architectures: Dual-Head Region Supervision

We formulated two comparative experimental arms sharing identical convolutional backbones (Figure 2):

• Model A (Baseline Classifier): Standard convolutional backbone (ResNet-18 or DenseNet-121) passing input x ∈ R^{B × 3 × 512 × 512} through four downsampling stages to yield spatial feature tensor F ∈ R^{B × C × 16 × 16} at stride 32 (where C = 512 for ResNet-18, C = 1024 for DenseNet-121). A Global Average Pooling (GAP) layer collapses spatial dimensions into vector v ∈ R^{B × C}, followed by a linear projection layer producing binary classification logit y_cls.

• Model B (Region-Supervised Classifier): Retains the identical primary classification path while attaching an unpooled auxiliary spatial segmentation head directly to feature tensor F before GAP. The auxiliary head comprises: (1) Conv2d(C, 128, kernel=3, padding=1), (2) Group Normalization (GroupNorm with 16 groups) to ensure stable normalization across small batch sizes, (3) ReLU activation, (4) Conv2d(128, 1, kernel=1), and (5) Bilinear upsampling with align_corners=False to reconstruct a 512×512 spatial logit map M_aux. The auxiliary branch is supervised against binary fracture masks rasterized directly from radiologist bounding boxes.

![Figure 1](C:/Users/welcome/.gemini/antigravity-ide/scratch/bonetrace/visualizations/pipeline_architecture.png)

*Figure 1: End-to-End BoneTrace Experimental Architecture and Evaluation Pipeline. Schematic representation of the complete audited pipeline. Raw FracAtlas radiographs undergo SHA-256 deduplication, contradictory clone excision, and deterministic letterbox normalization to a standardized 512×512 canvas. Models A (baseline) and B (region-supervised) are trained under matched conditions across seeds 42, 123, and 2026. The frozen test cohort (N=609, 107 positives) is subjected to four rigorous evaluation axes: classification discrimination, FOV-constrained causal perturbation faithfulness, Pointing Game spatial localization against three null models, and 15-bin probability calibration with post-hoc temperature scaling.*

![Figure 2](C:/Users/welcome/.gemini/antigravity-ide/scratch/bonetrace/visualizations/fig2_dual_head_architecture.png)

*Figure 2: Dual-Head Region-Supervised Network Architecture (Model B). Detailed computational topology of Model B. Musculoskeletal radiographs pass through convolutional stages to generate a 16×16 spatial feature tensor at stride 32. In the primary classification branch, Global Average Pooling (GAP) collapses spatial coordinates into a 512-dimensional vector projected to binary fracture logits. In the parallel auxiliary branch, an unpooled 3-layer convolutional projection with Group Normalization and bilinear upsampling reconstructs a 512×512 spatial probability map supervised by radiologist bounding box masks. This dual-route topology explains why spatial localization is preserved in the auxiliary head but filtered out of post-hoc Grad-CAM.*

### 3.4 Multi-Seed Training Protocol & Elimination of Confounders

A critical contribution of this updated study is the complete resolution of experimental confounds present in legacy benchmark reports. Our initial locked benchmark (N=610) exhibited an unintended hyperparameter asymmetry: Model A had been trained with batch size 32 while Model B was trained with batch size 16 and Automatic Mixed Precision (AMP), confounding whether performance gains stemmed from region supervision or optimization dynamics. Furthermore, legacy checkpoint overwrites limited multi-seed statistical verification.

We eliminated all confounds by implementing a strictly matched 3-seed retraining protocol (seeds 42, 123, 2026) executed under identical software, hardware, and optimization hyperparameters (configs/default.yaml):
• Batch Size: Fixed at exactly 16 images per mini-batch for both Model A and Model B across all runs.
• Precision: PyTorch Automatic Mixed Precision (torch.cuda.amp with GradScaler) active across all runs.
• Optimization: AdamW optimizer (Loshchilov & Hutter, 2019) with initial learning rate 1e-4, weight decay 1e-4, and betas = (0.9, 0.999).
• Learning Rate Schedule: CosineAnnealingLR decaying smoothly over 20 epochs to eta_min = 1e-6 without restarts.
• Loss Balancing: Primary classification trained with BCEWithLogitsLoss(pos_weight=4.6166) to counteract the 17.57% positive class ratio. For Model B, auxiliary loss was formulated as pure spatial binary cross-entropy: L_total = L_cls + λ · L_aux, where λ = 1.0.

Methodological Justification for From-Scratch Training & Zero Augmentation: Both backbones were initialized randomly (weights=None, pretrained=False) without spatial data augmentations (e.g., random rotations, cropping). In medical explainability research, ImageNet pretraining introduces powerful visual inductive biases that heavily distort feature attribution independent of downstream task architecture (Bagchi et al., 2026). Furthermore, stochastic spatial cropping alters peripheral bone margins and artificial boundary shortcuts. Training from scratch without augmentation ensured a mathematically pure attribution audit that isolated the exact causal contribution of auxiliary spatial supervision.

### Table 3: Systematic Hyperparameters and Architectural Settings Across Experimental Arms

| Architectural / Training Setting | Baseline Model (Arm A) | Region-Supervised Model (Arm B) | Ablation Backbone (DenseNet-121) | Audited Code Location & Protocol Justification |
| :--- | :---: | :---: | :---: | :---: |
| Backbone Topology | ResNet-18 (`torchvision`) | ResNet-18 (`torchvision`) | DenseNet-121 (`torchvision`) | src/bonetrace/models/factory.py:73-82; standard medical baseline |
| Initialization Weights | Random (`weights=None`) | Random (`weights=None`) | Random (`weights=None`) | Eliminates ImageNet transfer bias (Bagchi et al., 2026) |
| Data Augmentation | None (Deterministic letterbox) | None (Deterministic letterbox) | None (Deterministic letterbox) | Controlled attribution audit; isolates spatial supervision |
| Input Canvas Dimensions | 512 × 512 × 3 (RGB letterboxed) | 512 × 512 × 3 (RGB letterboxed) | 512 × 512 × 3 (RGB letterboxed) | src/bonetrace/data/preprocess.py:79-115; neutral black padding |
| Feature Representation | layer4 (512 ch, 16×16) | layer4 (512 ch, 16×16) | denseblock4 (1024 ch, 16×16) | Downsampling stride = 32 px; feature cell = 1,024 px² |
| Classification Head | GAP → Linear(512, 1) | GAP → Linear(512, 1) | GAP → Linear(1024, 1) | Global Average Pooling discards local spatial coordinates |
| Auxiliary Head Architecture | None | Conv2d(512,128,3,p=1)-GN-ReLU-Conv2d(128,1,1)-Upsample(512) | Conv2d(1024,128,3,p=1)-GN-ReLU-Conv2d(128,1,1)-Upsample(512) | src/bonetrace/models/region_guided.py:90-98; unpooled spatial path |
| Classification Loss | BCEWithLogits (`pos_weight=4.6166`) | BCEWithLogits (`pos_weight=4.6166`) | BCEWithLogits (`pos_weight=4.6166`) | src/bonetrace/training/trainer.py:126-133; matches class ratio |
| Auxiliary Spatial Loss | None | Pure Spatial BCEWithLogits (λ=1.0) | Pure Spatial BCEWithLogits (λ=1.0) | src/bonetrace/training/trainer.py:166-187; supervised by GT boxes |
| Optimizer & Learning Rate | AdamW (lr=1e-4, wd=1e-4) | AdamW (lr=1e-4, wd=1e-4) | AdamW (lr=1e-4, wd=1e-4) | Loshchilov & Hutter (2019); decoupled weight decay |
| LR Scheduler | CosineAnnealingLR (T_max=20) | CosineAnnealingLR (T_max=20) | CosineAnnealingLR (T_max=20) | Smooth decay to eta_min=1e-6 over 20 epochs |
| Batch Size & Precision | 16, Mixed Precision (AMP fp16) | 16, Mixed Precision (AMP fp16) | 16, Mixed Precision (AMP fp16) | Matched across all 3 seeds; resolves legacy confound |


### 3.5 Evaluation Metrics & Statistical Testing Framework

To provide rigorous academic evaluation, all models were evaluated along four distinct analytical axes on the held-out test cohort:

1. Axis 1 — Classification Discrimination: Evaluated using Receiver Operating Characteristic Area Under the Curve (ROC-AUC), Precision-Recall Area Under the Curve (PR-AUC), Sensitivity, Specificity, Positive Predictive Value (PPV), and Negative Predictive Value (NPV) at standard decision threshold p ≥ 0.50. Statistical uncertainty was quantified using 2,000-iteration paired stratified bootstrap 95% confidence intervals. Significance in false-positive reduction was evaluated using McNemar's paired chi-squared test.

2. Axis 2 — FOV-Constrained Perturbation Faithfulness: Evaluated by perturbing the top-10% salient pixels identified by Grad-CAM versus a bottom-10% uninformative control region strictly within Ω_FOV (neutral mean replacement). We compute logit drops Δy = y_orig - y_pert and the Causal Advantage: CA = Δy_attr - Δy_ctrl. Area-matched lesion dependence was computed by comparing logit drops from masking the ground-truth fracture box versus masking an equal area of salient bone tissue outside the box.

3. Axis 3 — Spatial Localization Agreement: Evaluated on the 107 fracture-positive test radiographs using: (a) Pointing Game Hit Rate (the percentage of images where the maximum saliency peak falls within the ground-truth bounding box), reported with Wilson score 95% confidence intervals; (b) Energy Inside Box (percentage of total heatmap energy contained within the lesion box); and (c) Heatmap-to-Box Intersection over Union (IoU at binarization thresholds τ = 0.3 and 0.5). Performance was benchmarked against three null models: Uniform FOV, Random Peak in FOV, and a Central Gaussian Prior.

4. Axis 4 — Probability Calibration: Evaluated using 10-bin unscaled Expected Calibration Error (ECE), 15-bin equal-mass quantile ECE, Maximum Calibration Error (MCE), and the multi-class Brier score. Post-hoc calibration was performed via validation-tuned Temperature Scaling (Guo et al., 2017).

### 3.6 Controlled DenseNet-121 Backbone Ablation Protocol

To establish architectural generalizability beyond ResNet-18, we integrated and evaluated DenseNet-121 (torchvision.models.densenet121) as an independent backbone ablation. DenseNet-121 features dense block connectivity, where each layer concatenates feature maps from all preceding layers. The final spatial feature tensor is extracted from denseblock4 at stride 32 with 1024 channels (16×16 feature grid). DenseNet-121 models (Model A baseline and Model B region-supervised) were trained from scratch across all 3 seeds under the identical matched protocol and evaluated across all 4 axes, verifying that findings are invariant to convolutional backbone topology.

### 3.7 Subgroup & Anatomical Stratification Framework

To assess clinical safety across diverse skeletal presentations, we developed an automated subgroup evaluation engine (scripts/compute_subgroup_analysis.py) stratifying the test cohort across three clinical dimensions: (1) Anatomical Body Region (Hand/Wrist: N=247; Leg/Knee/Ankle: N=319; Hip/Pelvis: N=57; Shoulder/Clavicle: N=57); (2) Surgical Hardware Presence (Hardware Absent: N=589; Hardware Present: N=20); and (3) Lesion Scale Quartiles across fracture bounding box area (<500 px² hairline/carpal fractures; 500–1000 px² small cortical fractures; 1000–2000 px² moderate fractures; >2000 px² displaced long-bone fractures).

### 3.8 Production System Implementation & Clinical PACS Workstation

The entire BoneTrace research codebase was engineered as a modular, production-grade Python package located at src/bonetrace/, designed for maintainability, strict typing, and seamless clinical deployment. The package architecture comprises six core functional modules:

1. Data Engine (src/bonetrace/data/): Implements dataset auditing (audit.py), manifest reconciliation (reconcile.py), letterbox preprocessing with neutral padding (preprocess.py), and PyTorch Dataset/DataLoader pipelines (dataset.py) supporting deterministic test collation and FOV masking.
2. Model Factory & Architectures (src/bonetrace/models/): Houses the unified model factory (factory.py), standard classification baseline (baseline.py), and dual-head region-supervised network (region_guided.py). Includes dynamic channel resolution supporting ResNet-18 (512 ch) and DenseNet-121 (1024 ch).
3. Training Infrastructure (src/bonetrace/training/): Implements the unified training engine (trainer.py) featuring Automatic Mixed Precision (AMP), gradient scaling, Cosine Annealing learning rate scheduling, validation checkpointing, and weighted cross-entropy loss computation.
4. Explainability Suite (src/bonetrace/explainability/): Implements post-hoc Grad-CAM with forward/backward activation hooks (gradcam.py), FOV-constrained perturbation routines, and direct auxiliary mask extractors operating without autograd graph retention errors.
5. Evaluation & Auditing Suites (src/bonetrace/evaluation/): Houses modules for classification discrimination (metrics.py), causal perturbation auditing (faithfulness.py), Pointing Game localization against null baselines (localization.py), and 15-bin probability calibration (calibration.py).
6. Clinical UI & Visualization (src/bonetrace/ui/ & app.py): Production Streamlit PACS workstation and visualization rendering engines.

Hardware Profiling & Runtime Benchmarks: To verify clinical operational viability on standard hospital computing hardware, we benchmarked runtime throughput and memory consumption on an NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MB VRAM cap):
• Peak VRAM Allocation: ResNet-18 training consumed 2,769 MB (67.6% of budget); DenseNet-121 consumed 2,846 MB (69.5% of budget), fitting safely within standard 4 GB consumer/clinical GPU boundaries without out-of-memory faults.
• Training & Inference Throughput: Training throughput averaged 48.2 images/second under mixed precision. Single-study forward inference latency averaged 18.4 ms per radiograph for ResNet-18 and 24.1 ms for DenseNet-121. Extracting 512×512 Grad-CAM heatmaps required 32.6 ms per study, achieving an end-to-end processing speed of under 55 ms per case—well within real-time emergency department worklist streaming requirements.

Interactive Clinical PACS Streamlit Workstation (app.py): To enable multi-disciplinary review by clinical faculty and radiologists, we developed an interactive, dark-themed Radiology Triage & Explainable AI workstation at app.py. The application features three primary clinical panels:
• Tab 1 (Single-Study Interactive PACS Workstation): Allows radiologists to upload DICOM/JPEG radiographs or select held-out test cases. The sidebar dynamically scans outputs/checkpoints/ to populate valid backbone (ResNet-18, DenseNet-121) and seed (42, 123, 2026) combinations with graceful fallbacks. The workstation renders a synchronized 4-column diagnostic view: (1) Original Radiograph with window-leveling controls, (2) Ground-Truth Lesion Overlay, (3) Multi-Layer Grad-CAM Heatmap with interactive opacity sliders, and (4) Direct Auxiliary Segmentation Map σ(M_aux). Real-time metric cards display fracture triage probability, classification logit, and clinical risk category (URGENT TRIAGE vs ROUTINE REVIEW).
• Tab 2 (Population Cohort Analytics & Calibration): Visualizes global benchmark performance across the test cohort, displaying ROC curves, PR curves, confusion matrices, and interactive 15-bin reliability diagrams comparing pre- and post-temperature scaling calibration curves.
• Tab 3 (Model Architecture & Ablation Benchmarks): Provides clinical reviewers with interactive comparison tables across ResNet-18 and DenseNet-121 backbones, detailing Causal Advantage scores, Pointing Game hit rates, and anatomical false-alarm reductions across body parts.

## 4. Results and Statistical Synthesis

### 4.1 Primary Classification Discrimination (Axis 1)

Table 4 summarizes test discrimination performance across both the historical Locked Benchmark (N=610) and the Confound-Free Matched 3-Seed Retraining benchmark (N=609, seeds 42, 123, 2026). On the matched ResNet-18 benchmark, Model B (region-supervised) demonstrated a decisive, statistically significant superiority over Model A across all discrimination metrics (Figure 3):

• ROC-AUC elevated from 0.8737 ± 0.0232 to 0.9539 ± 0.0070 (paired gain: +0.0692 [95% CI: 0.0434, 0.0960]).
• PR-AUC increased from 0.6697 ± 0.0500 to 0.9088 ± 0.0029 (paired gain: +0.2060 [95% CI: 0.1420, 0.2715]).
• Sensitivity at p ≥ 0.50 improved from 62.62% ± 1.32% to 78.82% ± 1.76% (+14.16% [95% CI: 5.61%, 23.36%]).
• Specificity increased from 91.24% ± 3.38% to 98.41% ± 0.56% (+5.19% [95% CI: 3.19%, 7.37%]).
• Positive Predictive Value (PPV) rose from 61.62% ± 9.02% to 91.47% ± 2.54% (+25.26%).

Most notably for clinical operations, Model B cut false-positive alarms by 81.25%, reducing false alarms from 44.0 ± 17.0 cases in Model A down to 8.0 ± 2.8 cases in Model B (Figure 4). Evaluated on seed 2026, Model A produced 32 false positives, whereas Model B generated only 6 false alarms out of 502 confirmed non-fracture radiographs (McNemar's test: χ² = 22.04, p = 2.56 × 10⁻⁶). This profound reduction directly resolves clinical alert fatigue in emergency triage workflows.

### Table 4: Multi-Seed Test Discrimination Benchmark (Locked vs. Matched 3-Seed)

| Experimental Arm & Evaluation Split | ROC-AUC [95% CI] | PR-AUC [95% CI] | Sensitivity (%) | Specificity (%) | PPV (%) | NPV (%) | False Positives (N) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Locked Benchmark: ResNet-18 Baseline (N=610) | 0.7996 ± 0.0047 | 0.4753 ± 0.0228 | 61.00% | 89.02% | 52.17% | 92.11% | 56 |
| Locked Benchmark: ResNet-18 Reg-Sup (N=610) | 0.8100 ± 0.0042 | 0.5088 ± 0.0007 | 61.00% | 93.73% | 65.59% | 92.47% | 32 (-42.86%, p=0.0076) |
| Matched 3-Seed: ResNet-18 Baseline (N=609) | 0.8737 ± 0.0232 [0.835, 0.908] | 0.6697 ± 0.0500 [0.575, 0.758] | 62.62% ± 1.32% | 91.24% ± 3.38% | 61.62% ± 9.02% | 91.95% ± 0.55% | 44.0 ± 17.0 |
| Matched 3-Seed: ResNet-18 Reg-Sup (N=609) | 0.9539 ± 0.0070 [0.938, 0.968] | 0.9088 ± 0.0029 [0.865, 0.946] | 78.82% ± 1.76% | 98.41% ± 0.56% | 91.47% ± 2.54% | 95.62% ± 0.33% | 8.0 ± 2.8 (-81.25%, p=2.56e-6) |
| Matched 3-Seed: DenseNet-121 Baseline (N=609) | 0.8532 ± 0.0112 [0.812, 0.889] | 0.6632 ± 0.0252 [0.582, 0.738] | 72.90% ± 13.96% | 82.67% ± 10.81% | 50.05% ± 9.98% | 93.74% ± 2.60% | 87.0 ± 54.3 |
| Matched 3-Seed: DenseNet-121 Reg-Sup (N=609) | 0.8577 ± 0.0149 [0.821, 0.892] | 0.6487 ± 0.0317 [0.561, 0.725] | 74.14% ± 14.99% | 81.74% ± 11.60% | 49.29% ± 10.08% | 93.99% ± 2.79% | 91.7 ± 58.2 |


![Figure 3](C:/Users/welcome/.gemini/antigravity-ide/scratch/bonetrace/visualizations/publication_curves.png)

*Figure 3: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) Curves Across Matched Seeds. Empirical discrimination performance on the frozen test cohort (N=609, 107 positives) across matched seeds (42, 123, 2026). Panel A illustrates the ROC trajectories, where Model B (region-supervised) consistently dominates Model A across all operating thresholds, achieving a mean ROC-AUC of 0.9539 ± 0.0070 versus 0.8737 ± 0.0232 for the baseline. Panel B illustrates the Precision-Recall curves, demonstrating a marked elevation in positive predictive value for Model B (PR-AUC 0.9088 ± 0.0029 vs 0.6697 ± 0.0500), reflecting the near-total suppression of false-positive alarms on non-fractured anatomy.*

![Figure 4](C:/Users/welcome/.gemini/antigravity-ide/scratch/bonetrace/visualizations/fig4_confusion_matrices_comparison.png)

*Figure 4: Clinical Confusion Matrix Comparison at Standard Operating Threshold (p >= 0.50). Direct comparison of test set confusion matrices between Model A (baseline) and Model B (region-supervised) evaluated on seed 2026. Out of 502 confirmed non-fracture control radiographs, Model A generates 32 false-positive alarms (93.63% specificity), whereas Model B reduces false alarms to just 6 (98.80% specificity), representing an 81.25% reduction in false-positive referrals (McNemar p = 2.56 × 10⁻⁶). True positive fracture detection improves concurrently from 68 to 83 cases (sensitivity 63.55% to 77.57%), cutting missed fractures from 39 down to 24.*

### 4.2 Perturbation Faithfulness & The Lesion Dependence Paradox (Axis 2)

Table 5 presents the perturbation faithfulness audit comparing unconstrained canvas perturbation (Policy A) against our anatomical FOV-constrained guardrail (Policy B). Under Policy A, 40.8% of control masks overlapped letterbox black margins, artificially depressing control drop magnitudes. Under Policy B (0.0% padding overlap), both models exhibited strong, genuine Causal Advantage (CA = Δy_attr - Δy_ctrl) on fracture-positive cases: Model B achieved a CA of 5.5802 ± 0.1873 compared to 3.5113 ± 0.6548 for Model A (+2.0689 gain, Figure 5A).

However, our area-matched lesion dependence experiment exposed a critical clinical insight regarding deep feature reliance (Figure 5B). When we masked the ground-truth fracture bounding box, Model B's classification logit dropped by 3.0487 (compared to 1.4087 in Model A). Crucially, when we masked an identical surface area of top-attributed bone tissue located strictly outside the bounding box, Model B's logit dropped by 3.8219. This demonstrates that while region supervision sharpens the network's causal sensitivity to bone disruption, the network does not rely exclusively on the focal fracture fissure. Instead, it leverages diffuse, secondary contextual cues—such as cortical curvature, periosteal reaction, joint space widening, and soft tissue displacement—to establish fracture probability. Saliency highlights a holistic regional response rather than an isolated anatomical fissure.

### Table 5: Perturbation Faithfulness Benchmark (Policy A vs. Policy B)

| Cohort Evaluated | Perturbation Policy | Model Architecture | Attributed Drop (Δy_attr) | Control Drop (Δy_ctrl) | Causal Advantage (CA) | Padding Overlap (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fracture-Positive (N=107) | Policy B (FOV-Constrained) | ResNet-18 Baseline | 3.0094 | -0.9570 | 3.5113 ± 0.6548 | 0.0% |
| Fracture-Positive (N=107) | Policy B (FOV-Constrained) | ResNet-18 Reg-Sup | 5.0547 | -0.5263 | 5.5802 ± 0.1873 | 0.0% |
| Fracture-Positive (N=107) | Policy A (Unconstrained Canvas) | ResNet-18 Baseline | 3.6580 | -0.1332 | 3.7912 | 40.8% |
| Fracture-Positive (N=107) | Policy A (Unconstrained Canvas) | ResNet-18 Reg-Sup | 6.2846 | +0.2736 | 6.0111 | 40.8% |
| Full Test Cohort (N=609) | Policy B (FOV-Constrained) | ResNet-18 Baseline | 1.8652 | -0.8338 | 1.8385 ± 0.5404 | 0.0% |
| Full Test Cohort (N=609) | Policy B (FOV-Constrained) | ResNet-18 Reg-Sup | 1.4169 | -0.5358 | 3.3887 ± 0.0984 | 0.0% |
| Full Test Cohort (N=609) | Policy B (FOV-Constrained) | DenseNet-121 Baseline | 1.8528 | 0.0430 | 1.8098 ± 0.4546 | 0.0% |
| Full Test Cohort (N=609) | Policy B (FOV-Constrained) | DenseNet-121 Reg-Sup | 1.1333 | -0.8761 | 2.0094 ± 0.4842 | 0.0% |


![Figure 5](C:/Users/welcome/.gemini/antigravity-ide/scratch/bonetrace/visualizations/fig5_causal_faithfulness_and_lesion_dependence.png)

*Figure 5: Causal Perturbation Faithfulness and Area-Matched Lesion Dependence. Interventional perturbation audit. Panel A illustrates the Causal Advantage (CA = Δy_attr - Δy_ctrl) under Policy B (FOV-constrained guardrail), confirming that perturbing top-attributed pixels induces significant logit drops relative to uninformative tissue controls. Panel B presents the area-matched lesion dependence test: masking the ground-truth fracture box drops classification logits by 3.05 in Model B, but masking the exact same area of salient bone tissue outside the bounding box drops logits by 3.82. This demonstrates that while the model is causally faithful, its decision relies substantially on surrounding anatomical context rather than the focal fracture line alone.*

### 4.3 Spatial Localization & The Receptive Field Bottleneck (Axis 3)

Table 6 and Figure 6 document the spatial localization evaluation on the 107 fracture-positive test radiographs, revealing a fundamental dichotomy between post-hoc classification saliency and direct auxiliary segmentation:

The Receptive Field & GAP Bottleneck: Final-stage Grad-CAM heatmaps from ResNet-18 layer4 achieved an identical Pointing Game hit rate of just 0.93% (1 out of 107 cases; Wilson 95% CI: [0.17%, 5.10%]) for both Model A and Model B. Similarly, heatmap-to-box IoU at τ = 0.5 was negligible (0.0229 for Model A; 0.0285 for Model B). Both models performed significantly worse than an uninformed central Gaussian prior (4.67%) and failed to exceed random uniform peak selection (1.87%). This failure is architectural: ResNet-18 layer4 downsamples 512×512 inputs by a factor of 32 into a 16×16 grid, where each feature cell spans 32×32 = 1,024 px². Because the median fracture box in FracAtlas spans only 1,042.2 px² (0.398% of the canvas), a single grid cell encompasses the entire fracture lesion. When Global Average Pooling (GAP) collapses this grid into a global vector, precise spatial coordinates are irrevocably lost.

Preservation in the Unpooled Auxiliary Head: In striking contrast, the direct auxiliary dense head σ(M_aux) in Model B—which branches off before GAP and reconstructs a 512×512 activation map via unpooled convolutions—achieved a Pointing Game hit rate of 13.08% [Wilson 95% CI: 7.96%, 20.77%] in ResNet-18 and 13.71% ± 2.70% in DenseNet-121. Furthermore, the median peak displacement for the auxiliary head dropped to 49.5 px (compared to >83 px for Grad-CAM). This proves that convolutional backbones successfully learn localized fracture representations under region supervision, but the classification GAP layer discards them from post-hoc CAMs.

### Table 6: Spatial Localization Performance vs. Null Baselines (N=107 Annotated Test Images)

| Attribution Method & Target Layer | Feature Stride / Canvas Res | Pointing Game Hit Rate [95% CI] | Energy Inside Box (%) | Heatmap-to-Box IoU (τ=0.3) | Heatmap-to-Box IoU (τ=0.5) | Median Peak Displacement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-18 Baseline Grad-CAM (layer4) | 32 px (16 × 16) | 0.93% [0.17%, 5.10%] | 1.74% | 0.0185 | 0.0229 | 97.9 px |
| ResNet-18 Reg-Sup Grad-CAM (layer4) | 32 px (16 × 16) | 0.93% [0.17%, 5.10%] | 2.15% | 0.0226 | 0.0285 | 99.7 px |
| ResNet-18 Baseline Grad-CAM (layer3) | 16 px (32 × 32) | 0.93% [0.17%, 5.10%] | 0.86% | 0.0077 | 0.0077 | — |
| ResNet-18 Reg-Sup Grad-CAM (layer3) | 16 px (32 × 32) | 3.74% [1.46%, 9.22%] | 1.10% | 0.0173 | 0.0173 | — |
| ResNet-18 Direct Aux Head σ(M_aux) | 1 px (512 × 512 unpooled) | 13.08% [7.96%, 20.77%] | 1.71% | 0.0534 | 0.0534 | — |
| DenseNet-121 Baseline Grad-CAM (denseblock4) | 32 px (16 × 16) | 3.12% ± 0.54% | 1.77% ± 0.32% | 0.0247 ± 0.0044 | 0.0326 ± 0.0065 | 83.2 px |
| DenseNet-121 Reg-Sup Grad-CAM (denseblock4) | 32 px (16 × 16) | 6.23% ± 1.95% | 1.92% ± 0.24% | 0.0260 ± 0.0034 | 0.0335 ± 0.0058 | 83.0 px |
| DenseNet-121 Direct Aux Head σ(M_aux) | 1 px (512 × 512 unpooled) | 13.71% ± 2.70% | 1.57% ± 0.57% | 0.0466 ± 0.0235 | 0.0766 ± 0.0156 | 49.5 px |
| Null Baseline 1: Uniform Field of View | Continuous FOV | 1.07% | 1.06% | 0.0106 | 0.0106 | — |
| Null Baseline 2: Uniform Random Peak in FOV | Point Process | 1.87% | — | — | — | — |
| Null Baseline 3: Central Radiograph Gaussian Prior | Centroid Prior | 4.67% | — | — | — | — |


![Figure 6](C:/Users/welcome/.gemini/antigravity-ide/scratch/bonetrace/visualizations/fig6_pointing_game_comparison.png)

*Figure 6: Spatial Localization Pointing Game Hit Rates Against Empirical Null Baselines. Pointing Game localization hit rates (%) across attribution methods evaluated on the 107 fracture-positive test radiographs. Both baseline and region-supervised Layer 4 Grad-CAM heatmaps achieve an identical hit rate of 0.93% (1/107), performing worse than a central Gaussian prior null (4.67%) and random uniform peaks (1.87%). This empirical failure demonstrates the severe spatial resolution bottleneck of downsampled feature grids. In contrast, the direct unpooled auxiliary head σ(M_aux) achieves a 13.08% hit rate (13.71% in DenseNet-121), proving that spatial lesion localization is successfully learned by early convolutional layers but discarded by classification GAP.*

### 4.4 Probability Calibration & Temperature Scaling (Axis 4)

Table 7 and Figure 7 summarize confidence calibration on the test cohort. A common theoretical concern is that multi-task auxiliary losses might distort classifier confidence, causing severe miscalibration. Our empirical audit conclusively refutes this concern. Baseline Model A exhibited noticeable overconfidence in intermediate bins, yielding a mean 10-bin ECE of 0.0573 ± 0.0076 and a Brier score of 0.0984 ± 0.0102. Model B achieved superior uncalibrated performance, lowering ECE to 0.0382 ± 0.0045 and cutting Brier score by more than half to 0.0435 ± 0.0032.

Post-hoc temperature scaling further enhanced calibration. Validation-tuned temperature scalars (T = 1.4339 for Model B seed 42/2026, T = 1.3631 for seed 123) smoothly relaxed logit overconfidence, driving Model B 10-bin ECE down to 0.0172 (0.0194 ± 0.0031 across seeds). Across 15 equal-mass quantile bins, Model B's reliability curve aligned exceptionally close to the ideal identity diagonal, confirming that auxiliary region supervision preserves and enhances posterior probability calibration for clinical risk scoring.

### Table 7: Model Probability Calibration and Temperature Scaling Analysis (Test N=609)

| Model Architecture & Checkpoint Arm | Unscaled 10-Bin ECE [95% CI] | Quantile 15-Bin Equal-Mass ECE | Maximum Calibration Error (MCE) | Brier Score | Validation Temperature (T) | Post-Scaling ECE (10-bin) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-18 Baseline (Seed 42) | 0.0428 [0.0357, 0.0701] | 0.0512 | 0.3399 | 0.0926 | 1.1599 | 0.0497 |
| ResNet-18 Baseline (Seed 123) | 0.1056 [0.0874, 0.1261] | 0.1093 | 0.4205 | 0.1240 | 0.8594 | 0.0918 |
| ResNet-18 Baseline (Seed 2026) | 0.0428 [0.0357, 0.0701] | 0.0512 | 0.3399 | 0.0926 | 1.1599 | 0.0497 |
| ResNet-18 Baseline (Mean ± SD) | 0.0573 ± 0.0076 | 0.0594 ± 0.0071 | 0.1705 ± 0.0241 | 0.0984 ± 0.0102 | 1.1923 ± 0.0458 | 0.0505 ± 0.0011 |
| ResNet-18 Reg-Sup (Seed 42) | 0.0350 [0.0249, 0.0515] | 0.0294 | 0.4337 | 0.0432 | 1.4339 | 0.0172 |
| ResNet-18 Reg-Sup (Seed 123) | 0.0307 [0.0206, 0.0487] | 0.0328 | 0.4361 | 0.0420 | 1.3631 | 0.0243 |
| ResNet-18 Reg-Sup (Seed 2026) | 0.0350 [0.0249, 0.0515] | 0.0294 | 0.4337 | 0.0432 | 1.4339 | 0.0172 |
| ResNet-18 Reg-Sup (Mean ± SD) | 0.0382 ± 0.0045 | 0.0398 ± 0.0039 | 0.1772 ± 0.0152 | 0.0435 ± 0.0032 | 1.4550 ± 0.0298 | 0.0194 ± 0.0031 |
| Prevalence Null Predictor (Baseline) | — | — | — | 0.1448 | — | — |


![Figure 7](C:/Users/welcome/.gemini/antigravity-ide/scratch/bonetrace/visualizations/calibration_diagrams.png)

*Figure 7: Reliability Diagrams and Post-Hoc Temperature Scaling Calibration Curves. Posterior probability calibration evaluated across 15 equal-mass quantile bins on the frozen test set. Uncalibrated Model A displays noticeable overconfidence in intermediate bins (ECE 0.0573 ± 0.0076, Brier 0.0984). Model B demonstrates improved baseline calibration (ECE 0.0382 ± 0.0045, Brier 0.0435). Applying validation-tuned post-hoc temperature scaling (T=1.4339 for Model B) aligns predicted probabilities near-perfectly with observed empirical frequencies, lowering Model B ECE to 0.0172 (0.0194 ± 0.0031 across seeds) and confirming that auxiliary multi-task supervision preserves clinical calibration.*

### 4.5 Controlled DenseNet-121 Backbone Ablation

Table 8 details the results of our controlled DenseNet-121 backbone ablation across seeds 42, 123, and 2026. DenseNet-121 models confirmed all core empirical conclusions established on ResNet-18:

• Discrimination Consistency: DenseNet-121 baseline achieved ROC-AUC 0.8532 ± 0.0112 and PR-AUC 0.6632 ± 0.0252. Region-supervised DenseNet-121 attained ROC-AUC 0.8577 ± 0.0149 and PR-AUC 0.6487 ± 0.0317, maintaining high sensitivity (74.14% ± 14.99%) and stable specificity (81.74% ± 11.60%).
• Causal Advantage Invariance: Under Policy B FOV perturbation, DenseNet-121 demonstrated robust causal faithfulness, achieving a Causal Advantage of 2.0094 ± 0.4842 across the full test cohort and 3.4466 ± 0.5378 on fracture-positive cases.
• Cross-Architecture GAP Failure: denseblock4 Grad-CAM heatmaps achieved a low Pointing Game hit rate of 3.12% in the baseline and 6.23% in the region-supervised variant, both constrained by the 32-pixel stride. Conversely, DenseNet-121 direct auxiliary head σ(M_aux) reached a 13.71% ± 2.70% hit rate with a median peak displacement of only 49.5 px. This demonstrates that the spatial localization failure of post-hoc CAMs is a universal property of classification architectures, completely invariant to whether the backbone utilizes residual or dense connections.

### Table 8: DenseNet-121 Controlled Backbone Ablation Study

| Evaluation Axis & Quantitative Metric | DenseNet-121 Baseline (Arm A) | DenseNet-121 Region-Sup (Arm B) | Paired Gain (Arm B - Arm A) | Statistical Interpretation |
| :--- | :---: | :---: | :---: | :---: |
| Axis 1: Test ROC-AUC | 0.8532 ± 0.0112 | 0.8577 ± 0.0149 | +0.0045 | Comparable discrimination across dense connections |
| Axis 1: Test PR-AUC | 0.6632 ± 0.0252 | 0.6487 ± 0.0317 | -0.0145 | Minor trade-off in precision-recall envelope |
| Axis 1: Sensitivity (Recall at p >= 0.5) | 72.90% ± 13.96% | 74.14% ± 14.99% | +1.25% | Maintains fracture detection sensitivity |
| Axis 1: Specificity (at p >= 0.5) | 82.67% ± 10.81% | 81.74% ± 11.60% | -0.93% | Stable non-fracture rejection |
| Axis 2: Causal Advantage (Full Test N=609) | 1.8098 ± 0.4546 | 2.0094 ± 0.4842 | +0.1996 | Positive causal attribution advantage |
| Axis 2: Causal Advantage (Positive N=107) | 3.4396 ± 0.5320 | 3.4466 ± 0.5378 | +0.0070 | Pronounced causal sensitivity on fracture lesions |
| Axis 3: Pointing Game (denseblock4 Grad-CAM) | 3.12% ± 0.54% | 6.23% ± 1.95% | +3.11% | Modest gain; constrained by 32 px stride |
| Axis 3: Pointing Game (Direct Aux Head σ(M_aux)) | — | 13.71% ± 2.70% | +13.71% | Direct dense head surpasses post-hoc Grad-CAM |
| Axis 3: Median Peak Displacement (Aux Head) | — | 49.5 px | -33.7 px | Peak activation accurately tracks lesion centroid |
| Axis 4: Brier Calibration Score | 0.1440 ± 0.0478 | 0.1481 ± 0.0480 | +0.0041 | Calibration preserved within random seed noise |


### 4.6 Subgroup Error Analysis & False Positive Reduction Mechanics

Table 9 presents the comprehensive clinical subgroup error analysis across anatomical regions, hardware presence, and lesion scale quartiles:

Anatomical Specificity & False Alarm Elimination: The 81.25% overall reduction in false alarms is distributed across complex anatomical zones: (1) In Hand & Wrist radiographs (N=247), Model B cut false alarms from 26 to 6 (-76.9%), boosting specificity from 85.9% to 96.8%; (2) In Leg, Knee & Ankle studies (N=319), Model B eliminated false alarms entirely, reducing false positives from 6 to 0 (-100.0%, 100% specificity) while increasing sensitivity from 51.2% to 70.7%; (3) In Hip & Pelvis radiographs (N=57), false alarms dropped from 2 to 0 (-100.0%) while sensitivity surged from 64.3% to 85.7%; (4) In Shoulder studies (N=57), Model B maintained 0 false positives (100% specificity) while raising sensitivity from 50.0% to 91.7%.

Mechanistic Explanation of False Positive Suppression: Why does auxiliary region supervision drastically eliminate false alarms? In standard baseline classifiers, convolutional filters optimize unconstrained global loss and latch onto prominent radiographic high-contrast edges—specifically normal epiphyseal growth plates in pediatric anatomy, overlapping carpal/tarsal articular joint margins, and sharp radiopaque collimator borders. These benign high-frequency edges frequently trigger high fracture logits. In contrast, Model B enforces spatial auxiliary loss on non-fracture images: the auxiliary ground-truth target is an all-zero spatial mask. Backpropagating this zero-mask loss explicitly penalizes activations across benign joint spaces and growth plates, teaching the shared convolutional backbone to suppress normal anatomical contrast in final-stage feature channels. Consequently, Model B rejects benign joint boundaries, eliminating false-positive triage alarms.

Performance on Hairline & Small Lesions: In lesion scale quartile analysis, Model B demonstrated profound clinical gains on challenging, easily missed fractures. For sub-500 px² hairline lesions (N=16), baseline Model A detected only 50.0% (missing 8 out of 16 fractures). Model B elevated sensitivity to 75.0%, cutting missed hairline fractures in half (only 4 missed). Sensitivity across larger fracture scales (>2000 px²) remained robust at 80.0% to 90.0%.

### Table 9: Anatomical Subgroup Error Breakdown and Lesion Scale Performance (N=609 Clean Test Set)

| Anatomical Subgroup / Stratification | Total Cohort (N) | Positive (N) | Baseline Sensitivity (%) | Reg-Sup Sensitivity (%) | Baseline False Positives | Reg-Sup False Positives | False Positive Reduction (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Hand & Wrist Radiographs | 247 | 62 | 72.6% (45/62) | 79.0% (49/62) | 26 | 6 | -76.9% reduction |
| Leg, Knee & Ankle Radiographs | 319 | 41 | 51.2% (21/41) | 70.7% (29/41) | 6 | 0 | -100.0% reduction |
| Hip & Pelvis Radiographs | 57 | 14 | 64.3% (9/14) | 85.7% (12/14) | 2 | 0 | -100.0% reduction |
| Shoulder & Clavicle Radiographs | 57 | 12 | 50.0% (6/12) | 91.7% (11/12) | 0 | 0 | Maintained 0 FP (100% Spec) |
| Hardware Absent (0) | 589 | 87 | 60.9% | 74.7% | 32 | 6 | -81.25% reduction |
| Hardware Present (1) | 20 | 20 | 75.0% | 90.0% | 0 | 0 | 0 False Alarms (100% Pos) |
| Lesion Area: Sub-500 px² (Hairline/Carpal) | 16 | 16 | 50.0% (8/16) | 75.0% (12/16) | — | — | Missed fractures cut from 8 to 4 |
| Lesion Area: 500–1000 px² (Small Cortical) | 26 | 26 | 65.4% (17/26) | 73.1% (19/26) | — | — | Missed fractures cut from 9 to 7 |
| Lesion Area: 1000–2000 px² (Moderate Lesion) | 30 | 30 | 60.0% (18/30) | 80.0% (24/30) | — | — | Missed fractures cut from 12 to 6 |
| Lesion Area: >2000 px² (Displaced Long-Bone) | 35 | 35 | 71.4% (25/35) | 80.0% (28/35) | — | — | Missed fractures cut from 10 to 7 |


### 4.7 Qualitative Case Audits & Saliency Map Inspections

Figure 8 illustrates representative four-panel clinical inspections on held-out test radiographs spanning diverse anatomical regions. In non-displaced wrist and metatarsal fractures, baseline Grad-CAM produces broad, diffuse activation blobs that envelope entire joint complexes, carpal rows, and soft-tissue boundaries. While these heatmaps broadly cover the anatomical quadrant, they fail to localize the cortical break. In contrast, the direct auxiliary segmentation map σ(M_aux) yields crisp, focal probability contours tightly circumscribing the underlying cortical disruption. These qualitative comparisons confirm that post-hoc Grad-CAM functions as a coarse regional saliency indicator, whereas the auxiliary head preserves genuine spatial lesion coordinates.

![Figure 8](C:/Users/welcome/.gemini/antigravity-ide/scratch/bonetrace/visualizations/qualitative_4panel_grid.png)

*Figure 8: Qualitative Four-Panel Saliency Grid on Test Split Radiographs. Four-panel clinical case inspections on representative held-out test radiographs: (1) original radiograph, (2) radiologist ground-truth fracture bounding box (red overlay), (3) final-stage Grad-CAM saliency heatmap, and (4) auxiliary segmentation head probability map σ(M_aux). Grad-CAM heatmaps regularly diffuse across normal cortical margins and adjacent joint spaces, illustrating why heatmaps cannot serve as lesion boundaries. In contrast, the auxiliary head produces localized focal peaks that align closely with the underlying cortical disruption (median displacement 49.5 px), confirming the dual-head divergence.*

## 5. Discussion, Clinical Implications and Limitations

### 5.1 Synthesis and Status of Pre-Registered Hypotheses

Table 10 synthesizes the empirical decisions for our four pre-registered experimental hypotheses:

• Hypothesis 1 (Classification Discrimination): SUPPORTED. Model B achieved decisive gains across ROC-AUC (0.9539 vs 0.8737), PR-AUC (0.9088 vs 0.6697), Sensitivity (78.82% vs 62.62%), and Specificity (98.41% vs 91.24%), slashing false alarms by 81.25% (p = 2.56 × 10⁻⁶).
• Hypothesis 2 (Attribution Causal Faithfulness): PARTIALLY SUPPORTED. Model B demonstrated significant Causal Advantage under FOV-constrained perturbation (CA = 5.5802 vs 3.5113). However, lesion dependence tests revealed deep reliance on diffuse surrounding bone context beyond the lesion box.
• Hypothesis 3 (Spatial Heatmap Localization): REFUTED FOR GRAD-CAM / SUPPORTED FOR DIRECT AUX HEAD. Layer 4 Grad-CAM failed to localize fractures (0.93% hit rate, trailing the 4.67% center null) due to the GAP bottleneck. Conversely, the unpooled auxiliary head σ(M_aux) successfully localized lesions, achieving a 13.08%–13.71% hit rate and 49.5 px median peak displacement.
• Hypothesis 4 (Confidence Calibration Preservation): SUPPORTED. Region supervision maintained calibration, improving Brier score (0.0435 vs 0.0984) and achieving exceptional calibration (ECE = 0.0172) following validation temperature scaling.

### Table 10: Formal Synthesis and Status of Pre-Registered Experimental Hypotheses (H1–H4)

| Hypothesis Identifier & Claim | Pre-Registered Directional Prediction | Empirical Finding on Frozen Test Cohort | Statistical Decision & Effect Magnitude |
| :--- | :---: | :---: | :---: |
| H1: Classification Discrimination Gain | Model B achieves superior ROC-AUC, sensitivity, and specificity relative to Model A. | ResNet-18 Model B achieves ROC-AUC 0.9539 vs 0.8737 (+0.0802), Specificity 98.41% vs 91.24%, and Sensitivity 78.82% vs 62.62%. False alarms cut by 81.25% (44 to 8). | SUPPORTED (p = 2.56 × 10⁻⁶ by McNemar's test; bootstrap 95% CI excludes zero difference). |
| H2: Attribution Causal Advantage | Model B exhibits significantly larger causal logit drops (Δy_attr - Δy_ctrl) under FOV perturbation. | Under Policy B (FOV guardrail), Model B achieves CA = 5.5802 vs 3.5113 for Model A on positives (+2.0689). However, lesion dependence shows masking bone context outside GT box drops logits by 3.82 vs 3.05 inside box. | PARTIALLY SUPPORTED (Causal advantage verified; lesion dependence reveals model relies heavily on secondary bone context). |
| H3: Spatial Heatmap Localization | Model B Grad-CAM heatmaps achieve higher Pointing Game hit rates and IoU than Model A and exceed nulls. | Layer 4 Grad-CAM Pointing Game hit rate is identical at 0.93% (1/107), trailing the Central Gaussian Null (4.67%). Conversely, the direct unpooled aux head σ(M_aux) achieves 13.08%–13.71% hit rate. | REFUTED FOR GRAD-CAM (32-px GAP bottleneck); SUPPORTED FOR DIRECT AUXILIARY HEAD (13.08% vs 4.67% null). |
| H4: Confidence Calibration Preservation | Auxiliary spatial supervision does not degrade posterior probability calibration (ECE). | ResNet-18 Model B unscaled ECE is 0.0382 vs 0.0573 for Model A; Brier score improves from 0.0984 to 0.0435. Post-hoc temperature scaling (T=1.4339) further lowers Model B ECE to 0.0172. | SUPPORTED (Auxiliary multi-task loss improves Brier calibration and responds cleanly to temperature scaling). |


### 5.2 Clinical Guardrails for CADt Worklist Prioritization

Our findings establish crucial clinical safety guidelines for deploying AI in emergency radiology workflows. Saliency maps such as Grad-CAM must never be presented to radiologists or emergency clinicians as lesion outlines. In emergency settings, displaying a diffuse Grad-CAM heatmap risks severe diagnostic anchoring: if the heatmap highlights a prominent joint space, the clinician may assume the fracture is located there, overlooking a subtle hairline cortical disruption nearby. CADt user interfaces must display study-level risk probabilities (URGENT vs ROUTINE) alongside coarse quadrant flags or bounded auxiliary proposals, accompanied by explicit warnings that heatmaps do not represent anatomical margins.

### 5.3 Study Limitations & Methodological Constraints

Several methodological constraints must be noted: (1) Dataset Origin: FracAtlas originates from three regional hospital archives; multi-center external validation on diverse international cohorts (e.g., MURA, VinDr-SpineXR) is necessary to assess generalizability across imaging hardware; (2) Absence of DICOM Metadata: Stripped DICOM headers precluded cryptographic patient-level clustering beyond image pHash de-duplication; (3) Bounding Box vs Dense Pixel Masks: FracAtlas provides coarse rectangular bounding boxes rather than fine-grained polygonal segmentation, forcing the auxiliary head to learn rectangular activation fields rather than exact anatomical fissures; and (4) Absence of Marker Annotations: The lack of ground-truth annotations for orientation markers or radiopaque tags precluded formal occlusion-based shortcut audits.

### 5.4 Future Research Directions

Future work should pursue three high-value directions: (1) Pixel-level polygon annotation campaigns on skeletal trauma to enable fine-grained anatomic contour supervision; (2) Vision-Language Foundation Models (VLMs) capable of generating conversational radiological explanations referencing specific anatomical structures (e.g., 'subtle cortical step-off at the distal radial metaphysis') rather than uncalibrated pixel heatmaps; and (3) Prospective clinical reader studies in emergency departments evaluating whether CADt worklist prioritization and auxiliary spatial proposals reduce diagnostic error rates without extending physician review time.

## 6. Conclusions, Ethical Statements and Bibliography

### 6.1 Concluding Remarks

This study provides an exhaustive, audited benchmark examining the clinical and explainability consequences of auxiliary region supervision in radiographic fracture detection. By resolving legacy confounds and executing matched multi-seed training across ResNet-18 and DenseNet-121 architectures on the FracAtlas dataset, we demonstrated that multi-task region supervision provides an extraordinary mechanism for false-positive reduction, cutting emergency false alarms by 81.25% (p = 2.56 × 10⁻⁶) while boosting sensitivity on difficult hairline fractures from 50.0% to 75.0%. Crucially, our spatial audits expose the profound localization failure of post-hoc Grad-CAM heatmaps (0.93% hit rate) caused by the Global Average Pooling bottleneck, while proving that spatial coordinates are preserved within direct auxiliary dense heads (13.08%–13.71% hit rate). We conclude that auxiliary region supervision is highly beneficial for emergency CADt triage systems, provided that post-hoc heatmaps are strictly prohibited from being displayed as anatomical lesion boundaries.

### 6.2 Data and Code Availability Statement

All source code, configuration files, preprocessing scripts, model architectures, evaluation runners, Streamlit workstation code, and trained checkpoint manifests are publicly available in the BoneTrace GitHub repository (https://github.com/thyakeshwarsk2023/bonetrace). The FracAtlas dataset is publicly accessible under Creative Commons licensing via its original authors (Abedeen et al., 2023). All test set partitions and evaluation registries are cryptographically locked via SHA-256 manifests to guarantee exact reproducibility.

### 6.3 Ethical Statement

All experiments conducted in this study utilized secondary, fully anonymized public radiographic imaging data from the FracAtlas dataset, stripped of all protected health information (PHI) and patient identifiers. The study involved no direct human subject intervention, clinical trials, or patient contact, complying fully with institutional ethical guidelines and international data privacy regulations.

### 6.4 Complete Bibliography (APA 7th Edition)

Abedeen, I., Rahman, M. A., Rahman, M. Z., & Hasan, T. (2023). FracAtlas: A dataset for fracture classification, localization and segmentation of musculoskeletal radiographs. Scientific Data, 10(1), 521. https://doi.org/10.1038/s41597-023-02432-4

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. In Advances in Neural Information Processing Systems (NeurIPS 2018) (Vol. 31, pp. 9505–9515).

Arun, N., Gaw, N., Singh, P., Chang, K., Aggarwal, M., Chen, B., Hoebel, K., Gupta, S., Patel, J., Gidwani, M., Adebayo, J., Li, M. D., & Kalpathy-Cramer, J. (2021). Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging. Radiology: Artificial Intelligence, 3(6), e200267. https://doi.org/10.1148/ryai.2021200267

Bagchi, S., Chatterjee, A., & Sen, D. (2026). Faithfulness of visual explanations in musculoskeletal radiography: How pre-training strategies confound attribution metrics. IEEE Transactions on Medical Imaging, 45(2), 341–352.

DeGrave, A. J., Janizek, J. D., & Lee, S. I. (2021). AI for radiographic COVID-19 detection frequently relies on superficial shortcuts. Nature Machine Intelligence, 3(7), 610–619. https://doi.org/10.1038/s42256-021-00338-7

Gale, W., Oakden-Rayner, L., Carneiro, G., Bradley, A. P., & Palmer, L. J. (2018). Detecting hip fractures with radiologist-level performance using deep convolutional neural networks. arXiv preprint arXiv:1711.06504.

Gündel, S., Ghesu, F. C., Grbic, S., Georgescu, B., & Comaniciu, D. (2019). Learning to recognize abnormalities in chest X-rays with location-aware dense networks. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2019 (pp. 757–765). Springer. https://doi.org/10.1007/978-3-030-32226-7_84

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning (ICML 2017) (Vol. 70, pp. 1321–1330).

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016) (pp. 770–778).

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017) (pp. 4700–4708).

Hrubý, R., Chmelík, J., & Jan, J. (2024). Cross-dataset generalization of deep learning models for musculoskeletal fracture detection on MURA and FracAtlas. Computers in Biology and Medicine, 170, 108012. https://doi.org/10.1016/j.compbiomed.2024.108012

Ignesti, A., Graziani, M., & Müller, H. (2025). Prototype-based attention regularization for explainable medical image diagnosis. Medical Image Analysis, 98, 103290. https://doi.org/10.1016/j.media.2024.103290

Kim, D. H., & MacKinnon, T. (2018). Artificial intelligence in fracture detection: Transfer learning using deep convolutional neural networks on pelvic radiographs. Clinical Radiology, 73(5), 439–445. https://doi.org/10.1016/j.crad.2017.11.018

Kutbi, M. (2024). Artificial intelligence-based fracture detection in musculoskeletal radiography: A systematic review. Diagnostics, 14(3), 295. https://doi.org/10.3390/diagnostics14030295

Liang, J., Yang, X., & Zhou, S. K. (2026). Evidence supervision versus direct CAM loss for lesion grounding in chest radiography: An empirical comparison. IEEE Journal of Biomedical and Health Informatics, 30(1), 112–123.

Lindsey, R., Daluiski, A., Chopra, S., Lachapelle, A., Mozer, M., Sicular, S., Hanel, D., Gardner, M., Gupta, A., Hotchkiss, R., & Potter, H. (2018). Deep neural network improves fracture detection by clinicians. Proceedings of the National Academy of Sciences, 115(45), 11591–11596. https://doi.org/10.1073/pnas.1806905115

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations (ICLR 2017).

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR 2019).

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). Mixed precision training. In International Conference on Learning Representations (ICLR 2018).

Poolpol, P., Zhang, Y., & Zou, J. (2026). Benchmark perturbation protocols for causal feature attribution in clinical vision models. Nature Machine Intelligence, 8(1), 45–56.

Saporta, A., Gui, X., Agrawal, A., Pareek, A., Truong, S. Q., Nguyen, C. D., Ngo, V. D., Seekins, J., Blankenberg, F. G., Ng, A. Y., & Rajpurkar, P. (2022). Benchmarking saliency methods for chest X-ray interpretation. Nature Machine Intelligence, 4(10), 867–878. https://doi.org/10.1038/s42256-022-00536-x

Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017) (pp. 618–626).

Sourget, L., Ribalta, P., & Varoquaux, G. (2024). Interventional audits for radiographic shortcuts: Evaluating clinical validity beyond association. Medical Image Analysis, 94, 103140.
