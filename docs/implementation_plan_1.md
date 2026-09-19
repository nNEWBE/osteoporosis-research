# Implementation Plan: ResNet101 + DenseNet201 Fuzzy Fusion for Knee Osteoporosis Detection

Design and implement a complete, research-grade, publication-ready deep learning and fuzzy logic hybrid system for classifying Normal vs Osteoporosis knee radiographs, specifically optimized to run within the resource constraints of Google Colab's Free Tier (T4 GPU, 15GB VRAM, 12GB system RAM).

---

## User Review Required

> [!IMPORTANT]
> **CRITICAL DATASET AUDIT FINDINGS (Discovered during initial scan):**
> 1. **Intra-class Duplicates:** Out of 780 `Normal` images, only **384 are unique** (396 are identical duplicate files). Out of 793 `Osteoporosis` images, only **395 are unique** (398 are identical duplicate files).
> 2. **Cross-class Contradictions:** Exactly **24 identical image files** exist in *both* `Normal` and `Osteoporosis` folders (48+ conflicting files total).
> 3. **Impact:** If trained raw without deduplication, a random 70/15/15 split will leak identical images between train and test, producing falsely inflated 99%+ accuracy that fails clinically.
>
> **Proposed Action:** The pipeline will include an automated **Data Hygiene & Audit Layer** that filters out contradictory pairs and bitwise duplicates, preserving **~731 pristine unique images** for the honest 70/15/15 stratified split. We will document both raw and deduplicated protocols so you can explicitly report this dataset phenomenon in your thesis/paper.

---

## Proposed System Architecture

```text
                                 Input Knee Radiograph (224x224)
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     │                                                     │
                     ▼                                                     ▼
         Pretrained ResNet-101                                 Pretrained DenseNet-201
         (Transfer Learning)                                   (Transfer Learning)
                     │                                                     │
                     ▼                                                     ▼
         P_ResNet(Osteoporosis)                                P_Dense(Osteoporosis)
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                                ▼
                                    Fuzzy Inference Engine
                              ┌───────────────────────────────────┐
                              │ 1. Fuzzification (Low/Med/High)   │
                              │ 2. Rule Base Evaluation (Mamdani) │
                              │ 3. Defuzzification (Centroid/COG) │
                              └───────────────────────────────────┘
                                                │
                                                ▼
                                Final Osteoporosis Confidence P_final
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
       Normal (P_final < Threshold)                         Osteoporosis (P_final >= Threshold)
```

---

## Proposed Changes & Deliverables

### 1. Documentation Suite (`docs/`)
Comprehensive markdown documentation covering the research methodology, clinical context, and Colab execution guide.

#### [NEW] [01_methodology_and_pipeline.md](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/docs/01_methodology_and_pipeline.md)
- Complete theoretical formulation of ResNet101, DenseNet201, and Fuzzy Logic Ensembling.
- Mathematical equations for triangular/trapezoidal membership functions, t-norm operators, and Centroid defuzzification.
- Contrast between traditional Softmax averaging vs Fuzzy Inference under uncertainty.

#### [NEW] [02_google_colab_guide.md](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/docs/02_google_colab_guide.md)
- Step-by-step tutorial on running on Colab Free Tier (T4 GPU).
- Google Drive dataset mounting, ZIP unzipping, execution instructions.
- Memory optimization tricks (Mixed Precision AMP, sequential backbone training to stay <15GB VRAM, clearing GPU cache).

#### [NEW] [03_limitations_and_clinical_aspects.md](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/docs/03_limitations_and_clinical_aspects.md)
- Detailed breakdown of the audit discovery (50.5% duplicate rate & 24 cross-class conflicting pairs).
- Clinical implications of excluding Osteopenia (boundary conditions between normal T-score > -1.0, osteopenia -1.0 to -2.5, and osteoporosis < -2.5).
- 2D projection radiograph limitations vs 3D DXA (Dual-energy X-ray Absorptiometry) gold standard.
- Need for multi-center external validation and patient-level de-identification.

#### [NEW] [04_evaluation_and_metrics.md](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/docs/04_evaluation_and_metrics.md)
- Formal definitions of medical diagnostic metrics: Sensitivity (Recall), Specificity, Precision (PPV), Negative Predictive Value (NPV), F1-Score, and ROC-AUC.
- Explanation of the 1,000-iteration Bootstrapping protocol for calculating 95% Confidence Intervals [CI_lower, CI_upper].
- Comparative benchmark template: ResNet101 alone vs DenseNet201 alone vs Simple Average vs Fuzzy Fusion.

---

### 2. Core Python Implementation (`src/`)
Modular, clean, testable code for local inspection and modular execution.

#### [NEW] [src/audit_and_clean.py](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/src/audit_and_clean.py)
- MD5/SHA256 image hasher to identify exact duplicates and cross-class conflicting labels.
- Stratified 70/15/15 train/val/test split generator saving clean manifest CSVs (`train.csv`, `val.csv`, `test.csv`).

#### [NEW] [src/dataset.py](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/src/dataset.py)
- Custom PyTorch `KneeXRayDataset` supporting CSV manifests or directories.
- Training augmentations: RandomHorizontalFlip, RandomRotation(±10°), ColorJitter (contrast/brightness), RandomAffine.
- Validation/Test transforms: Deterministic resize to 224x224 and ImageNet normalization.

#### [NEW] [src/models.py](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/src/models.py)
- `build_resnet101(pretrained=True, num_classes=2, dropout=0.3)`
- `build_densenet201(pretrained=True, num_classes=2, dropout=0.3)`
- Layer freezing / two-stage fine-tuning capabilities.

#### [NEW] [src/fuzzy_engine.py](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/src/fuzzy_engine.py)
- Self-contained vectorized Fuzzy Inference System (Pure Python/NumPy with optional Torch support).
- Zero dependency issues on Colab (no fragile C compiler or outdated scikit-fuzzy installation bugs).
- Configurable membership shapes (Low, Medium, High) and complete 9-rule Mamdani & Sugeno engines with Centroid defuzzification.

#### [NEW] [src/train.py](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/src/train.py)
- Mixed-Precision training loop (`torch.cuda.amp.autocast` + `GradScaler`).
- Memory management (`gc.collect()` and `torch.cuda.empty_cache()` after each epoch).
- ReduceLROnPlateau, EarlyStopping, and best model checkpointing.
- Sequential training driver: trains ResNet101, saves weights, unloads from VRAM, then trains DenseNet201.

#### [NEW] [src/evaluate.py](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/src/evaluate.py)
- Comprehensive test runner:
  - Generates probabilities for ResNet101, DenseNet201, Simple Average, and Fuzzy Fusion.
  - Computes Accuracy, Sensitivity, Specificity, Precision, F1-Score, ROC-AUC.
  - Computes 95% Confidence Intervals via 1000-fold bootstrap.
  - Exports publication-ready plots: ROC curves, Confusion Matrices, Probability Distribution Histograms, and Fuzzy Surface plots.

---

### 3. Turnkey Google Colab Notebook
#### [NEW] [Osteoporosis_Fuzzy_Fusion_Colab.ipynb](file:///c:/Users/Shuvo%20Debnath/VsCode/Research%20Projects/Fardin%20Osteoporosis/Osteoporosis_Fuzzy_Fusion_Colab.ipynb)
- Single-click, self-contained Jupyter notebook formatted specifically for Google Colab Free Tier.
- Automated GPU verification, dataset loading (Drive or direct zip upload), training, fuzzy fusion tuning, statistical evaluation, and plot generation.

---

## Verification Plan

### Automated / Code Quality Verification
1. **Dataset Audit Script**: Run `src/audit_and_clean.py` locally to verify duplicate detection, conflicting label filtering, and exact 70/15/15 train/val/test CSV generation.
2. **Fuzzy Engine Unit Tests**: Run synthetic test cases through `src/fuzzy_engine.py` (e.g. [0.0, 0.0] -> Low, [1.0, 1.0] -> High, conflicting inputs [0.9, 0.1] -> expected uncertainty resolution) to verify Mamdani and Sugeno math.
3. **Colab Notebook Syntax & Structure Validation**: Validate `.ipynb` JSON structure, ensuring all code blocks and markdown cells are properly formatted.

### Manual Verification
- Review the 4 documentation files in `docs/` to ensure all research justifications, mathematical formulas, clinical caveats, and Colab usage steps are clear.
