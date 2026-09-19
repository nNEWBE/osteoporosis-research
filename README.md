# Hybrid ResNet101–DenseNet201 with Fuzzy Logic Fusion for Knee Osteoporosis Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning and fuzzy logic ensemble framework for automated binary classification of knee radiographs into **Normal** and **Osteoporosis** states. 

This repository pairs **ResNet-101** (macro-cortical feature extraction) and **DenseNet-201** (trabecular micro-texture extraction) with a **9-rule Mamdani Fuzzy Inference System (FIS)** to model inter-model uncertainty and maximize diagnostic screening sensitivity.

---

## 🌟 Key Highlights

1. **Highest Diagnostic Sensitivity (94.74%)**: 
   The proposed Fuzzy Fusion ($\tau=0.30$) achieved **$94.74\%$ Sensitivity**, outperforming standalone DenseNet-201 ($92.98\%$), Simple Averaging ($91.23\%$), and ResNet-101 ($85.96\%$) on the untouched holdout test set. In clinical screening, maximizing sensitivity is vital to prevent missed fragility fractures.
2. **Cryptographic Dataset Audit (Zero-Leakage Guarantee)**:
   Discovered that **50.4%** of the raw cohort comprised intra-class duplicate files, and **24 identical image files** had contradictory cross-class labels. By purging noise, we constructed a **pristine 731-image cohort** partitioned into an honest, non-leaking **70/15/15 stratified split**.
3. **Google Colab Free Tier Optimization**:
   Engineered with Automatic Mixed Precision (AMP), gradient clipping, and sequential backbone training with explicit VRAM purging, enabling complete training and evaluation on free **Tesla T4 GPUs (< 6 GB VRAM used)** in under 7 minutes.

---

## 🏗️ System Architecture

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
                              │ 2. 9-Rule Mamdani Inference       │
                              │ 3. Centroid Defuzzification (COG) │
                              └───────────────────────────────────┘
                                                │
                                                ▼
                                Final Osteoporosis Confidence P_final
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
       Normal (P_final < 0.30)                              Osteoporosis (P_final >= 0.30)
```

---

## 📊 Final Experimental Benchmark Results

Evaluated on the untouched holdout test set ($N=112$: 55 Normal, 57 Osteoporosis):

| Model / Fusion Framework | Accuracy | **Sensitivity (Recall)** | Specificity | Precision | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet-101** | 84.82% | 85.96% | 83.64% | 84.48% | 85.22% | 0.9209 |
| **DenseNet-201** | **88.39%** | 92.98% | 83.64% | 85.48% | **89.08%** | **0.9700** |
| **Simple Average** | **88.39%** | 91.23% | **85.45%** | **86.67%** | 88.89% | 0.9598 |
| **Fuzzy Fusion ($\tau=0.30$)** | 87.50% | **94.74%** 🏆 | 80.00% | 83.08% | 88.52% | 0.9451 |

---

## 📁 Repository Structure

```
.
├── Osteoporosis_Fuzzy_Fusion_Colab.ipynb    # Turnkey 1-Click Google Colab Notebook
├── requirements.txt                         # Environment dependencies
├── README.md                                # Project documentation
│
├── docs/                                    # Academic Documentation Suite
│   ├── 01_methodology_and_pipeline.md       # Full mathematical & architectural formulation
│   ├── 02_google_colab_guide.md             # Free-tier Colab execution & optimization guide
│   ├── 03_limitations_and_clinical_aspects.md# Audit analysis, DXA caveats, clinical scope
│   ├── 04_evaluation_and_metrics.md         # Diagnostic metrics & 1,000-bootstrap 95% CI protocol
│   └── 05_system_requirements_and_experimental_setup.md # Hardware, hyperparameters & partitions
│
├── src/                                     # Modular Python Codebase
│   ├── audit_and_clean.py                   # MD5 hasher, duplicate filter, and manifest generator
│   ├── dataset.py                           # PyTorch Dataset, transforms, and DataLoaders
│   ├── models.py                            # Custom ResNet-101 & DenseNet-201 architectures
│   ├── fuzzy_engine.py                      # Vectorized pure NumPy Mamdani & Sugeno FIS
│   ├── train.py                             # Sequential AMP mixed-precision training pipeline
│   └── evaluate.py                          # Multi-model statistical benchmarking & bootstrapping
│
├── manifests/                               # Stratified 70/15/15 Data Manifests
│   ├── train.csv                            # 510 clean images (251 Normal, 259 Osteo)
│   ├── val.csv                              # 109 clean images (54 Normal, 55 Osteo)
│   └── test.csv                             # 112 clean images (55 Normal, 57 Osteo)
│
└── results/                                 # Evaluation Output Records
    └── benchmark_results.json               # Final recorded experimental metrics
```

---

## 🚀 Quickstart Guide

### Option 1: Run in Google Colab (Recommended)
1. Upload `Osteoporosis_Fuzzy_Fusion_Colab.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Upload your dataset archive (`OS Collected Data.zip`) or mount Google Drive.
4. Click **Runtime -> Run all**.

### Option 2: Run Locally
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Audit dataset and generate clean manifests
python src/audit_and_clean.py --data_dir "./OS Collected Data"

# 4. Train backbones sequentially (saves VRAM)
python src/train.py --epochs 20 --batch_size 16

# 5. Evaluate and benchmark models
python src/evaluate.py
```

---

## 📜 Research Documentation

For detailed analysis, refer to the documentation in `docs/`:
- [Methodology & Architecture](docs/01_methodology_and_pipeline.md)
- [Colab Free Tier Guide](docs/02_google_colab_guide.md)
- [Limitations & Clinical Aspects](docs/03_limitations_and_clinical_aspects.md)
- [Statistical Evaluation & Metrics](docs/04_evaluation_and_metrics.md)
- [System Requirements & Hyperparameters](docs/05_system_requirements_and_experimental_setup.md)

---

## 📄 License
This project is licensed under the MIT License.
