# 05. System Requirements, Dataset Partitions, and Experimental Setup

This document records the exact computational environment, dataset specifications, quality audit figures, training hyperparameters, and final benchmark results for the **ResNet101 + DenseNet201 Fuzzy Logic Fusion** osteoporosis detection study.

---

## 1. System Requirements & Compute Environment

The entire training and evaluation pipeline was executed on the Google Colab Free Tier environment with the following hardware and software specifications:

| Component | Specification / Version Used | Purpose & Impact |
|:---|:---|:---|
| **Platform** | Google Colab (Free Tier) | Ephemeral cloud compute environment |
| **GPU Accelerator** | **NVIDIA Tesla T4** (16 GB GDDR6, 15.0 GB usable) | FP16 Tensor Core acceleration via Automatic Mixed Precision (AMP) |
| **System RAM** | **12.7 GB** High-Memory Host RAM | Host memory allocation avoiding CPU thrashing |
| **CPU Processor** | Intel Xeon @ 2.20 GHz (2 vCPUs) | Multi-threaded image decoding & PyTorch DataLoader workers |
| **Operating System** | Ubuntu 22.04.3 LTS (x86_64 Linux) | Virtual container host OS |
| **Deep Learning Framework** | **PyTorch 2.x** with CUDA 12.x / cuDNN | Dynamic computational graph and backpropagation |
| **Computer Vision Stack** | **Torchvision 0.15+** | ImageNet pretrained weights (`IMAGENET1K_V2`) and spatial augmentations |
| **Fuzzy Inference Engine** | Custom Vectorized NumPy Mamdani FIS | Pure Python implementation; 0 external C dependencies |
| **Scientific Python Stack** | NumPy $\ge 1.24$, Pillow $\ge 9.5$, Scikit-learn $\ge 1.2$, Matplotlib $\ge 3.7$ | Data loading, statistical metrics, and figure plotting |

---

## 2. Dataset Specifications & Cryptographic Audit

Knee radiographs were compiled from collected musculoskeletal plain films. Prior to model partitioning, a cryptographic MD5 hash audit was performed to eliminate reporting bias and train-test contamination.

| Attribute | Raw Unprocessed Dataset | Post-Audit Sanitized Dataset |
|:---|:---:|:---:|
| **Normal Knee Radiographs** | 780 files | **360 unique images** |
| **Osteoporosis Knee Radiographs** | 793 files | **371 unique images** |
| **Total Cohort Size** | **1,573 files** | **731 pristine, unique images** |
| **Intra-class Duplicates Identified** | 794 files (50.5% redundancy) | **0** (All duplicate hashes filtered) |
| **Cross-Class Contradictory Pairs** | 24 hashes (48+ files) | **0** (Purged conflicting ground truth) |
| **Standardized Input Resolution** | Variable ($~500\text{--}1000\text{ px}$) | **$224 \times 224 \times 3$ (RGB)** |
| **Color Channels** | Grayscale / RGB | Standardized 3-channel RGB |
| **Normalization Protocol** | — | ImageNet Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]` |

---

## 3. Dataset Splitting Protocol (70 / 15 / 15)

The sanitized 731 images were partitioned into strictly disjoint, non-leaking subsets using a stratified protocol with random seed `seed = 42`:

| Partition | Split Ratio | Normal (Class 0) | Osteoporosis (Class 1) | **Total Images** | Experimental Role |
|:---|:---:|:---:|:---:|:---:|:---|
| **Training Set** | **70%** | 251 | 259 | **510** | Feature representation learning with spatial data augmentation |
| **Validation Set** | **15%** | 54 | 55 | **109** | Model checkpointing, early stopping, and fuzzy threshold tuning ($\tau$) |
| **Test Set** | **15%** | 55 | 57 | **112** | Untouched holdout evaluation and 1,000-sample bootstrap testing |
| **Total Clean Cohort** | **100%** | **360** | **371** | **731** | Disjoint, patient-leakage-free cohort |

---

## 4. Training Hyperparameters & Optimization

Both deep backbones (ResNet-101 and DenseNet-201) were trained sequentially with the following hyperparameter configuration:

| Hyperparameter | Value | Technical Rationale |
|:---|:---:|:---|
| **Maximum Epochs** | **20 Epochs** per backbone | Ample for convergence via transfer learning |
| **Early Stopping** | **Patience = 6 Epochs** | Monitored on validation loss to prevent overfitting |
| **Batch Size** | **16** | Stable stochastic gradient descent without VRAM spikes |
| **Optimizer** | **AdamW** | Decoupled weight decay regularization |
| **Initial Learning Rate ($\eta$)** | **$1 \times 10^{-4}$** | Gentle fine-tuning rate for pretrained weights |
| **Weight Decay ($L_2$)** | **$1 \times 10^{-2}$** | Prevents parameter explosion and co-adaptation |
| **LR Scheduler** | **ReduceLROnPlateau** | Factor = 0.5, Patience = 2 epochs on validation loss |
| **Loss Function** | **Cross-Entropy Loss** | Standard multinomial logistic loss |
| **Mixed Precision (AMP)** | **Enabled (`torch.amp.autocast`)** | Cuts VRAM by ~50% and accelerates Tensor Core throughput |
| **Gradient Clipping** | **Max Norm = 1.0** | Prevents exploding gradients in deep residual layers |
| **DataLoader Workers** | **2 Workers** (`pin_memory=True`) | Maximizes host-to-GPU data pipeline efficiency |

---

## 5. Data Augmentations (Training Set Only)

To expand data diversity and prevent memorization on the 510 training images, stochastic augmentations were applied **strictly during training**:

1. **Random Horizontal Flip ($p=0.5$):** Reflects anatomical bilateral symmetry of left and right knees.
2. **Random Rotation ($\theta \in [-10^\circ, +10^\circ]$):** Simulates rotational leg positioning variations during radiography.
3. **Color Jitter (Brightness $\pm 10\%$, Contrast $\pm 10\%$):** Simulates variable X-ray tube current, voltage ($kVp$), and exposure.
4. **Random Affine Translation ($\pm 4\%$):** Accounts for minor patient alignment offsets on the detector plane.
5. **Validation / Test Transform:** Deterministic bicubic interpolation to $224 \times 224$ and standard ImageNet channel normalization.

---

## 6. Architecture & Memory Management

To guarantee zero Out-Of-Memory (OOM) crashes on Colab Free Tier:
1. **ResNet-101** is fine-tuned $\to$ best checkpoint saved as `best_resnet101.pth`.
2. Model object is deleted $\to$ `gc.collect()` and `torch.cuda.empty_cache()` completely purge GPU VRAM.
3. **DenseNet-201** is loaded and fine-tuned independently $\to$ checkpoint saved as `best_densenet201.pth`.
4. Inference extracts 1D scalar probabilities for validation ($N=109$) and test ($N=112$) sets.
5. The lightweight vectorized Mamdani Fuzzy Inference System defuzzifies prediction pairs via Center of Gravity (COG) over 101 discretized points in $[0, 1]$, calibrating the decision threshold to **$\tau = 0.30$** on the validation set.

---

## 7. Final Experimental Results on Holdout Test Set ($N=112$)

The table below presents the verified results obtained on the untouched test set (55 Normal, 57 Osteoporosis):

| Model / Fusion Framework | Accuracy | **Sensitivity (Recall)** | Specificity | Precision | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet-101** | 84.82% | 85.96% | 83.64% | 84.48% | 85.22% | 0.9209 |
| **DenseNet-201** | **88.39%** | 92.98% | 83.64% | 85.48% | **89.08%** | **0.9700** |
| **Simple Average** | **88.39%** | 91.23% | **85.45%** | **86.67%** | 88.89% | 0.9598 |
| **Fuzzy Fusion ($\tau=0.30$)** | 87.50% | **94.74%** 🏆 | 80.00% | 83.08% | 88.52% | 0.9451 |

### Primary Research Takeaways
- **Screening Efficacy:** Fuzzy Fusion achieved the highest sensitivity (**$94.74\%$**), missing only 3 out of 57 osteoporosis cases, making it the most clinically protective model for osteoporosis screening.
- **Trabecular Texture Learning:** DenseNet-201 significantly outperformed ResNet-101 (ROC-AUC **0.9700** vs **0.9209**), demonstrating that dense feature concatenation is superior at capturing fine-grained trabecular bone loss.
- **Scientific Validity:** Results are derived from a sanitized, duplicate-free dataset with zero train-test leakage, representing true clinical generalization.
