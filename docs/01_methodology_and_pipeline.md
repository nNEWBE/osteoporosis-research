# 01. Methodology and Pipeline Architecture

## 1. Executive Overview

This document details the architectural design, mathematical formulation, and deep learning pipeline for automated binary classification of knee radiographs into **Normal** and **Osteoporosis** states. The proposed framework implements a dual-stream Convolutional Neural Network (CNN) feature extraction and decision network (ResNet-101 and DenseNet-201) coupled with a **Fuzzy Inference System (FIS)**.

Rather than relying on naive linear weighting or simple probability averaging, the fuzzy fusion layer explicitly models model uncertainty, inter-model conflict, and non-linear diagnostic confidence transitions.

```text
                                  Knee Radiograph Input
                                            │
                             ┌──────────────┴──────────────┐
                             │                             │
                             ▼                             ▼
                    ┌─────────────────┐           ┌─────────────────┐
                    │   ResNet-101    │           │  DenseNet-201   │
                    │ Transfer Learn. │           │ Transfer Learn. │
                    └────────┬────────┘           └────────┬────────┘
                             │                             │
                     P_ResNet(Osteo)               P_Dense(Osteo)
                             │                             │
                             └──────────────┬──────────────┘
                                            ▼
                                ┌───────────────────────┐
                                │ Fuzzy Inference Engine│
                                │  - Tri-Membership     │
                                │  - Mamdani Rule Base  │
                                │  - Centroid Defuzz.   │
                                └───────────┬───────────┘
                                            ▼
                                 Final Confidence P_final
                                            │
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
                   Normal (P_final < τ)         Osteoporosis (P_final ≥ τ)
```

---

## 2. Backbone Architectures

### 2.1 ResNet-101 (Residual Networks)
- **Core Mechanism**: Residual learning solves the degradation problem in deep networks via identity shortcut connections:
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
  where $\mathbf{x}$ and $\mathbf{y}$ are the input and output vectors of the residual layer $\mathcal{F}$.
- **Clinical Rationale**: ResNet-101's deep 101-layer bottleneck architecture excels at extracting macro-architectural bone contours, cortical thinning, and global geometric deformities around the femoral condyles and tibial plateau.
- **Transfer Learning Setup**:
  - Pretrained on ImageNet-1K.
  - Final classification layer ($1000$ classes) replaced with:
    $$\text{Dropout}(p=0.3) \to \text{Linear}(2048 \to 256) \to \text{ReLU} \to \text{BatchNorm1d} \to \text{Linear}(256 \to 2)$$
  - Training strategy: Two-stage fine-tuning (linear warm-up of classifier head followed by unfreezing of deeper residual blocks).

### 2.2 DenseNet-201 (Densely Connected Convolutional Networks)
- **Core Mechanism**: In DenseNet, each layer receives feature maps from all preceding layers as direct inputs:
  $$\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}])$$
  where $[\dots]$ represents concatenation of feature maps from stages $0$ through $\ell-1$.
- **Clinical Rationale**: Because feature maps are reused across all depths, DenseNet-201 has exceptional sensitivity to fine-grained micro-textural bone trabecular patterns and subtle subchondral bone mineral loss that residual pooling can sometimes smooth over.
- **Transfer Learning Setup**:
  - Pretrained on ImageNet-1K.
  - Classifier replaced with:
    $$\text{Dropout}(p=0.3) \to \text{Linear}(1920 \to 256) \to \text{ReLU} \to \text{BatchNorm1d} \to \text{Linear}(256 \to 2)$$

---

## 3. The Fuzzy Fusion Mechanism

### 3.1 Motivation
Standard ensembling methods (e.g., Simple Averaging or Logistic Meta-Classifiers) treat class probabilities linearly:
$$P_{\text{avg}} = \frac{P_{\text{ResNet}} + P_{\text{DenseNet}}}{2}$$
In clinical diagnostics, when one model is highly confident ($P=0.95$) and the other is ambivalent ($P=0.52$), simple averaging drops the probability to $0.735$, potentially crossing under strict diagnostic thresholds. Conversely, when both models exhibit slight uncertainty ($P_1 = 0.55, P_2 = 0.58$), a simple average fails to capture that neither model has detected definitive pathological markers.

Fuzzy logic introduces **linguistic variables** and **approximate reasoning** to handle diagnostic vagueness and conflicting predictions.

---

### 3.2 Fuzzification: Membership Functions
Let $x_1 = P_{\text{ResNet}}(\text{Osteo}) \in [0, 1]$ and $x_2 = P_{\text{DenseNet}}(\text{Osteo}) \in [0, 1]$.
We define three linguistic terms for each input: $\mathcal{T} = \{\text{Low (L)}, \text{Medium (M)}, \text{High (H)}\}$.

1. **Low (L) Membership $\mu_L(x)$** (Trapezoidal / Half-Triangular):
   $$\mu_L(x) = \begin{cases} 
   1 & x \le 0.15 \\
   \frac{0.40 - x}{0.40 - 0.15} & 0.15 < x < 0.40 \\
   0 & x \ge 0.40
   \end{cases}$$

2. **Medium (M) Membership $\mu_M(x)$** (Triangular):
   $$\mu_M(x) = \begin{cases} 
   0 & x \le 0.25 \\
   \frac{x - 0.25}{0.50 - 0.25} & 0.25 < x \le 0.50 \\
   \frac{0.75 - x}{0.75 - 0.50} & 0.50 < x < 0.75 \\
   0 & x \ge 0.75
   \end{cases}$$

3. **High (H) Membership $\mu_H(x)$** (Trapezoidal / Half-Triangular):
   $$\mu_H(x) = \begin{cases} 
   0 & x \le 0.60 \\
   \frac{x - 0.60}{0.85 - 0.60} & 0.60 < x < 0.85 \\
   1 & x \ge 0.85
   \end{cases}$$

---

### 3.3 Rule Base (Mamdani Model)
The Mamdani knowledge base consists of 9 intuitive medical rules reflecting consensus and conflict resolution:

| Rule # | ResNet-101 Input ($x_1$) | DenseNet-201 Input ($x_2$) | Output Consequent ($y$) | Clinical Interpretation |
|---|---|---|---|---|
| **R1** | Low | Low | **Low (Normal)** | Both models agree: pristine bone density |
| **R2** | Low | Medium | **Low (Normal)** | Moderate texture, normal architecture |
| **R3** | Low | High | **Medium (Uncertain)** | Severe conflict: DenseNet sees micro-pore, ResNet disagrees |
| **R4** | Medium | Low | **Low (Normal)** | Conservative screening posture |
| **R5** | Medium | Medium | **Medium (Uncertain)** | Borderline osteopenic transition |
| **R6** | Medium | High | **High (Osteoporosis)** | DenseNet confirms micro-loss, ResNet suspects macro-loss |
| **R7** | High | Low | **Medium (Uncertain)** | Conflict: ResNet detects cortical thinning, DenseNet normal |
| **R8** | High | Medium | **High (Osteoporosis)** | Strong primary indicator of osteoporosis |
| **R9** | High | High | **High (Osteoporosis)** | Unanimous positive identification |

---

### 3.4 Inference Engine & Aggregation
For each rule $i \in \{1, \dots, 9\}$, the rule firing strength $w_i$ is computed using the **minimum t-norm** (Mamdani implication):
$$w_i = \min\Big(\mu_{A_i}(x_1),\, \mu_{B_i}(x_2)\Big)$$

The individual output fuzzy sets are aggregated using the **maximum s-norm**:
$$\mu_{\text{agg}}(y) = \max_{i=1}^{9} \Big( \min(w_i, \mu_{C_i}(y)) \Big)$$
where $\mu_{C_i}(y)$ is the membership function of the consequent linguistic term on the output universe $Y = [0, 1]$.

---

### 3.5 Defuzzification: Center of Gravity (Centroid)
The aggregated fuzzy set is defuzzified into a single crisp probability $P_{\text{final}} \in [0, 1]$ via the continuous Centroid / Center of Gravity (COG) operator:
$$P_{\text{final}} = \frac{\int_0^1 y \cdot \mu_{\text{agg}}(y) \, dy}{\int_0^1 \mu_{\text{agg}}(y) \, dy}$$
In practice, this integral is discretized across $N=101$ sample points over $[0, 1]$:
$$P_{\text{final}} = \frac{\sum_{k=0}^{100} y_k \cdot \mu_{\text{agg}}(y_k)}{\sum_{k=0}^{100} \mu_{\text{agg}}(y_k)}$$

A clinical decision threshold $\tau = 0.50$ (tunable on validation ROC) converts $P_{\text{final}}$ into the binary prediction:
$$\hat{Y} = \begin{cases} \text{Osteoporosis} & P_{\text{final}} \ge \tau \\ \text{Normal} & P_{\text{final}} < \tau \end{cases}$$

---

## 4. End-to-End Experimental Protocol

1. **Dataset Sanitation**: Hash checking to remove intra-class duplicates and cross-class contradictions.
2. **Stratified Splitting**: 70% Train, 15% Validation, 15% Test.
3. **Training Phase 1**: ResNet-101 fine-tuning with cosine annealing and mixed precision.
4. **Training Phase 2**: DenseNet-201 fine-tuning under identical data partitions.
5. **Validation Phase**:
   - Extract validation probabilities from both models.
   - Calibrate fuzzy membership intervals.
   - Determine optimal classification threshold $\tau^*$.
6. **Testing Phase**:
   - Evaluate ResNet-101, DenseNet-201, Simple Average, and Fuzzy Fusion on the untouched test set.
   - Compute 95% Confidence Intervals via 1,000-sample bootstrapping.
