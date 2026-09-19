# 04. Evaluation Protocols, Statistical Metrics, and Confidence Intervals

## 1. Primary Clinical Evaluation Metrics

In medical imaging research, accuracy alone is insufficient and often misleading. Our evaluation framework incorporates a full suite of diagnostic performance metrics:

### 1.1 Mathematical Formulations
Let:
- **True Positive (TP)**: Osteoporosis correctly identified as Osteoporosis.
- **True Negative (TN)**: Normal correctly identified as Normal.
- **False Positive (FP)**: Normal misclassified as Osteoporosis (Type I Error).
- **False Negative (FN)**: Osteoporosis misclassified as Normal (Type II Error).

| Metric | Formula | Clinical Interpretation |
|---|---|---|
| **Sensitivity (Recall / TPR)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | **Most critical metric for screening.** Measures the proportion of actual osteoporosis patients correctly flagged. High sensitivity minimizes missed fractures. |
| **Specificity (TNR)** | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | Measures how well the model avoids falsely labeling healthy subjects as osteoporotic, reducing unnecessary clinical alarm and costly DXA backlogs. |
| **Precision (PPV)** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Positive Predictive Value: When the system predicts osteoporosis, how often is it correct? |
| **Negative Predictive Value (NPV)** | $\frac{\text{TN}}{\text{TN} + \text{FN}}$ | When the model rules out osteoporosis, what is the probability the patient is truly normal? |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}}$ | Harmonic mean balancing precision and sensitivity. |
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$ | Overall fraction of correct predictions across both classes. |
| **ROC-AUC** | $\int_0^1 \text{TPR}(\tau) \, d(\text{FPR}(\tau))$ | Area under the Receiver Operating Characteristic curve across all discrimination thresholds $\tau \in [0, 1]$. |

---

## 2. Statistical Rigor: 95% Bootstrap Confidence Intervals

For publication in peer-reviewed journals (e.g. IEEE JBHI, Nature Scientific Reports, Elsevier BME), single-point estimates must be accompanied by **95% Confidence Intervals (95% CI)**.

### Bootstrapping Protocol
1. Given a test set $\mathcal{D}_{\text{test}} = \{(x_i, y_i)\}_{i=1}^N$ with $N$ samples.
2. For $b = 1, \dots, B$ (where $B = 1,000$ iterations):
   - Sample $N$ instances **with replacement** from $\mathcal{D}_{\text{test}}$ to form bootstrap sample $\mathcal{D}_b^*$.
   - Compute the metric score $M_b$ (e.g., Sensitivity, AUC, F1) on $\mathcal{D}_b^*$.
3. Sort the resulting distribution $[M_1, M_2, \dots, M_B]$.
4. The 95% Confidence Interval is extracted using the percentile method:
   $$\text{CI}_{95\%} = \left[ M_{(0.025 \cdot B)}, \; M_{(0.975 \cdot B)} \right]$$

---

## 3. Benchmark Comparison Protocol

To empirically validate the contribution of the Fuzzy Fusion layer, every experiment evaluates four distinct models on the exact same holdout test set:

| Model / Fusion Method | Description | Input to Decision |
|---|---|---|
| **Standalone ResNet-101** | Feature extraction focusing on macro-structural knee contours | Single model softmax probability |
| **Standalone DenseNet-201** | Feature extraction focusing on dense trabecular micro-texture | Single model softmax probability |
| **Simple Probability Average** | Classical baseline ensemble: $P_{\text{avg}} = 0.5 \cdot P_{\text{Res}} + 0.5 \cdot P_{\text{Dense}}$ | Linear arithmetic mean |
| **Weighted Average** | Optimal linear combination: $P_{\text{w}} = w_1 P_{\text{Res}} + (1-w_1) P_{\text{Dense}}$ | Tuned on validation ROC |
| **Fuzzy Logic Fusion (Proposed)** | 9-rule Mamdani FIS with Centroid defuzzification | Non-linear linguistic inference |

### Standard Reporting Table Template for Paper
```
========================================================================================================
Model                  Accuracy (95% CI)    Sensitivity (95% CI)   Specificity (95% CI)   ROC-AUC (95% CI)
--------------------------------------------------------------------------------------------------------
ResNet-101             --.-% [--.-, --.-]   --.-% [--.-, --.-]     --.-% [--.-, --.-]     0.--- [0.---, 0.---]
DenseNet-201           --.-% [--.-, --.-]   --.-% [--.-, --.-]     --.-% [--.-, --.-]     0.--- [0.---, 0.---]
Simple Average         --.-% [--.-, --.-]   --.-% [--.-, --.-]     --.-% [--.-, --.-]     0.--- [0.---, 0.---]
Fuzzy Fusion (Ours)    --.-% [--.-, --.-]   --.-% [--.-, --.-]     --.-% [--.-, --.-]     0.--- [0.---, 0.---]
========================================================================================================
```
