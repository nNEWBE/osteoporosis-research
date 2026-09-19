# 03. Limitations, Clinical Caveats, and Ethical Considerations

## 1. Quantitative Dataset Audit Findings

Prior to modeling, a cryptographic hash audit (MD5/SHA256) was conducted across all 1,573 images in the dataset. The analysis uncovered substantial structural data anomalies that must be accounted for in any scientific publication:

### Summary of Audit Discoveries

| Metric | Normal Folder | Osteoporosis Folder | Combined / Cross-Class | Total / Impact |
|---|---|---|---|---|
| **Raw Image Files** | 780 | 793 | — | 1,573 files |
| **Unique Image Hashes** | 384 | 395 | — | 779 unique |
| **Intra-class Duplicates** | 396 (50.8%) | 398 (50.2%) | — | 794 redundant files |
| **Cross-Class Contradictions** | — | — | **24 hashes** (48+ files) | Labeled as BOTH Normal & Osteo |
| **Clean Pristine Dataset** | **360** | **371** | — | **731 clean unique images** |

### Why This Matters for Scientific Integrity
1. **Catastrophic Train-Test Leakage**: If the raw 1,573 files are split randomly (70/15/15) without cryptographic deduplication, ~50% of the test set images are identical bitwise clones of images in the training set. This artificially inflates accuracy to near 100%, creating an illusion of high performance that completely collapses on external hospital data.
2. **Contradictory Ground Truth Noise**: The 24 cross-class duplicate images present identical radiographic patterns with opposite labels. Training on conflicting labels forces the neural networks to memorize file names rather than true anatomical biomarkers.
3. **Research Transparency**: By explicitly reporting this audit and performing experiments on deduplicated, sanitized partitions, your paper demonstrates scientific rigor that peer reviewers value.

---

## 2. Clinical and Methodological Limitations

### 2.1 The Clinical Spectrum: Exclusion of Osteopenia
- **WHO Diagnostic Criteria**:
  - **Normal**: T-score $\ge -1.0$
  - **Osteopenia (Low Bone Mass)**: $-2.5 < \text{T-score} < -1.0$
  - **Osteoporosis**: $\text{T-score} \le -2.5$
  - **Severe Osteoporosis**: $\text{T-score} \le -2.5$ with fragility fracture.
- **Limitation**: The current dataset excludes the 374 Osteopenia cases. Binary classification (Normal vs Osteoporosis) artificially simplifies the problem by testing only the extremes of the spectrum.
- **Clinical Implication**: In real clinical practice, the transition between mild osteopenia and early osteoporosis is subtle and continuous. The current system should be framed as a **high-confidence extreme-state detector**, not a comprehensive bone mineral density staging system.

### 2.2 2D Radiography vs. Gold Standard DXA
- **DXA (Dual-Energy X-ray Absorptiometry)**: The clinical gold standard measures areal Bone Mineral Density (aBMD in $\text{g/cm}^2$) primarily at the lumbar spine ($L_1\text{–}L_4$) and proximal femur (femoral neck, total hip).
- **Plain Knee Radiographs**: Knee X-rays capture projected 2D attenuation of the distal femur and proximal tibia. While trabecular thinning and cortical reduction in the distal femur correlate with systemic osteoporosis, 2D plain films are prone to projection geometry artifacts, soft-tissue magnification, patient rotation, and knee flexion angles.
- **Osteoarthritis Confounder**: Knee radiographs in elderly populations frequently exhibit concurrent knee osteoarthritis (subchondral sclerosis, osteophyte formation, joint space narrowing). Subchondral sclerosis artificially increases local radio-opacity, which can mask underlying osteoporotic trabecular demineralization.

### 2.3 Absence of Patient-Level Metadata
- The dataset provides no patient identification numbers (Patient IDs), age, sex, BMI, menopausal status, prior fragility fractures, or medication records (e.g. glucocorticoid therapy, bisphosphonates).
- **Risk of Patient-Level Leakage**: If a single patient underwent bilateral knee imaging (both left and right knees) or serial examinations over time, these distinct images may have identical or near-identical features. Without Patient IDs, patient-level stratification cannot be strictly guaranteed beyond bitwise deduplication.

### 2.4 Single-Source Distribution Shift
- The data lacks multi-scanner provenance (GE Healthcare, Siemens Healthineers, Philips Healthcare, Shimadzu).
- Exposure parameters ($kVp$, $mAs$, source-to-image distance) and post-processing filters (edge enhancement, window/level presets) vary across medical institutions.
- Performance on external cohorts cannot be assumed without explicit out-of-distribution (OOD) validation on multi-institutional datasets.

---

## 3. Recommended Phrasing for Research Papers & Theses

When writing your methodology and discussion sections, use the following phrasing to turn these limitations into academic strengths:

> *"To ensure rigorous internal validity and eliminate optimistic reporting bias, our study first conducted an automated cryptographic hash verification across the cohort. This uncovered that 50.4% of the raw repository comprised redundant duplicate instances and 24 conflicting cross-class label pairs. By sanitizing the repository into 731 pristine, unique knee radiographs and implementing a strictly disjoint 70/15/15 stratified partition, our experimental results reflect genuine generalizability rather than memorized identical twins."*

> *"While Dual-Energy X-ray Absorptiometry (DXA) remains the definitive clinical standard for quantitative BMD measurement, our hybrid ResNet101-DenseNet201 fuzzy framework is designed for opportunistic screening: identifying high-risk individuals from routine musculoskeletal knee radiographs who should be prioritized for confirmatory DXA evaluation."*
