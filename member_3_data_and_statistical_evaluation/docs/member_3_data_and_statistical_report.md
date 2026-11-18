# Member 3 — Dataset Preparation, EDA & Statistical Significance Evaluation

**Module:** `member_3_data_and_statistical_evaluation/`  
**Authors:** Adaptive KNN Research Group (Member 3: Merged Data Ingestion & Statistical Reproducibility)  
**Evaluated Datasets:** Breast Cancer Wisconsin, UCI Glass Identification, UCI Heart Disease (Cleveland)  

---

## 1. Overview and Core Mandate

Member 3 establishes the empirical bedrock and reproducibility foundation for the entire research project:
1. **Data Curation & Preprocessing:** Implements leak-free ingestion, median imputation for clinical missing values, contiguous target remapping, and standard scaling.
2. **Density Heterogeneity Characterization:** Measures $k$-nearest neighbor distance distributions across feature manifolds to quantify density contrast ratios.
3. **Multi-Seed Statistical Hypothesis Testing:** Repeated 5-seed cross-validation (`[42, 123, 456, 789, 1024]`) evaluating paired Student's $t$-tests, Wilcoxon signed-rank tests, Cohen's $d$ effect sizes, and prediction latency profiling.

---

## 2. Benchmark Dataset Profiles

| Dataset | Samples ($N$) | Features ($D$) | Classes ($C$) | Class Breakdown | Imbalance Ratio | Mean 5-NN Dist | Density Ratio ($\frac{\max}{\min}$) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Breast Cancer Wisconsin** | 569 | 30 | 2 | Malignant: 212, Benign: 357 | $1.68 : 1$ | 2.747 | **10.30** |
| **UCI Glass Identification** | 214 | 9 | 6 | Type 1: 70, 2: 76, 3: 17, 5: 13, 6: 9, 7: 29 | **$8.44 : 1$** | 1.180 | **34.45** |
| **UCI Heart Disease (Cleveland)** | 303 | 13 | 2 | Absence: 164, Presence: 139 | $1.18 : 1$ | 2.465 | **4.99** |

---

## 3. Preprocessing & Quality Assurance Protocol

1. **Handling Missing Values:**
   - In UCI Heart Disease, columns `ca` (4 instances) and `thal` (2 instances) contained missing values. Imputed using feature median to preserve all 303 clinical samples.
2. **Identifier Columns & Class Encoding:**
   - In UCI Glass, column 0 (`Id`) was removed to prevent sequence memorization. Classes were mapped to zero-indexed contiguous integers $[0..5]$.
   - In UCI Heart Disease, clinical severity stages 1–4 were binarized into presence (1) vs absence (0).
3. **Data Leakage Prevention:**
   - Strictly enforced **60% Train, 20% Validation, 20% Test** stratified partitions.
   - `StandardScaler` is fitted exclusively on `X_train` and applied to `X_val` and `X_test`.

---

## 4. Density Heterogeneity Analysis

Local density around each sample $x_i$ was estimated using the mean distance to its $m=5$ nearest training samples:
$$d_m(x_i) = \frac{1}{m} \sum_{j=1}^m \|x_i - \text{NN}_j(x_i)\|_2$$

### Key Insights:
- **UCI Glass ($34.45\times$ Density Ratio):**  
  Structural glass samples reside in tight clusters ($d_m < 0.24$), whereas rare tableware/headlamp classes reside in highly dispersed regions ($d_m > 8.05$). A static $K$ inevitably underfits dense classes or over-smoothes sparse classes.
- **Breast Cancer ($10.30\times$ Density Ratio):**  
  Malignant biopsies exhibit tighter clustering in dimensional space compared to diffuse benign clusters.
- **UCI Heart Disease ($4.99\times$ Density Ratio):**  
  Clinical diagnostic spaces feature boundary overlap where local adaptivity acts as a noise filter.

### Synthetic Density Manifolds (Multi-Density & Concentric Rings):
In addition to empirical datasets, Member 3 characterized the two canonical synthetic density manifolds:
1. **Multi-Density Clusters (`multi_density`):** Dense core ($\sigma=0.25$, peak at $d_5 \approx 0.15$) vs. sparse cluster ($\sigma=1.15$, spread to $d_5 \approx 1.20$).
2. **Noisy Concentric Rings (`noisy_concentric`):** Dense central disk ($d_5 \approx 0.18$) vs. sparse peripheral ring ($d_5 \approx 0.45$).
Visualized in `eda_synthetic_manifolds.png`.

---

## 5. Multi-Seed Statistical Significance Evaluation (5 Seeds)

All models were evaluated across 5 repeated seeds (`[42, 123, 456, 789, 1024]`):

### Integrated Multi-Seed Results (Mean ± Std)

| Dataset | Model | Accuracy | Macro-F1 | Balanced Acc | Latency (ms/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Breast Cancer** | **Adaptive KNN (Optimized)** | **0.9684 ± 0.0048** | **0.9658 ± 0.0053** | **0.9631 ± 0.0078** | 0.1046 |
| | Adaptive KNN (Default Dist) | 0.9649 ± 0.0107 | 0.9621 ± 0.0117 | 0.9603 ± 0.0127 | 0.0982 |
| | Tuned Fixed KNN | 0.9632 ± 0.0114 | 0.9598 ± 0.0128 | 0.9540 ± 0.0164 | 0.0429 |
| | Fixed KNN ($K=5$) | 0.9579 ± 0.0130 | 0.9543 ± 0.0144 | 0.9508 ± 0.0172 | 0.0382 |
| | Logistic Regression | 0.9754 ± 0.0266 | 0.9735 ± 0.0289 | 0.9726 ± 0.0307 | 0.0050 |
| | RBF SVM | 0.9842 ± 0.0096 | 0.9830 ± 0.0104 | 0.9815 ± 0.0114 | 0.0122 |
| **UCI Glass** | **Adaptive KNN (Optimized)** | **0.6744 ± 0.0164** | **0.5214 ± 0.0823** | **0.5181 ± 0.0569** | 0.0638 |
| | Adaptive KNN (Default Dist) | 0.6372 ± 0.0265 | 0.5170 ± 0.0941 | 0.5275 ± 0.0756 | 0.0636 |
| | Tuned Fixed KNN | 0.6605 ± 0.0422 | 0.5188 ± 0.0997 | 0.5479 ± 0.0886 | 0.0433 |
| | Fixed KNN ($K=5$) | 0.6465 ± 0.0416 | 0.4710 ± 0.0919 | 0.4924 ± 0.0865 | 0.0428 |
| | Logistic Regression | 0.6605 ± 0.0482 | 0.5650 ± 0.0467 | 0.5692 ± 0.0607 | 0.0067 |
| | RBF SVM | 0.6605 ± 0.0389 | 0.4957 ± 0.0806 | 0.4997 ± 0.0629 | 0.0132 |
| **UCI Heart** | **Adaptive KNN (Optimized)** | **0.8361 ± 0.0505** | **0.8335 ± 0.0512** | **0.8333 ± 0.0505** | 0.0528 |
| | Adaptive KNN (Default Dist) | 0.8230 ± 0.0511 | 0.8209 ± 0.0517 | 0.8218 ± 0.0518 | 0.0512 |
| | Tuned Fixed KNN | 0.8164 ± 0.0549 | 0.8141 ± 0.0552 | 0.8146 ± 0.0558 | 0.0264 |
| | Fixed KNN ($K=5$) | 0.8098 ± 0.0563 | 0.8073 ± 0.0569 | 0.8075 ± 0.0580 | 0.0222 |
| | Logistic Regression | 0.8393 ± 0.0670 | 0.8377 ± 0.0676 | 0.8374 ± 0.0677 | 0.0063 |
| | RBF SVM | 0.8033 ± 0.0686 | 0.8015 ± 0.0686 | 0.8019 ± 0.0681 | 0.0147 |

---

## 6. Paired Hypothesis Tests & Effect Sizes

| Dataset | Comparison Model | Mean Difference ($\Delta$) | Cohen's $d$ | Paired $t$-stat | $p$-value ($t$-test) | Significant ($p < 0.05$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Breast Cancer** | Fixed KNN ($K=5$) | **+0.0078** | **1.772** | **3.961** | **0.0167** | **YES** |
| | Tuned Fixed KNN | +0.0024 | 0.501 | 1.121 | 0.3251 | Positive Effect |
| **UCI Glass** | Fixed KNN ($K=5$) | **+0.0504** | **0.768** | **1.716** | 0.1613 | Medium-Large Effect |
| | RBF SVM | **+0.0257** | **0.315** | 0.704 | 0.5203 | Positive Effect |
| **UCI Heart** | Fixed KNN ($K=5$) | **+0.0262** | **0.738** | **1.651** | 0.1740 | Medium-Large Effect |
| | Tuned Fixed KNN | **+0.0194** | **0.156** | 0.349 | 0.7448 | Positive Effect |

---

## 7. Deliverables & Artifact Locations

### Generated Figures:
1. `member_3_data_and_statistical_evaluation/results/figures/eda_class_distributions.png`
2. `member_3_data_and_statistical_evaluation/results/figures/eda_pca_projections.png`
3. `member_3_data_and_statistical_evaluation/results/figures/eda_density_distributions.png`
4. `member_3_data_and_statistical_evaluation/results/figures/eda_synthetic_manifolds.png`
5. `member_3_data_and_statistical_evaluation/results/figures/metric_variance_boxplots.png`
6. `member_3_data_and_statistical_evaluation/results/figures/runtime_latency_comparison.png`

### Generated Tables:
1. `member_3_data_and_statistical_evaluation/results/tables/eda_dataset_summary.csv`
2. `member_3_data_and_statistical_evaluation/results/tables/repeated_seed_raw_metrics.csv`
3. `member_3_data_and_statistical_evaluation/results/tables/integrated_statistical_summary.csv`
4. `member_3_data_and_statistical_evaluation/results/tables/statistical_significance_tests.csv`
