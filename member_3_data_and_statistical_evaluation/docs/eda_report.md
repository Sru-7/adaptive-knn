# Member 3 — Dataset Preparation & Exploratory Data Analysis

## 1. Selected Datasets Overview

We selected three distinct classification benchmarks from the UCI Machine Learning Repository and Scikit-Learn representing diverse sample sizes, dimensionalities, class counts, and geometric characteristics:

| Dataset | Samples (N) | Features (D) | Classes (C) | Majority : Minority Ratio | Mean 5-NN Dist | Density Ratio (Max/Min) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Breast Cancer Wisconsin** | 569 | 30 | 2 | 1.68 : 1 | 2.747 | 10.3 |
| **UCI Glass Identification** | 214 | 9 | 6 | 8.44 : 1 | 1.18 | 34.45 |
| **UCI Heart Disease (Cleveland)**| 303 | 13 | 2 | 1.18 : 1 | 2.465 | 4.99 |

---

## 2. Preprocessing & Quality Assurance Protocol

1. **Handling Missing Values:**
   - In UCI Heart Disease, columns `ca` (4 missing) and `thal` (2 missing) contained `?` markers. Missing entries were imputed using the feature median, preserving all 303 clinical samples.
2. **Identifier Columns & Class Encoding:**
   - In UCI Glass, column 0 (`Id`) was removed to prevent artificial sequence memorization. Classes [1, 2, 3, 5, 6, 7] were mapped to consecutive integer indices [0..5].
   - In UCI Heart Disease, the original target (0 = no disease, 1-4 = stage severity) was binarized (0 vs 1) following standard diagnostic benchmarks.
3. **Data Leakage Prevention:**
   - Datasets are partitioned into stratified Train (60%), Validation (20%), and Test (20%) sets.
   - `StandardScaler` is fitted strictly on the training partition and applied to validation and test partitions.

---

## 3. Density Heterogeneity Insights

As shown in `eda_density_distributions.png`:
- In **Glass**, the density ratio exceeds **34.45x**, indicating vast disparities between dense structural glass types and sparse container/headlamp classes. A static $K$ inevitably underfits dense classes or introduces cross-class contamination into sparse classes.
- In **Breast Cancer**, malignant tumors often appear in denser feature manifolds while benign samples show wider dispersion.
- In **Heart Disease**, overlapping clusters in clinical feature space benefit from adaptive smoothing in sparse regions.

---

## 4. Synthetic Density Manifolds (Multi-Density & Concentric Rings)

As illustrated in `eda_synthetic_manifolds.png`:
1. **Multi-Density Clusters (Dense Core vs Sparse Cloud):**
   - Class 0 is tightly concentrated ($\sigma = 0.25$), resulting in an acute 5-NN distance peak at $d_5 \approx 0.15$.
   - Class 1 is diffuse ($\sigma = 1.15$), yielding a broad distribution extending up to $d_5 \approx 1.20$.
   - Fixed-K models fail because any global $K$ that suppresses noise in Class 1 cross-contaminates Class 0.
2. **Noisy Concentric Rings (Dense Center vs Sparse Periphery):**
   - Class 0 occupies the high-density interior disk ($d_5 pprox 0.18$).
   - Class 1 occupies the outer diffuse ring ($d_5 pprox 0.45$).
   - The density disparity proves the necessity of dynamic $K(x)$ expansion from inner core to outer periphery.

---

## 5. Generated Deliverables
- Tables:
  - `member_3_data_and_statistical_evaluation/results/tables/eda_dataset_summary.csv`
- Figures:
  - `member_3_data_and_statistical_evaluation/results/figures/eda_class_distributions.png`
  - `member_3_data_and_statistical_evaluation/results/figures/eda_pca_projections.png`
  - `member_3_data_and_statistical_evaluation/results/figures/eda_density_distributions.png`
  - `member_3_data_and_statistical_evaluation/results/figures/eda_synthetic_manifolds.png`
