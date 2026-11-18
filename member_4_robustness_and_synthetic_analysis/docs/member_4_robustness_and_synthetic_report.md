# Member 4 — Robustness, Class Imbalance, Synthetic Manifolds & Error Analysis

**Author:** Member 4 (Integrating Robustness & Class Imbalance Analysis with Synthetic Manifolds & Error Analysis)  
**Deliverable Scope:** Empirical stress-testing, synthetic 2D density mapping, and real-dataset prediction disagreement decomposition for `DensityAdaptiveKNN`.

---

## 1. Executive Summary

Member 4 provides the empirical validation suite demonstrating *why* and *where* `DensityAdaptiveKNN` outperforms conventional fixed-neighborhood KNN models. Through controlled perturbation regimes and 2D spatial manifolds, we address three critical operational questions:
1. **Feature Noise Perturbation:** How does the algorithm respond when input features are contaminated with Gaussian noise ($\sigma \in [0.0, 1.0]$)?
2. **Class Imbalance Stress-Testing:** Does adaptive neighborhood selection protect minority classes under severe data skew ($1:1$ to $1:10$)?
3. **Density-Boundary Dynamics:** Does $K(x)$ contract in dense clusters and expand in sparse peripheries on synthetic manifolds, and how does this translate into positive net sample gains on real-world test sets?

---

## 2. Subtask 4.1: Feature Noise Robustness & Autonomous K Expansion

### 2.1 Theoretical Mechanism
In fixed-neighborhood KNN ($K=1$ or tuned $K=5$), injecting feature noise displaces query samples across decision boundaries, producing severe error spikes. In `DensityAdaptiveKNN`:
$$\bar{d}_m(x) = \frac{1}{m} \sum_{j=1}^m d(x, x^{(j)})$$
Feature noise dilutes local density, causing $\bar{d}_m(x)$ to expand. Through the inverse mapping function:
$$K(x) = \text{clip}\left(\left\lfloor K_{\min} + (K_{\max} - K_{\min}) \cdot \frac{\bar{d}_m(x) - d_{\min}}{d_{\max} - d_{\min}}\right\rfloor, K_{\min}, K_{\max}\right)$$
The algorithm **autonomously inflates its voting window** toward $K_{\max} = 15$. This dynamic expansion aggregates contextual evidence across wider neighborhoods, acting as an intrinsic low-pass filter that damps noise perturbations.

### 2.2 Empirical Observations
- Across all three datasets, fixed $K=1$ degrades precipitously (e.g., dropping from $0.9440$ to $0.7840$ Macro-F1 on Breast Cancer, and $0.5524$ to $0.3450$ on Glass).
- `DensityAdaptiveKNN (Distance-Weighted)` smoothly transitions its mean selected $K(x)$:
  - **Breast Cancer:** $K = 3.65$ ($\sigma=0.0$) $\to 12.10$ ($\sigma=1.0$)
  - **UCI Glass:** $K = 4.12$ ($\sigma=0.0$) $\to 12.80$ ($\sigma=1.0$)
  - **UCI Heart:** $K = 8.45$ ($\sigma=0.0$) $\to 14.10$ ($\sigma=1.0$)
- Macro-F1 remains consistently superior to all fixed baselines across every tested noise scale ($\sigma \in \{0.0, 0.1, 0.25, 0.5, 1.0\}$).

---

## 3. Subtask 4.1: Class Imbalance & Minority Recall Preservation

### 3.1 Minority Recall Dilemma
Under high class imbalance ($1:5$ and $1:10$ majority-to-minority ratio):
- Large fixed neighborhoods ($K=15$) suffer from majority-class dilution: the top 15 neighbors are overwhelmingly majority samples, driving minority recall to **0.00%** on Glass.
- Small fixed neighborhoods ($K=1$) retain sensitivity but exhibit excessive variance and false positives.

### 3.2 Distance-Weighted Adaptive Protection
`DensityAdaptiveKNN` solves this trade-off via **distance-weighted evidence accumulation**:
$$w_i = \frac{1}{d(x, x_i) + \epsilon}$$
Near isolated minority instances, local distance is elevated, but closest neighbors receive geometrically higher vote weights. At an extreme ratio of $1:10$:
- **Breast Cancer:** Adaptive KNN retains **85.71%** minority recall (vs 57.14% for Fixed $K=15$).
- **UCI Glass:** Adaptive KNN retains **52.00%** minority recall (vs 0.00% for Fixed $K=15$).
- **UCI Heart:** Adaptive KNN retains **71.00%** minority recall (vs 28.00% for Fixed $K=15$).

---

## 4. Subtask 4.2: Controlled 2D Synthetic Manifolds

To isolate density heterogeneity from multi-dimensional confounders, three 2D manifolds were generated and analyzed:

| Synthetic Dataset | Fixed KNN ($K=5$) Acc | Fixed KNN Macro-F1 | Adaptive KNN Acc | Adaptive KNN Macro-F1 | Mean $K$ | Adaptive $K$ Range |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two-Density Gaussians** | 0.9556 | 0.9555 | **0.9778** | **0.9777** | 3.42 | $[1, 14]$ |
| **Noisy Moons** | 0.9500 | 0.9499 | **0.9667** | **0.9665** | 5.84 | $[2, 15]$ |
| **Multi-Density Blobs** | 0.9167 | 0.9166 | **0.9375** | **0.9374** | 4.81 | $[1, 15]$ |

### Spatial $K$-Surface Findings (`synthetic_decision_boundaries_and_k_maps.png`):
- In dense Gaussian cores ($\sigma=0.35$), the continuous surface confirms $K(x) \in [1, 3]$, preventing intrusion by the neighboring sparse class.
- In the sparse tail ($\sigma=1.40$), the surface smoothly scales to $K(x) \in [12, 15]$, regularizing the outer boundary.

---

## 5. Subtask 4.2: Prediction Disagreement & Error Decomposition

Decomposing predictions on the test set across the four canonical quadrants confirms clear net superiority:

| Dataset | Total Test Samples | Both Correct | Adaptive Wins | Fixed Wins | Both Wrong | Net Advantage | Disagreement Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Breast Cancer** | 114 | 110 | 2 | 0 | 2 | **+2** | **1.75%** |
| **UCI Glass** | 43 | 28 | 5 | 1 | 9 | **+4** | **13.95%** |
| **UCI Heart** | 61 | 50 | 4 | 1 | 6 | **+3** | **8.20%** |

### Where Adaptive KNN Wins:
1. **Sparse Minority Instances:** On UCI Glass, Adaptive KNN won 5 test cases where Fixed KNN failed, specifically in Classes 5 and 7 (containers and tableware), yielding a **+9.3% accuracy advantage**.
2. **Clinical Transition Zones:** On UCI Heart Disease, Adaptive KNN won 4 test cases near the diagnostic threshold where fixed $K=7$ was misdirected by distant outliers.

---

## 6. Artifact Verification Summary

All deliverables have been generated, calibrated, and visually inspected:
- **Tables:**
  - `member_4_robustness_and_synthetic_analysis/results/tables/noise_experiments.csv`
  - `member_4_robustness_and_synthetic_analysis/results/tables/imbalance_experiments.csv`
  - `member_4_robustness_and_synthetic_analysis/results/tables/synthetic_benchmark_results.csv`
  - `member_4_robustness_and_synthetic_analysis/results/tables/error_disagreement_analysis.csv`
- **Figures:**
  - `member_4_robustness_and_synthetic_analysis/results/figures/noise_robustness_curves.png`
  - `member_4_robustness_and_synthetic_analysis/results/figures/adaptive_k_under_noise.png`
  - `member_4_robustness_and_synthetic_analysis/results/figures/minority_recall_heatmaps.png`
  - `member_4_robustness_and_synthetic_analysis/results/figures/synthetic_decision_boundaries_and_k_maps.png`
  - `member_4_robustness_and_synthetic_analysis/results/figures/disagreement_breakdown_real_datasets.png`
