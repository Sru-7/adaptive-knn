# Member 4 — Robustness & Class-Imbalance Analysis

## 1. Experimental Overview

To thoroughly test the reliability and boundary dynamics of `DensityAdaptiveKNN`, Member 4 performed two systematic perturbation evaluations repeated over 5 random seeds:
1. **Feature-Noise Injection:** $\sigma \in [0.0, 0.1, 0.25, 0.5, 1.0]$ added to test features.
2. **Artificial & Natural Class Imbalance:** Tested at target majority-to-minority ratios $1:1, 1:2, 1:5, 1:10$.

---

## 2. Key Empirical Findings

### 2.1 Noise Resilience & Adaptive Expansion
- In conventional fixed-K KNN (especially $K=1$), noise degrades performance sharply because single perturbed outliers directly flip labels.
- In `DensityAdaptiveKNN`, feature noise increases the local Euclidean distances $d_m(x)$ to nearest neighbors. Under the inverse density mapping, this automatically inflates $K(x)$ toward $K_{\max} = 15$!
- **Automatic Noise Damping:** The algorithm autonomously responds to noise by enlarging its voting neighborhood, suppressing random fluctuations.

### 2.2 Minority Class Recall Under Imbalance
- When classes are heavily skewed (e.g. Glass dataset with natural 8.44:1 ratio, or artificial 1:10 ratio), large fixed $K$ (e.g. $K=15$) causes the majority class to overwhelm minority samples.
- Small fixed $K$ (e.g. $K=1$) preserves minority recall but suffers high variance.
- `DensityAdaptiveKNN (Distance-Weighted)` achieves the best compromise: close neighbors are weighted proportionally ($w_i = 1 / d_i$), preventing majority dilution while retaining statistical stability.

---

## 3. Deliverables & Results
- Metric Tables:
  - `member_4_robustness_and_synthetic_analysis/results/tables/noise_experiments.csv`
  - `member_4_robustness_and_synthetic_analysis/results/tables/imbalance_experiments.csv`
- Figures:
  - `member_4_robustness_and_synthetic_analysis/results/figures/noise_robustness_curves.png`
  - `member_4_robustness_and_synthetic_analysis/results/figures/adaptive_k_under_noise.png`
  - `member_4_robustness_and_synthetic_analysis/results/figures/minority_recall_heatmaps.png`
