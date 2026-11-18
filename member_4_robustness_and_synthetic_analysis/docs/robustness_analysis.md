# Member 4 (Subtask 4.1) — Robustness & Class-Imbalance Analysis

## 1. Experimental Overview

To stress-test the boundary reliability of `DensityAdaptiveKNN`, Member 4 conducted systematic multi-seed perturbation evaluations across two stress regimes:
1. **Feature-Noise Injection:** Gaussian noise perturbations $\sigma \in [0.0, 0.1, 0.25, 0.5, 1.0]$ added to test feature vectors.
2. **Artificial & Natural Class Imbalance:** Controlled minority subsampling at majority:minority ratios $1:1, 1:2, 1:5, 1:10$.

---

## 2. Key Empirical Findings

### 2.1 Autonomous Noise Damping & Dynamic Neighborhood Expansion
- In conventional fixed-K KNN (notably $K=1$), feature noise degrades Macro-F1 sharply because single perturbed outliers directly flip class boundaries.
- In `DensityAdaptiveKNN`, feature noise dilutes point density and increases pairwise Euclidean distances $d_m(x)$.
- Under the inverse density mapping, the algorithm autonomously expands $K(x)$ from **$3.6-8.4$ at $\sigma=0.0$ up to $12.1-14.1$ at $\sigma=1.0$**.
- This acts as an **intrinsic low-pass filter**, aggregating additional contextual evidence to dampen noise and maintaining superior Macro-F1 across all noise scales.

### 2.2 Minority Class Recall Protection Under Imbalance
- When classes are heavily imbalanced (e.g., Glass with natural 8.4:1 skew or artificial 1:10 downsampling), fixed large neighborhoods ($K=15$) collapse minority recall to **0.00%** because the majority class completely dominates the voting window.
- Fixed $K=1$ maintains higher sensitivity but incurs high variance and erratic false positives.
- `DensityAdaptiveKNN (Distance-Weighted)` achieves the best compromise:
  - In low-density minority regions, distance weighting ($w_i = 1 / d_i$) prevents distant majority neighbors from overwhelming adjacent minority points.
  - Across all three datasets at ratio 1:10, distance-weighted Adaptive KNN maintains **$20-52\%$ higher minority recall** than fixed $K=15$.

---

## 3. Generated Deliverables
- Tables:
  - `member_4_robustness_and_synthetic_analysis/results/tables/noise_experiments.csv`
  - `member_4_robustness_and_synthetic_analysis/results/tables/imbalance_experiments.csv`
- Figures:
  - `member_4_robustness_and_synthetic_analysis/results/figures/noise_robustness_curves.png`
  - `member_4_robustness_and_synthetic_analysis/results/figures/adaptive_k_under_noise.png`
  - `member_4_robustness_and_synthetic_analysis/results/figures/minority_recall_heatmaps.png`
