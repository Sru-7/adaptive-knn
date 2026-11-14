# Member 7 — Adaptive-K Mapping Strategies (New Algorithmic Contribution)

## 1. Algorithmic Formulations

We developed and systematically investigated four distinct transformation mechanisms mapping local distance $d_m(x)$ to neighborhood size $K(x) \in [K_{\min}, K_{\max}]$:

1. **Linear Mapping:**
   $$K_{\text{lin}}(d) = \text{round}\left(K_{\min} + \frac{d - d_{\min}}{d_{\max} - d_{\min}} (K_{\max} - K_{\min})\right)$$
   *Pros:* Computationally instantaneous.
   *Cons:* Highly sensitive to extreme single-sample outlier distances that compress normal points into $K_{\min}$.

2. **Quantile / Rank-Based Mapping:**
   $$K_{\text{quant}}(d) = \text{round}\left(K_{\min} + \hat{F}(d) \cdot (K_{\max} - K_{\min})\right)$$
   where $\hat{F}(d) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(d_m(x_i) \le d)$ is the empirical CDF of training set local distances.
   *Pros:* Completely invariant to monotonic distance stretching; uniformly distributes $K$ utilization across $[K_{\min}, K_{\max}]$.

3. **Sigmoidal Logistic Mapping:**
   $$K_{\text{sig}}(d) = \text{round}\left(K_{\min} + \frac{K_{\max} - K_{\min}}{1 + e^{-2 (d - d_{\text{median}}) / \text{IQR}}}\right)$$
   *Pros:* Focuses dynamic adaptation around the transition boundary while saturating cleanly in asymptotic regimes.

4. **Step Threshold Mapping:**
   Binary split at the median distance: $K_{\min}$ if $d \le d_{\text{median}}$, else $K_{\max}$.

---

## 2. Comparative Performance Across Datasets

| Dataset | Best Mapping Strategy | Best Macro-F1 | Notes |
| :--- | :---: | :---: | :--- |
| **Breast Cancer Wisconsin** | Quantile / Sigmoid | **0.970** | Smooth decision margins preserve high sensitivity |
| **UCI Glass Identification** | Quantile | **0.589** | Robustness to outlier distances in rare classes |
| **UCI Heart Disease** | Quantile / Linear | **0.868** | Stable boundary separation in overlapping clinical space |

**Generalization Champion:** The **Quantile / Rank-Based mapping** generalizes best across all 3 datasets because it eliminates scale dependence and avoids outlier compression.

---

## 3. Sensitivity Analysis ($m, K_{\min}, K_{\max}$)

- **Reference Neighborhood $m$:** Performance is stable across $m \in [3, 7]$. Setting $m > 10$ slightly degrades local resolution on smaller datasets (e.g. Glass). Optimal default: **$m=5$**.
- **Lower Bound $K_{\min}$:** Setting $K_{\min}=1$ allows dense clusters to preserve tight separation without cross-cluster contamination.
- **Upper Bound $K_{\max}$:** Setting $K_{\max}=15$ or $21$ provides ample variance reduction in sparse regions without over-diluting small classes.

---

## 4. Generated Deliverables
- Tables:
  - `member_7_mapping_strategies/results/tables/mapping_comparison.csv`
  - `member_7_mapping_strategies/results/tables/sensitivity_grid_results.csv`
- Figures:
  - `member_7_mapping_strategies/results/figures/mapping_curves.png`
  - `member_7_mapping_strategies/results/figures/mapping_comparison_bars.png`
  - `member_7_mapping_strategies/results/figures/sensitivity_heatmaps.png`
