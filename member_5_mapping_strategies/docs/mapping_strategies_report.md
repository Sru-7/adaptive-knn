# Member 5 — Adaptive-K Mapping Strategies & Algorithmic Sensitivity

**Author:** Member 5 (Algorithmic Mapping Strategies & Hyperparameter Sensitivity Analysis)  
**Deliverable Scope:** Mathematical transformations from local distance $d_m(x)$ to neighborhood size $K(x) \in [K_{\min}, K_{\max}]$ and multi-dimensional grid search.

---

## 1. Algorithmic Formulations

Member 5 formulated and systematically benchmarked four distinct transformation mechanisms mapping local distance $d_m(x)$ to neighborhood size $K(x)$:

1. **Linear Inverse Density Mapping:**
   $$K_{\text{lin}}(d) = \text{round}\left(K_{\min} + \frac{d - d_{\min}}{d_{\max} - d_{\min}} (K_{\max} - K_{\min})\right)$$
   - *Characteristics:* Computationally instantaneous; simple intuitive scaling.
   - *Limitation:* Sensitive to extreme outlier distances that compress normal points into $K_{\min}$.

2. **Quantile / Rank-Based Mapping (Generalization Champion):**
   $$K_{\text{quant}}(d) = \text{round}\left(K_{\min} + \hat{F}(d) \cdot (K_{\max} - K_{\min})\right)$$
   where $\hat{F}(d) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(d_m(x_i) \le d)$ is the empirical CDF of training set local distances.
   - *Characteristics:* Completely invariant to monotonic distance stretching; uniform utilization across $[K_{\min}, K_{\max}]$.
   - *Advantage:* Delivers the highest Macro-F1 on heterogeneous datasets (UCI Glass: **0.7120**, UCI Heart: **0.9015**).

3. **Sigmoidal Logistic Mapping:**
   $$K_{\text{sig}}(d) = \text{round}\left(K_{\min} + \frac{K_{\max} - K_{\min}}{1 + \exp\left(-2 \frac{d - d_{\text{median}}}{\text{IQR}}\right)}\right)$$
   - *Characteristics:* Centers maximum adaptation sensitivity around the median distance while saturating smoothly at outer density tails.

4. **Step Threshold Mapping:**
   $$K_{\text{thresh}}(d) = \begin{cases} K_{\min}, & \text{if } d \le d_{\text{median}} \\ K_{\max}, & \text{if } d > d_{\text{median}} \end{cases}$$
   - *Characteristics:* Binary piece-wise baseline serving as a control condition.

---

## 2. Comparative Performance Across Datasets

| Dataset | Linear Macro-F1 | Quantile Macro-F1 | Sigmoidal Macro-F1 | Threshold Macro-F1 | Best Mapping Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Breast Cancer Wisconsin** | **0.9852** | 0.9809 | 0.9825 | 0.9650 | **Linear** |
| **UCI Glass Identification** | 0.6858 | **0.7120** | 0.7015 | 0.6540 | **Quantile (Rank-Based)** |
| **UCI Heart Disease** | 0.8923 | **0.9015** | 0.8950 | 0.8688 | **Quantile (Rank-Based)** |

### Analytical Insights:
- On regular high-dimensional manifolds (**Breast Cancer**), **Linear mapping** achieves the highest Macro-F1 (**0.9852**) because feature density varies continuously.
- On highly skewed manifolds (**UCI Glass**, **UCI Heart**), **Quantile mapping** outperforms Linear by **+2.6%** on Glass and **+0.9%** on Heart, because empirical CDF normalization protects against outlier distance compression.

---

## 3. Multi-Dimensional Sensitivity Analysis ($m, K_{\min}, K_{\max}$)

We evaluated grid combinations across $m \in \{3, 5, 7, 10, 15\}$, $K_{\min} \in \{1, 3, 5\}$, and $K_{\max} \in \{9, 15, 21\}$:

1. **Density Reference Window ($m$):**
   - Performance peaks at **$m = 5$** across all three datasets.
   - Setting $m < 3$ incurs local noise in density estimation. Setting $m > 10$ over-smoothes local structure in smaller datasets.
2. **Lower Neighborhood Bound ($K_{\min}$):**
   - Setting **$K_{\min} = 1$** consistently yields the highest validation Macro-F1, enabling dense clusters to retain sharp class separation.
3. **Upper Neighborhood Bound ($K_{\max}$):**
   - Setting **$K_{\max} = 15$** provides the optimal variance reduction without cross-class dilution.

---

## 4. Generated Deliverables
- **Tables:**
  - `member_5_mapping_strategies/results/tables/mapping_comparison.csv`
  - `member_5_mapping_strategies/results/tables/sensitivity_grid_results.csv`
- **Figures:**
  - `member_5_mapping_strategies/results/figures/mapping_curves.png`
  - `member_5_mapping_strategies/results/figures/mapping_comparison_bars.png`
  - `member_5_mapping_strategies/results/figures/sensitivity_heatmaps.png`
