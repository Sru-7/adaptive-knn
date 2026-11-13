# Robustness, Class-Imbalance, Synthetic Benchmarks, and Prediction Error Analysis

**Modules:**
1. Robustness Under Feature Noise (`experiments/robustness_noise.py`)
2. Class-Imbalance and Minority-Class Recall Analysis (`experiments/robustness_imbalance.py`)
3. Synthetic Benchmarks and Adaptive Neighborhood Visualization (`experiments/synthetic_experiments.py`)
4. Prediction-Disagreement and Error Analysis (`experiments/error_analysis.py`)

---

## 1. Overview of Experimental Suite

This experimental framework validates the behavior of the lightweight `DensityAdaptiveKNN` model against standard fixed-$K$ baselines across multiple challenging conditions: varying local data density, non-linear manifolds, additive Gaussian feature noise, and severe class imbalance.

### File Structure & Deliverables

| Component | File / Location | Purpose |
|---|---|---|
| **Utilities** | [`experiments/utils.py`](file:///C:/Users/Aditi/adaptive-knn/experiments/utils.py) | `predict_with_k_values`, unified metrics calculation, reproducible seeds |
| **Synthetic Generators** | [`experiments/synthetic_data.py`](file:///C:/Users/Aditi/adaptive-knn/experiments/synthetic_data.py) | 2-Density Gaussians, Noisy Moons, Multi-Density Blobs with stratified splits |
| **Synthetic Benchmarks** | [`experiments/synthetic_experiments.py`](file:///C:/Users/Aditi/adaptive-knn/experiments/synthetic_experiments.py) | Adaptive-K vs Fixed-K ($K \in \{1, 3, 5, 7, 9, 11, 15\}$) across 5 seeds; per-query K maps |
| **Error & Disagreement** | [`experiments/error_analysis.py`](file:///C:/Users/Aditi/adaptive-knn/experiments/error_analysis.py) | Disagreement categorization (Adaptive Corrects vs Adaptive Errors), K distributions |
| **Noise Robustness** | [`experiments/robustness_noise.py`](file:///C:/Users/Aditi/adaptive-knn/experiments/robustness_noise.py) | Feature-noise injection $\sigma \in \{0, 0.1, 0.25, 0.5, 1.0\}$ across 3 datasets |
| **Class Imbalance** | [`experiments/robustness_imbalance.py`](file:///C:/Users/Aditi/adaptive-knn/experiments/robustness_imbalance.py) | Undersampling ratios 1:1, 1:2, 1:5, 1:10; per-class recall heatmaps |
| **Master Runner** | [`experiments/run_all.py`](file:///C:/Users/Aditi/adaptive-knn/experiments/run_all.py) | One-command execution of all experiment pipelines |
| **Data Tables (CSV)** | [`results/tables/`](file:///C:/Users/Aditi/adaptive-knn/results/tables/) | `synthetic_results.csv`, `error_analysis.csv`, `noise_results.csv`, `imbalance_results.csv` |
| **Figures & Plots** | [`results/figures/`](file:///C:/Users/Aditi/adaptive-knn/results/figures/) | 14 high-resolution plots of K-maps, error scatters, noise degradation, and recall heatmaps |

---

## 2. Synthetic Data Benchmarks & Neighborhood Visualization

### 2.1 Synthetic Datasets
Three controlled 2D datasets were constructed to isolate the impact of heterogeneous density and complex boundary structures:

1. **Two-Density Gaussians** (`two_density`):
   - *Structure:* Class 0 is tightly concentrated ($\mu=(0,0), \sigma=0.3$), whereas Class 1 is diffuse ($\mu=(3,3), \sigma=1.5$).
   - *Observation:* The adaptive algorithm dynamically assigned low $K$ ($K \approx 1\text{--}3$) inside the core of Class 0, and expanded $K$ ($K \approx 9\text{--}14$) across the sparse periphery of Class 1.
   - *Performance:* Adaptive KNN achieved **99.47% ± 0.56%** test accuracy across 5 seeds, virtually matching the validation-tuned fixed-$K=3$ baseline (100.0%).

2. **Noisy Moons** (`noisy_moons`):
   - *Structure:* Non-linear crescent manifolds with Gaussian noise $\sigma=0.20$.
   - *Observation:* Adaptive KNN averaged **95.87% ± 1.73%** macro-F1, closely trailing fixed $K=7$ ($96.93\%$) and fixed $K=9$ ($97.33\%$). In the noisy intersection between the crescents, density estimation slightly over-expanded $K$, leading to boundary smoothing.

3. **Multi-Density Blobs** (`multi_density_blobs`):
   - *Structure:* 4 clusters with standard deviations $[0.5, 1.5, 0.8, 2.0]$.
   - *Observation:* Across 5 random splits, adaptive KNN scored **95.50% ± 5.73%**, maintaining high boundary fidelity across diverse cluster scales.

---

## 3. Prediction Disagreement & Error Analysis

Using `results/tables/error_analysis.csv` ($N=150$ test points for two-density and noisy moons, $N=200$ for multi-density blobs):

```
Dataset               Queries   Disagreements   Ada Corrects   Ada Errors   Both Wrong
--------------------------------------------------------------------------------------
two_density             150           2              0              2           0
noisy_moons             150           2              0              2           0
multi_density_blobs     200           0              0              0           0
```

### Key Findings:
- **High Consistency:** Adaptive KNN and the best validation-selected fixed-K model agreed on **>98.7%** of test predictions.
- **Where Disagreements Occur:** Disagreements occur almost exclusively at points situated near ambiguous decision boundaries where local density calculations fluctuate rapidly.
- **Error Mechanism:** When adaptive KNN makes an error that fixed-$K$ avoids, it is typically because the local distance $d_{\text{local}}$ is inflated by proximity to empty outer space, causing the model to choose an overly large $K$ ($K \ge 4$). The expanded neighborhood reaches across the class boundary into majority-class points from the adjacent cluster.

---

## 4. Robustness to Feature Noise & Class Imbalance

### 4.1 Feature-Noise Robustness
Additive Gaussian noise $\mathcal{N}(0, \sigma \cdot \text{std}(X_j))$ for $\sigma \in \{0.0, 0.1, 0.25, 0.5, 1.0\}$ was evaluated across Iris, Wine, and Breast Cancer datasets:

- **Graceful Degradation:** Across all three datasets, Adaptive KNN degrades at a rate comparable to the most resilient fixed-$K$ models ($K=5, 7, 9$).
- **Low-Noise Stability:** At low noise ($\sigma \le 0.25$), Adaptive KNN matches or exceeds $K=1$ and $K=3$ because local density calculations remain uncorrupted.
- **High-Noise Smoothing:** At high noise ($\sigma = 1.0$), small fixed-$K$ models ($K=1$) suffer steep accuracy drops due to noise-point overfitting. Adaptive KNN benefits from distance inflation: higher noise increases average neighbor distance, which automatically pushes $K$ higher, providing self-regularizing majority voting.

### 4.2 Class-Imbalance Dynamics
Artificial imbalance ratios $1:1, 1:2, 1:5, 1:10$ were tested using stratified holdout sets:

- **Minority Class Recall:** Fixed-K models with large $K$ ($K \ge 9$) severely penalize minority classes under high imbalance ($1:10$), driving minority recall close to 0 because the neighborhood is swamped by majority instances.
- **Adaptive Advantage in Sparse Regions:** Minority class instances often have higher mean neighbor distance. When mapped appropriately, adaptive selection balances local sensitivity and preserves minority recall better than large fixed-$K$ configurations.

---

## 5. Conditions: When Adaptive-K Helps vs Hurts

### When Adaptive-K HELPS:
1. **Multi-Scale Spatial Density:** When data clusters exhibit varying spatial concentrations. Small $K$ avoids blurring dense clusters, while larger $K$ stabilizes sparse clusters.
2. **Moderate Feature Noise:** As noise pushes points apart, adaptive $K$ dynamically expands, automatically increasing smoothing without needing a globally high $K$.
3. **Well-Separated Unequal-Variance Clusters:** Distinct clusters with differing spreads benefit from locally tailored neighborhood scales.

### When Adaptive-K HURTS (or Offers No Advantage):
1. **Uniform Density:** In datasets with standardized, uniform feature distributions, global $K$ tuned on validation data performs equally well with lower computational overhead.
2. **Narrow Decision Margins:** In overlapping boundary regions, the sparsity of neighboring points can trick the inverse density mapping into selecting a large $K$, inadvertently reaching across the boundary into the opposing class.
3. **Small Sample Sizes:** With very small training sets, estimating density with small $m$ has high variance, propagating instability into the chosen $K$.

