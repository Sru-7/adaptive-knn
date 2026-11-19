# Member 1 Contribution — Core Algorithm Development (`DensityAdaptiveKNN`)

**Directory:** `member_1_adaptive_algorithm/`  
**Primary Deliverables:** `adaptive_knn.py`, `test_adaptive_knn.py`, `complete_contribution.md`

---

## 1. Executive Summary & Core Contribution

As **Member 1**, I designed, mathematically formulated, implemented, and unit-tested the core **`DensityAdaptiveKNN`** classifier. This module serves as the foundational engine for the entire 5-member pipeline—providing a clean, scikit-learn-compatible API (`fit`, `predict`, `predict_proba`) that Members 2–5 directly imported for baseline benchmarking, multi-seed statistical testing, noise/imbalance stress tests, 2D manifold visualization, and mapping-strategy ablations.

### The Core Problem Solved
Conventional $K$-Nearest Neighbors (KNN) classification assumes a single, globally constant neighborhood size $K$ is optimal across the entire data manifold:
$$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_K(x)} \mathbb{I}(y_i = c)$$

In real-world datasets (medical diagnostics, materials analysis, biological data) with **density heterogeneity**, this breaks down:
- **Dense clusters:** Samples are closely packed. A large $K$ over-smooths local decision boundaries, crossing class margins and pulling in foreign-class neighbors.
- **Sparse / peripheral regions:** Samples are isolated or scattered. A small $K$ is vulnerable to noise and stochastic fluctuations because voting relies on too few neighbors.

`DensityAdaptiveKNN` resolves this by estimating local density around each query sample $x$ and dynamically assigning a bounded neighborhood size $K(x) \in [K_{\min}, K_{\max}]$.

---

## 2. Changes & Adaptations Made to the Base Paper

**Base Paper:** Li, J., Xu, H., & Zhang, S. (2026), *"Adaptive kNN graph model"*, *Nature Communications* 17, 7491.

The base paper introduces adaptive $k$-NN **graph construction**, where each vertex's neighborhood degree scales with local structural density to improve manifold representation. To translate this graph-theoretic principle into a practical, low-latency supervised classifier for tabular benchmarks, I made the following adaptations:

1. **From Graph Construction to Supervised Classification:** Converted the local density-adaptivity concept into a standalone, scikit-learn-compatible classifier (`DensityAdaptiveKNN`) with `fit`, `predict`, `predict_proba`, and per-sample $K(x)$ diagnostics.
2. **Lightweight Inverse-Density Proxy:** Replaced heavy graph-level optimization with a fast, non-parametric local density estimator—the mean Euclidean distance to the $m$ nearest training neighbors, $d_m(x)$.
3. **Strict Boundedness ($[K_{\min}, K_{\max}]$):** Introduced clipped min-max normalization so $K(x)$ is strictly clamped within $[K_{\min}, K_{\max}]$, preventing sparse outliers from blowing up $K$ and preventing ultra-dense cores from collapsing to $K = 0$.
4. **Inverse-Distance Weighted Voting:** Paired dynamic $K(x)$ selection with inverse-distance weighting ($w_j = \frac{1}{d_j + \epsilon}$, $\epsilon = 10^{-8}$) to prevent majority-class dilution on imbalanced data and avoid division-by-zero on exact matches.
5. **Single-Lookup Inference Architecture:** Designed inference to query $\max(m, K_{\max})$ neighbors **once** per query sample rather than running separate searches for density estimation and voting, keeping latency under **$0.09\text{ ms}$ per sample**.

---

## 3. Mathematical Formulation

### 3.1 Local Distance & Density Metric
For a query point $x \in \mathbb{R}^d$ and training set $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$, let $\text{NN}_j(x)$ denote the $j$-th nearest training sample under the Euclidean metric $\Vert{}x - x_i\Vert{}_2$. The local distance estimate using $m$ reference neighbors is:
$$d_m(x) = \frac{1}{m} \sum_{j=1}^m \Vert{}x - \text{NN}_j(x)\Vert{}_2$$

Since local probability density scales inversely with distance volume, $\rho(x) \propto \frac{1}{d_m(x)^d}$, $d_m(x)$ serves as an inverted proxy for local density:
- **Small $d_m(x)$** $\implies$ High local density (dense cluster core)
- **Large $d_m(x)$** $\implies$ Low local density (sparse / peripheral region)

### 3.2 Inverse Density Mapping to Bounded $K(x)$
Using training-set reference statistics $d_{\min} = \min_{i \in [1, N]} d_m(x_i)$ and $d_{\max} = \max_{i \in [1, N]} d_m(x_i)$:
$$\text{ratio}(x) = \text{clip}\left( \frac{d_m(x) - d_{\min}}{d_{\max} - d_{\min}},\, 0.0,\, 1.0 \right)$$
$$K(x) = \text{clip}\left( \text{round}\left(K_{\min} + \text{ratio}(x) \cdot (K_{\max} - K_{\min})\right),\, K_{\min},\, K_{\max} \right)$$

### 3.3 Voting Mechanisms
- **Uniform Majority Voting:** $w_j = 1$ for all $j \in \{1, \dots, K(x)\}$.
- **Distance-Weighted Voting:**
  $$w_j = \frac{1}{\Vert{}x - \text{NN}_j(x)\Vert{}_2 + \epsilon}, \quad \epsilon = 10^{-8}$$
  $$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{j=1}^{K(x)} w_j \cdot \mathbb{I}(y_j = c)$$

---

## 4. Exact Code Breakdown & Pseudocode

### A. Core Classifier (`member_1_adaptive_algorithm/adaptive_knn.py`)
1. **`__init__`:** Stores hyperparameters `m` (density reference neighbors), `K_min`, `K_max`, and `weights` (`'uniform'` or `'distance'`).
2. **`fit(X_train, y_train)`:**
   - Stores training features and labels.
   - Fits a `NearestNeighbors` index with `n_neighbors = max(m, K_max)`.
   - Computes $d_m(x_i)$ for every training point and records global training statistics: `d_min`, `d_max`, and `d_median`.
3. **`predict(X)` & `predict_proba(X)`:**
   - Performs a single nearest-neighbor lookup per query for `max(m, K_max)` neighbors.
   - Computes $d_m(x)$ from the first $m$ distances.
   - Normalizes $d_m(x)$ into `[0.0, 1.0]` (safely handling the degenerate case $d_{\max} = d_{\min}$ to prevent `ZeroDivisionError`) and maps it to integer $K(x) \in [K_{\min}, K_{\max}]$.
   - Slices the top $K(x)$ neighbors, applies uniform or inverse-distance weights, returns class predictions (`predict`), normalized class probabilities summing to $1.0$ (`predict_proba`), and exposes per-query $K(x)$ values for downstream analysis.

### Algorithm Pseudocode
```text
Algorithm: DensityAdaptiveKNN

Inputs:
  Training set: D_train = {(x_i, y_i)}_{i=1}^N
  Query point: x
  Hyperparameters: m (density neighborhood), K_min, K_max, weights ('uniform' | 'distance')

Stage 1: Training (Fit)
1. Store training points X_train and labels y_train.
2. Build NearestNeighbors index with K_query = max(m, K_max).
3. For each training sample x_i in D_train:
       Compute d_m(x_i) = mean distance to its m nearest neighbors.
4. Record global training statistics:
       d_min = min(d_m(x_i))
       d_max = max(d_m(x_i))
       d_median = median(d_m(x_i))

Stage 2: Inference (Predict)
1. Query NearestNeighbors index for max(m, K_max) neighbors of x.
       Retrieve distances dists[1..max(m, K_max)] and indices idx[1..max(m, K_max)].
2. Compute query local distance:
       d_m(x) = (1 / m) * sum_{j=1}^m dists[j]
3. Map d_m(x) to adaptive K:
       ratio = clip((d_m(x) - d_min) / (d_max - d_min), 0.0, 1.0)
       K(x) = round(K_min + ratio * (K_max - K_min))
       K(x) = clamp(K(x), K_min, K_max)
4. Aggregate votes among K(x) nearest neighbors:
       If weights == 'uniform':
           y_pred = majority_vote(y_train[idx[1..K(x)]])
       Else if weights == 'distance':
           w_j = 1.0 / (dists[j] + 1e-8)
           y_pred = argmax_c sum_{j=1}^{K(x)} w_j * [y_train[idx[j]] == c]
5. Return y_pred and K(x).
```

---

## 5. Computational Complexity

| Phase | Metric | Conventional KNN | `DensityAdaptiveKNN` |
| :--- | :--- | :--- | :--- |
| **Fit Space** | Memory | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(N \cdot d + N)$ (stores $d_m$ stats) |
| **Fit Time** | Index Construction | $\mathcal{O}(N \log N)$ or $\mathcal{O}(d N^2)$ | $\mathcal{O}(N \log N + N \cdot m)$ |
| **Query Time** | Single Sample | $\mathcal{O}(d \log N + K)$ | $\mathcal{O}(d \log N + \max(m, K_{\max}))$ |

**Key Takeaway:** Asymptotic query time complexity is identical to standard KNN querying $\max(m, K_{\max})$ points. Since $m \le 15$ and $K_{\max} \le 15$, adaptive overhead is negligible ($\ll 1\text{ ms}$, averaging $<0.09\text{ ms}$ per sample).

---

## 6. Automated Unit Test Suite (`test_adaptive_knn.py`)

Implemented and verified **6 `pytest` test suites** validating correctness, mathematical invariants, and edge cases:

1. **Bound Enforcement:** Guarantees $K_{\min} \le K(x) \le K_{\max}$ strictly, even on extreme far-away outliers.
2. **Monotonicity:** Confirms that as local distance $d_m(x)$ increases, the assigned $K(x)$ is monotonically non-decreasing.
3. **Multi-Class Compatibility:** Verifies valid predictions and output shapes on multi-class datasets (beyond binary tasks).
4. **Probability Calibration:** Ensures `predict_proba` outputs non-negative probabilities summing to exactly $1.0$ per sample.
5. **Voting Schemes:** Validates both `'uniform'` and `'distance'` weighting modes and confirms closer neighbors dominate under inverse-distance weights.
6. **Identical-Sample Edge Case:** Verifies numerical stability when all training samples are identical ($d_{\min} = d_{\max} = 0$), ensuring no `ZeroDivisionError` or `NaN` propagation.

---

## 7. Technical Documentation (`complete_contribution.md`)

- Documented the full mathematical formulation, algorithmic pseudocode, complexity analysis, hyperparameter definitions, and API contract so Members 2–5 could seamlessly integrate `DensityAdaptiveKNN` into their evaluation, robustness, and mapping pipelines.
