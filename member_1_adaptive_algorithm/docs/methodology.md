# Member 1 — Adaptive-K Algorithm Development

## 1. Executive Summary & Algorithmic Concept

Conventional $K$-Nearest Neighbors (KNN) classification assumes that a single, globally constant neighborhood size $K$ is optimal for the entire data manifold:
$$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_K(x)} \mathbb{I}(y_i = c)$$

However, real-world datasets—such as medical diagnostics, materials analysis, and biological data—exhibit substantial **density heterogeneity**:
- **Dense clusters**: Samples are closely packed. A large $K$ over-smooths local decision boundaries, crossing class boundaries and pulling in irrelevant or foreign-class neighbors.
- **Sparse / peripheral regions**: Samples are isolated or scattered. A small $K$ is vulnerable to noise and stochastic fluctuations because voting is based on too few samples.

Inspired by Li et al. (*Nature Communications* 17, 7491, 2026), our lightweight adaptive-K framework estimates local density around each query sample $x$ and dynamically assigns a neighborhood size $K(x) \in [K_{\min}, K_{\max}]$.

---

## 2. Mathematical Formulation

### 2.1 Local Distance & Density Metric
For a query point $x \in \mathbb{R}^d$ and training set $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$, let $\text{NN}_j(x)$ denote the $j$-th nearest training sample under Euclidean metric $\|x - x_i\|_2$.

The local distance estimate using $m$ reference neighbors is:
$$d_m(x) = \frac{1}{m} \sum_{j=1}^m \|x - \text{NN}_j(x)\|_2$$

Because local density $\rho(x) \propto \frac{1}{d_m(x)^d}$, $d_m(x)$ serves as an inverted proxy for density:
- Small $d_m(x) \implies$ High local density
- Large $d_m(x) \implies$ Low local density (sparse region)

### 2.2 Inverse Density Mapping to Bounded $K(x)$
To provide noise resilience in sparse regions while preserving high-resolution boundaries in dense regions, we formulate the inverse density mapping:

$$\text{ratio}(x) = \text{clip}\left( \frac{d_m(x) - d_{\min}}{d_{\max} - d_{\min}}, 0, 1 \right)$$
$$K(x) = \text{clip}\left( \text{round}\left(K_{\min} + \text{ratio}(x) \cdot (K_{\max} - K_{\min})\right), K_{\min}, K_{\max} \right)$$

where $d_{\min} = \min_{i \in [1, N]} d_m(x_i)$ and $d_{\max} = \max_{i \in [1, N]} d_m(x_i)$ are precomputed on the training set.

### 2.3 Distance-Weighted Voting
In addition to standard uniform majority voting, distance-weighted voting calculates evidence weights:
$$w_j = \frac{1}{\|x - \text{NN}_j(x)\|_2 + \epsilon}, \quad \epsilon = 10^{-8}$$
$$\hat{y}(x) = \arg\max_{c \in \mathcal{C}} \sum_{j=1}^{K(x)} w_j \cdot \mathbb{I}(y_j = c)$$

---

## 3. Algorithm Pseudocode

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

## 4. Computational Complexity

| Phase | Metric | Conventional KNN | DensityAdaptiveKNN |
| :--- | :--- | :--- | :--- |
| **Fit Space** | Memory | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(N \cdot d + N)$ (stores $d_m$ stats) |
| **Fit Time** | Index Construction | $\mathcal{O}(N \log N)$ or $\mathcal{O}(d N^2)$ | $\mathcal{O}(N \log N + N \cdot m)$ |
| **Query Time** | Single sample | $\mathcal{O}(d \log N + K)$ | $\mathcal{O}(d \log N + \max(m, K_{\max}))$ |

**Key Takeaway**: The asymptotic query time complexity is identical to standard KNN querying $\max(m, K_{\max})$ points. Because $m \le 15$ and $K_{\max} \le 15$, overhead is negligible ($\ll 1$ millisecond per sample).
