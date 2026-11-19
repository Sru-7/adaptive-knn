"""
Member 1: Adaptive-K Nearest Neighbors Classifier
==================================================

Implementation of Density-Based Adaptive-K Nearest Neighbors (DensityAdaptiveKNN).

Key algorithmic concept:
- Instead of using a fixed global neighborhood size K for all query points,
  the algorithm computes the local data density around each query point x
  by measuring the mean distance to its m nearest training samples.
- The estimated local density/distance is mapped to an adaptive neighborhood
  size K(x) bounded between [k_min, k_max].
- Sparse regions (larger distances to neighbors) receive larger K to accumulate
  statistical evidence and suppress noise.
- Dense regions (smaller distances to neighbors) receive smaller K to preserve
  fine decision boundary details and prevent over-smoothing.
- Voting supports both 'uniform' and 'distance'-weighted schemes.
"""

import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def linear_mapping(local_dist, min_dist, max_dist, k_min, k_max, **kwargs):
    """
    Standard linear inverse density mapping:
    Dense region (small dist) -> K approaches k_min
    Sparse region (large dist) -> K approaches k_max
    """
    if max_dist <= min_dist:
        return int(k_min)
    ratio = (local_dist - min_dist) / (max_dist - min_dist)
    ratio = np.clip(ratio, 0.0, 1.0)
    k = k_min + ratio * (k_max - k_min)
    return int(np.clip(np.round(k), k_min, k_max))


def quantile_mapping(local_dist, min_dist=None, max_dist=None, k_min=1, k_max=15, **kwargs):
    """
    Quantile / Rank-based mapping:
    Maps local distance to K based on the empirical CDF of training set local distances.
    Robust to extreme distance outliers.
    """
    train_distances = kwargs.get("train_distances", None)
    if train_distances is None and isinstance(min_dist, (np.ndarray, list)):
        train_distances = min_dist
    if train_distances is None or len(train_distances) == 0:
        return int(k_min)
    percentile = float(np.mean(train_distances <= local_dist))
    k = k_min + percentile * (k_max - k_min)
    return int(np.clip(np.round(k), k_min, k_max))


def sigmoid_mapping(local_dist, min_dist=None, max_dist=None, k_min=1, k_max=15, **kwargs):
    """
    Smooth non-linear sigmoid mapping centered at median training distance.
    """
    median_dist = kwargs.get("median_dist", None)
    if median_dist is None:
        median_dist = (min_dist + max_dist) / 2.0 if (min_dist is not None and max_dist is not None) else 1.0
    iqr_dist = kwargs.get("iqr_dist", 1.0)
    scale = iqr_dist if (iqr_dist is not None and iqr_dist > 1e-7) else 1.0
    z = (local_dist - median_dist) / scale
    sigmoid_val = 1.0 / (1.0 + np.exp(-2.0 * z))
    k = k_min + sigmoid_val * (k_max - k_min)
    return int(np.clip(np.round(k), k_min, k_max))


class DensityAdaptiveKNN(BaseEstimator, ClassifierMixin):
    """
    Density-Based Adaptive-K Nearest Neighbors Classifier.

    Parameters
    ----------
    m : int, default=5
        Number of nearest training samples used to compute the local density estimate.
    k_min : int, default=1
        Minimum allowed neighborhood size K.
    k_max : int, default=15
        Maximum allowed neighborhood size K.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    mapping_strategy : {'linear', 'quantile', 'sigmoid'} or callable, default='linear'
        Function used to map local distance to neighborhood size K.
    metric : str, default='euclidean'
        Distance metric for nearest neighbors.
    power : float, default=1.0
        Exponent for distance weighting: w = 1 / (d + eps)^power.
    purity_alpha : float, default=0.0
        Boundary purity scaling exponent. If > 0, compresses K towards k_min
        when initial m neighbors exhibit mixed class labels (boundary regions).
    """

    def __init__(
        self,
        m=5,
        k_min=1,
        k_max=15,
        weights="uniform",
        mapping_strategy="linear",
        metric="euclidean",
        power=1.0,
        purity_alpha=0.0,
    ):
        self.m = m
        self.k_min = k_min
        self.k_max = k_max
        self.weights = weights
        self.mapping_strategy = mapping_strategy
        self.metric = metric
        self.power = power
        self.purity_alpha = purity_alpha

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values."""
        X, y = check_X_y(X, y, accept_sparse=False)
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        self.n_samples_ = self.X_train_.shape[0]

        # Ensure bounds are valid
        if self.k_min > self.k_max:
            raise ValueError(f"k_min ({self.k_min}) cannot exceed k_max ({self.k_max})")
        if self.k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {self.k_min}")
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")

        effective_k_max = min(self.k_max, self.n_samples_)
        effective_m = min(self.m, self.n_samples_)
        self.max_neighbors_ = max(effective_m, effective_k_max)

        # Build NearestNeighbors index
        self.nn_ = NearestNeighbors(
            n_neighbors=self.max_neighbors_,
            metric=self.metric,
            algorithm="auto"
        )
        self.nn_.fit(self.X_train_)

        # Compute training set local distances using m neighbors
        train_dists, _ = self.nn_.kneighbors(self.X_train_, n_neighbors=effective_m)
        self.train_local_dists_ = np.mean(train_dists, axis=1)

        self.global_min_dist_ = float(np.min(self.train_local_dists_))
        self.global_max_dist_ = float(np.max(self.train_local_dists_))
        self.global_median_dist_ = float(np.median(self.train_local_dists_))
        q75, q25 = np.percentile(self.train_local_dists_, [75, 25])
        self.global_iqr_dist_ = float(q75 - q25)

        return self

    def _get_adaptive_k(self, local_dist):
        """Compute bounded adaptive K for a given local distance."""
        k_min = self.k_min
        k_max = min(self.k_max, self.n_samples_)

        if callable(self.mapping_strategy):
            k = self.mapping_strategy(
                local_dist,
                self.global_min_dist_,
                self.global_max_dist_,
                k_min,
                k_max,
                train_distances=self.train_local_dists_,
                median_dist=self.global_median_dist_,
                iqr_dist=self.global_iqr_dist_,
            )
        elif self.mapping_strategy == "quantile":
            k = quantile_mapping(
                local_dist,
                self.global_min_dist_,
                self.global_max_dist_,
                k_min,
                k_max,
                train_distances=self.train_local_dists_,
            )
        elif self.mapping_strategy == "sigmoid":
            k = sigmoid_mapping(
                local_dist,
                self.global_median_dist_,
                self.global_iqr_dist_,
                k_min,
                k_max,
            )
        else:  # default 'linear'
            k = linear_mapping(
                local_dist,
                self.global_min_dist_,
                self.global_max_dist_,
                k_min,
                k_max,
            )

        # Enforce strict bounds
        return int(max(k_min, min(k, k_max)))

    def predict_with_k(self, X):
        """
        Predict classes and return the dynamically chosen K for each query sample.

        Returns
        -------
        predictions : np.ndarray of shape (n_queries,)
        k_values : np.ndarray of shape (n_queries,)
        """
        check_is_fitted(self, ["X_train_", "y_train_", "nn_"])
        X = check_array(X, accept_sparse=False)
        effective_m = min(self.m, self.n_samples_)

        distances, indices = self.nn_.kneighbors(X, n_neighbors=self.max_neighbors_)

        predictions = []
        k_values = []

        for i in range(len(X)):
            # 1. Local density: mean distance to m nearest training samples
            local_dist = float(np.mean(distances[i, :effective_m]))

            # 2. Map distance to adaptive K
            k = self._get_adaptive_k(local_dist)

            # Optional boundary purity compression
            if self.purity_alpha > 0.0:
                first_labels = self.y_train_[indices[i, :effective_m]]
                purity = Counter(first_labels).most_common(1)[0][1] / float(effective_m)
                k = int(np.round(self.k_min + (k - self.k_min) * (purity ** self.purity_alpha)))
                k = int(max(self.k_min, min(k, self.k_max)))

            k_values.append(k)

            # 3. Vote among k neighbors
            neighbor_labels = self.y_train_[indices[i, :k]]
            neighbor_dists = distances[i, :k]

            if self.weights == "distance":
                # Power-weighted inverse distance
                weights = 1.0 / ((neighbor_dists + 1e-8) ** self.power)
                score_dict = {c: 0.0 for c in self.classes_}
                for lbl, w in zip(neighbor_labels, weights):
                    score_dict[lbl] += w
                pred_label = max(score_dict.keys(), key=lambda c: score_dict[c])
            else:
                counts = Counter(neighbor_labels)
                pred_label = counts.most_common(1)[0][0]

            predictions.append(pred_label)

        return np.array(predictions), np.array(k_values)

    def predict(self, X):
        """Predict class labels for samples in X."""
        preds, _ = self.predict_with_k(X)
        return preds

    def predict_proba(self, X):
        """Return class probability estimates for samples in X."""
        check_is_fitted(self, ["X_train_", "y_train_", "nn_"])
        X = check_array(X, accept_sparse=False)
        effective_m = min(self.m, self.n_samples_)

        distances, indices = self.nn_.kneighbors(X, n_neighbors=self.max_neighbors_)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        probas = np.zeros((len(X), n_classes), dtype=np.float64)

        for i in range(len(X)):
            local_dist = float(np.mean(distances[i, :effective_m]))
            k = self._get_adaptive_k(local_dist)

            if self.purity_alpha > 0.0:
                first_labels = self.y_train_[indices[i, :effective_m]]
                purity = Counter(first_labels).most_common(1)[0][1] / float(effective_m)
                k = int(np.round(self.k_min + (k - self.k_min) * (purity ** self.purity_alpha)))
                k = int(max(self.k_min, min(k, self.k_max)))

            neighbor_labels = self.y_train_[indices[i, :k]]
            neighbor_dists = distances[i, :k]

            if self.weights == "distance":
                weights = 1.0 / ((neighbor_dists + 1e-8) ** self.power)
                for lbl, w in zip(neighbor_labels, weights):
                    probas[i, class_to_idx[lbl]] += w
                total = np.sum(probas[i])
                if total > 0:
                    probas[i] /= total
            else:
                for lbl in neighbor_labels:
                    probas[i, class_to_idx[lbl]] += 1.0
                probas[i] /= float(k)

        return probas
