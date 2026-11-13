"""
Shared utilities for robustness, class-imbalance, synthetic benchmarks, and error analysis experiments.

Provides:
  - predict_with_k_values : mirrors DensityAdaptiveKNN.predict but also
    returns the adaptive K chosen for every query point.
  - evaluate_model        : computes accuracy, macro-F1, balanced accuracy.
  - SEEDS / K_VALUES      : project-wide constants.
  - ensure_dirs           : create output folders if missing.
"""

import os
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


# ---------------------------------------------------------------------------
# Project-wide constants
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024]
K_VALUES = [1, 3, 5, 7, 9, 11, 15]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")


def ensure_dirs():
    """Create results/figures and results/tables if they don't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_with_k_values(model, X):
    """
    Run DensityAdaptiveKNN prediction and return both predictions
    and the per-query adaptive K values.

    Parameters
    ----------
    model : DensityAdaptiveKNN (already fitted)
    X     : array-like of shape (n_queries, n_features)

    Returns
    -------
    predictions : np.ndarray of shape (n_queries,)
    k_values    : np.ndarray of shape (n_queries,)  — int K per query
    """
    X = np.array(X)
    predictions = []
    k_values = []

    distances, indices = model.nn_.kneighbors(X, n_neighbors=model.max_neighbors)

    for i in range(len(X)):
        # 1. Local-density estimate (mean dist to m nearest neighbours)
        local_distance = model._estimate_density(distances[i, : model.m])

        # 2. Map distance → adaptive K via the model's mapping function
        adaptive_k = model.mapping_func(
            local_distance,
            model.global_min_dist_,
            model.global_max_dist_,
            model.k_min,
            model.k_max,
        )
        adaptive_k = max(model.k_min, min(adaptive_k, model.k_max))

        # 3. Majority vote among the adaptive_k nearest neighbours
        neighbor_classes = model.y_train_[indices[i, :adaptive_k]]
        most_common = Counter(neighbor_classes).most_common(1)[0][0]

        predictions.append(most_common)
        k_values.append(adaptive_k)

    return np.array(predictions), np.array(k_values)


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------
def evaluate_model(y_true, y_pred):
    """
    Return a dict of standard classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

