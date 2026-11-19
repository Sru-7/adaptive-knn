"""
Member 1: Unit Tests for DensityAdaptiveKNN
===========================================
Tests core algorithmic properties:
1. Bound enforcement: k_min <= K(x) <= k_max.
2. Monotonicity of mapping functions.
3. Prediction shape and valid class sets.
4. Support for uniform vs distance-weighted voting.
5. Multi-class target support.
6. Probability calibration (sums to 1.0).
7. Extreme edge cases (zero distances, single queries).
"""

import pytest
import numpy as np
from member_1_adaptive_algorithm.adaptive_knn import DensityAdaptiveKNN, linear_mapping, quantile_mapping, sigmoid_mapping


def test_adaptive_knn_basic_fit_predict():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(100, 4))
    y_train = rng.choice([0, 1], size=100)
    X_test = rng.normal(size=(20, 4))

    clf = DensityAdaptiveKNN(m=5, k_min=3, k_max=11)
    clf.fit(X_train, y_train)

    preds, k_vals = clf.predict_with_k(X_test)
    assert len(preds) == len(X_test)
    assert len(k_vals) == len(X_test)
    assert set(preds).issubset({0, 1})
    assert np.all(k_vals >= 3)
    assert np.all(k_vals <= 11)


def test_adaptive_k_bounds_strictness():
    rng = np.random.default_rng(123)
    X_train = rng.uniform(0, 1, size=(80, 3))
    y_train = rng.choice([0, 1, 2], size=80)
    # Test queries with extreme outliers (e.g. 1000 units away)
    X_test = np.vstack([rng.uniform(0, 1, size=(10, 3)), np.ones((5, 3)) * 1000.0])

    clf = DensityAdaptiveKNN(m=3, k_min=1, k_max=7)
    clf.fit(X_train, y_train)
    preds, k_vals = clf.predict_with_k(X_test)

    assert np.all(k_vals >= 1)
    assert np.all(k_vals <= 7)


def test_monotonicity_of_mapping():
    """Verify that larger local distance leads to larger or equal K."""
    min_dist, max_dist = 0.5, 5.0
    k_min, k_max = 1, 15

    distances = np.linspace(min_dist, max_dist, 20)
    k_results = [linear_mapping(d, min_dist, max_dist, k_min, k_max) for d in distances]

    # Verify monotonic non-decreasing
    for i in range(len(k_results) - 1):
        assert k_results[i] <= k_results[i + 1]


def test_distance_weighted_voting():
    # Construct an unambiguous 1D scenario
    # Training: x=0 (class 0), x=1 (class 1), x=1.1 (class 1)
    # Query at x=0.01: closest point is class 0 with very small distance.
    X_train = np.array([[0.0], [1.0], [1.1], [1.2]])
    y_train = np.array([0, 1, 1, 1])
    X_query = np.array([[0.01]])

    # Distance-weighted with k=3
    clf = DensityAdaptiveKNN(m=2, k_min=3, k_max=3, weights="distance")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_query)
    assert pred[0] == 0  # Close neighbor dominates despite 2 farther opposing neighbors


def test_multiclass_and_predict_proba():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(60, 5))
    y = np.repeat([0, 1, 2], 20)

    clf = DensityAdaptiveKNN(m=4, k_min=2, k_max=8, weights="uniform")
    clf.fit(X, y)

    probas = clf.predict_proba(X[:10])
    assert probas.shape == (10, 3)
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)


def test_edge_case_identical_points():
    """When all training samples are identical, min_dist == max_dist."""
    X_train = np.ones((20, 2))
    y_train = np.zeros(20)
    X_test = np.ones((5, 2))

    clf = DensityAdaptiveKNN(m=3, k_min=3, k_max=9)
    clf.fit(X_train, y_train)
    preds, k_vals = clf.predict_with_k(X_test)
    assert np.all(preds == 0)
    assert np.all(k_vals == 3)
