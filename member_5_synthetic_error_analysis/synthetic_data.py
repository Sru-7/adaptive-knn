"""
Member 5: Controlled Synthetic Data Generator
============================================

Generates 2D synthetic manifolds designed to stress-test adaptive vs fixed neighborhoods:
1. Two-density Gaussians: Dense cluster vs sparse cluster.
2. Noisy Moons: Intertwined crescents with non-linear boundary and noise.
3. Multi-density Blobs: Multiple Gaussian clusters with varying spread.
"""

import numpy as np
from sklearn.datasets import make_moons, make_blobs


def make_two_density_data(n_samples=600, random_state=42):
    """
    Class 0: tight Gaussian at (0, 0), sigma=0.35 (dense)
    Class 1: spread Gaussian at (2.5, 2.5), sigma=1.40 (sparse)
    """
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2
    X0 = rng.normal(loc=[0.0, 0.0], scale=0.35, size=(n_half, 2))
    X1 = rng.normal(loc=[2.5, 2.5], scale=1.40, size=(n_samples - n_half, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_half, dtype=int), np.ones(n_samples - n_half, dtype=int)])
    return X, y


def make_noisy_moons_data(n_samples=600, noise=0.22, random_state=42):
    """Interlocking moons with noise."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def make_multi_density_blobs_data(n_samples=800, random_state=42):
    """Four clusters with varying variances."""
    X, y = make_blobs(
        n_samples=n_samples,
        centers=[(-3, -3), (3, -3), (-3, 3), (3, 3)],
        cluster_std=[0.5, 1.6, 0.8, 2.2],
        random_state=random_state
    )
    # Binary labels: 0 for diagonal 1, 1 for diagonal 2
    y_bin = ((y == 0) | (y == 3)).astype(int)
    return X, y_bin
