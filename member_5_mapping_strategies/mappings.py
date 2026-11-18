"""
Member 5: Algorithmic Mapping Strategies & Theoretical Formulations
===================================================================

Provides alternative mathematical transformations from local distance d_m(x)
to neighborhood size K(x):

1. Linear Inverse Density Mapping:
   K_lin(d) = round( K_min + ((d - d_min) / (d_max - d_min)) * (K_max - K_min) )

2. Quantile / Rank-Based Mapping:
   K_quant(d) = round( K_min + F_emp(d) * (K_max - K_min) )
   where F_emp is the empirical CDF over training distances.
   Key property: Scale-invariant and completely robust to outlier stretch.

3. Sigmoid / Threshold Non-Linear Mapping:
   K_sig(d) = round( K_min + (1 / (1 + exp(-2 * (d - d_median) / IQR))) * (K_max - K_min) )
   Key property: High sensitivity near the boundary/median, flat saturation at tails.
"""

import numpy as np


def linear_mapping(local_dist, min_dist, max_dist, k_min, k_max, **kwargs):
    """Linear scaling between min_dist and max_dist."""
    if max_dist <= min_dist:
        return int(k_min)
    ratio = (local_dist - min_dist) / (max_dist - min_dist)
    ratio = np.clip(ratio, 0.0, 1.0)
    k = k_min + ratio * (k_max - k_min)
    return int(np.clip(np.round(k), k_min, k_max))


def quantile_mapping(local_dist, min_dist=None, max_dist=None, k_min=1, k_max=15, **kwargs):
    """Quantile / Rank mapping using empirical CDF of training set distances."""
    train_distances = kwargs.get("train_distances", None)
    if train_distances is None and isinstance(min_dist, (np.ndarray, list)):
        train_distances = min_dist
    if train_distances is None or len(train_distances) == 0:
        return int(k_min)
    percentile = float(np.mean(train_distances <= local_dist))
    k = k_min + percentile * (k_max - k_min)
    return int(np.clip(np.round(k), k_min, k_max))


def sigmoid_mapping(local_dist, min_dist=None, max_dist=None, k_min=1, k_max=15, **kwargs):
    """Sigmoidal smooth transition centered at median distance."""
    median_dist = kwargs.get("median_dist", None)
    if median_dist is None:
        median_dist = (min_dist + max_dist) / 2.0 if (min_dist is not None and max_dist is not None) else 1.0
    iqr_dist = kwargs.get("iqr_dist", 1.0)
    scale = iqr_dist if (iqr_dist is not None and iqr_dist > 1e-7) else 1.0
    z = (local_dist - median_dist) / scale
    sig = 1.0 / (1.0 + np.exp(-2.0 * z))
    k = k_min + sig * (k_max - k_min)
    return int(np.clip(np.round(k), k_min, k_max))


def threshold_mapping(local_dist, min_dist=None, max_dist=None, k_min=1, k_max=15, **kwargs):
    """Step threshold: assigns k_min if below median, k_max if above."""
    median_dist = kwargs.get("median_dist", None)
    if median_dist is None:
        median_dist = (min_dist + max_dist) / 2.0 if (min_dist is not None and max_dist is not None) else local_dist
    if local_dist <= median_dist:
        return int(k_min)
    return int(k_max)
