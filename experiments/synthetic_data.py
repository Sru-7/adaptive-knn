"""
Synthetic dataset generators for controlled density and boundary structure analysis.

Three controlled 2-D datasets for analysing adaptive-K behaviour:

1. Two-density Gaussians  – one tight + one spread-out cluster per class.
2. Noisy moons             – crescent shapes with tuneable noise.
3. Multi-density blobs     – four blobs with deliberately different variances.
"""

import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------
# Dataset 1: Two-Density Gaussians
# -----------------------------------------------------------------------
def make_two_density(n_samples=600, random_state=42):
    """
    Class 0: tight Gaussian centred at (0, 0), σ = 0.3
    Class 1: spread Gaussian centred at (3, 3), σ = 1.5

    This forces adaptive-K to choose a small K in the dense class-0 region
    and a large K in the sparse class-1 region.
    """
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2

    X0 = rng.normal(loc=[0, 0], scale=0.3, size=(n_half, 2))
    X1 = rng.normal(loc=[3, 3], scale=1.5, size=(n_samples - n_half, 2))

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)]).astype(int)

    return X, y


# -----------------------------------------------------------------------
# Dataset 2: Noisy Moons
# -----------------------------------------------------------------------
def make_noisy_moons(n_samples=600, noise=0.20, random_state=42):
    """
    Two interleaving crescents with Gaussian noise.
    Dense cores + noisy overlap region make fixed-K fragile.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


# -----------------------------------------------------------------------
# Dataset 3: Multi-Density Blobs
# -----------------------------------------------------------------------
def make_multi_density_blobs(n_samples=800, random_state=42):
    """
    Four blobs with cluster_std = [0.5, 1.5, 0.8, 2.0].
    Each blob has a different local density, exercising the full
    range of the adaptive mapping function.
    """
    data = make_blobs(
        n_samples=n_samples,
        centers=4,
        cluster_std=[0.5, 1.5, 0.8, 2.0],
        random_state=random_state,
    )
    X, y = data[0], data[1]
    return X, y


# -----------------------------------------------------------------------
# Convenience: get all datasets as a dict
# -----------------------------------------------------------------------
def load_all_synthetic(random_state=42, test_size=0.25, val_size=0.15):
    """
    Return a dict of { name: (X_train, X_val, X_test, y_train, y_val, y_test) }
    with reproducible train / val / test splits.
    """
    generators = {
        "two_density":      lambda rs: make_two_density(n_samples=600, random_state=rs),
        "noisy_moons":      lambda rs: make_noisy_moons(n_samples=600, noise=0.20, random_state=rs),
        "multi_density_blobs": lambda rs: make_multi_density_blobs(n_samples=800, random_state=rs),
    }

    datasets = {}
    for name, gen_fn in generators.items():
        X, y = gen_fn(random_state)

        # First split: train+val vs test
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Second split: train vs val
        relative_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=relative_val, random_state=random_state, stratify=y_tv
        )

        datasets[name] = (X_train, X_val, X_test, y_train, y_val, y_test)

    return datasets


# -----------------------------------------------------------------------
# Quick sanity check
# -----------------------------------------------------------------------
if __name__ == "__main__":
    datasets = load_all_synthetic()
    for name, (Xtr, Xv, Xte, ytr, yv, yte) in datasets.items():
        print(
            f"{name:25s}  train={len(ytr):4d}  val={len(yv):3d}  "
            f"test={len(yte):3d}  classes={np.unique(ytr)}"
        )

