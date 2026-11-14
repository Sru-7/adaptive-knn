"""
Member 2: Baseline Models & Evaluation Framework
================================================

Implements the standard baseline suite and unified evaluation pipeline:
- Fixed-K KNN for K in [1, 3, 5, 7, 9, 11, 15] (both uniform and distance-weighted)
- Global K validation tuning (select best K on validation set)
- Logistic Regression (L2 penalty, scaled features)
- Support Vector Classifier (both Linear and RBF kernels)
- DensityAdaptiveKNN (Member 1)
- Unified metric calculation: Accuracy, Macro-F1, Balanced Accuracy, Macro Precision,
  Macro Recall, Inference Latency.
"""

import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

K_GRID = [1, 3, 5, 7, 9, 11, 15]


def tune_fixed_knn(X_train, y_train, X_val, y_val, weights="uniform"):
    """
    Find best K on validation set among K_GRID based on macro-F1.
    """
    best_k = K_GRID[0]
    best_score = -1.0
    for k in K_GRID:
        if k > len(X_train):
            continue
        knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_val)
        score = f1_score(y_val, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def evaluate_predictions(y_true, y_pred, latency_ms=0.0):
    """Compute comprehensive classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "latency_ms": float(latency_ms),
    }


def get_all_models(X_train, y_train, X_val, y_val):
    """
    Instantiates all baseline models and tuned variants.
    """
    best_k_uniform = tune_fixed_knn(X_train, y_train, X_val, y_val, weights="uniform")
    best_k_distance = tune_fixed_knn(X_train, y_train, X_val, y_val, weights="distance")

    models = {}

    # 1. Tuned KNN Baselines
    models[f"Tuned Fixed KNN (K={best_k_uniform}, uniform)"] = KNeighborsClassifier(
        n_neighbors=best_k_uniform, weights="uniform"
    )
    models[f"Tuned Fixed KNN (K={best_k_distance}, distance)"] = KNeighborsClassifier(
        n_neighbors=best_k_distance, weights="distance"
    )

    # 2. Fixed K=1, 3, 5, 7, 9, 11, 15 (Uniform)
    for k in K_GRID:
        if k <= len(X_train):
            models[f"Fixed KNN (K={k}, uniform)"] = KNeighborsClassifier(
                n_neighbors=k, weights="uniform"
            )

    # 3. Fixed K=1, 3, 5, 7, 9, 11, 15 (Distance-weighted)
    for k in K_GRID:
        if k <= len(X_train):
            models[f"Fixed KNN (K={k}, distance)"] = KNeighborsClassifier(
                n_neighbors=k, weights="distance"
            )

    # 4. Parametric / Linear Baselines
    models["Logistic Regression"] = LogisticRegression(
        max_iter=1000, random_state=42
    )
    models["Linear SVM"] = SVC(
        kernel="linear", random_state=42
    )
    models["RBF SVM"] = SVC(
        kernel="rbf", random_state=42
    )

    return models, best_k_uniform, best_k_distance
