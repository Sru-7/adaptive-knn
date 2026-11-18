"""
Member 3: Dataset Loading and Preprocessing Pipeline
===================================================

Provides standardized, reproducible loading and preprocessing for:
1. Breast Cancer Wisconsin (Diagnostic) — Scikit-learn built-in
2. Glass Identification — UCI Machine Learning Repository
3. Heart Disease (Cleveland) — UCI Machine Learning Repository

Features:
- Missing value imputation (median / mode)
- Class label normalization
- Train / Validation / Test stratified splitting
- Strict leakage-free standard scaling (fit on train, transform val/test)
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Path to local data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_breast_cancer_dataset():
    """Load Breast Cancer Wisconsin Diagnostic dataset."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    return {
        "name": "Breast Cancer Wisconsin",
        "key": "breast_cancer",
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_names": target_names,
        "n_classes": 2,
    }


def load_glass_dataset():
    """
    Load UCI Glass Identification dataset.
    214 samples, 9 chemical attributes, 6 glass types.
    Column 0 is ID (dropped).
    """
    csv_path = os.path.join(DATA_DIR, "glass.data")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing Glass dataset at {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    # Column 0: ID (drop)
    # Columns 1-9: Features (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
    # Column 10: Target class (1, 2, 3, 5, 6, 7)
    feature_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
    X = df.iloc[:, 1:10].to_numpy(dtype=np.float64)
    raw_y = df.iloc[:, 10].to_numpy(dtype=int)

    # Remap classes 1,2,3,5,6,7 to 0..5 for zero-indexed consecutive labels
    unique_classes = np.unique(raw_y)
    class_map = {c: i for i, c in enumerate(unique_classes)}
    y = np.array([class_map[c] for c in raw_y])

    target_names = [f"Class_{c}" for c in unique_classes]
    return {
        "name": "UCI Glass Identification",
        "key": "glass",
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_names": target_names,
        "n_classes": len(unique_classes),
    }


def load_heart_disease_dataset():
    """
    Load UCI Heart Disease (Cleveland) dataset.
    303 samples, 13 clinical features.
    Target binarized: 0 = Absence, 1 = Presence of heart disease.
    Missing values in 'ca' and 'thal' imputed using median.
    """
    csv_path = os.path.join(DATA_DIR, "processed.cleveland.data")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing Heart Disease dataset at {csv_path}")

    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    df = pd.read_csv(csv_path, header=None, names=feature_names + ["target"], na_values="?")

    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(df[feature_names])

    # Binarize target: 0 vs (1, 2, 3, 4) -> 0 vs 1
    raw_y = df["target"].to_numpy()
    y = (raw_y > 0).astype(int)

    return {
        "name": "UCI Heart Disease (Cleveland)",
        "key": "heart_disease",
        "X": X_imputed,
        "y": y,
        "feature_names": feature_names,
        "target_names": ["Absence", "Presence"],
        "n_classes": 2,
    }


def get_all_datasets():
    """Returns a dict of all 3 benchmark datasets."""
    return {
        "breast_cancer": load_breast_cancer_dataset(),
        "glass": load_glass_dataset(),
        "heart_disease": load_heart_disease_dataset(),
    }


def get_train_val_test_splits(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Split dataset into reproducible train (60%), validation (20%), and test (20%) sets.
    Features are scaled using StandardScaler fitted strictly on X_train.
    """
    assert np.isclose(train_size + val_size + test_size, 1.0)
    
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Then split temp into train and validation
    val_ratio_of_temp = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, random_state=random_state, stratify=y_temp
    )

    # Standard scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
    }
