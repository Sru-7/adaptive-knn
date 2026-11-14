"""
Member 2: Conventional baselines and evaluation framework.

Run from the repository root:
    python -m experiments.baselines

Models:
- Uniform fixed-K KNN for K_VALUES
- Distance-weighted fixed-K KNN for K_VALUES
- Logistic Regression (validation-tuned C)
- RBF-SVM (validation-tuned C)

Protocol:
- Stratified train/validation/test split for each dataset and seed.
- Imputation/scaling are fitted inside Pipeline on training data only.
- Model selection uses validation macro-F1 only.
- Final metrics are computed on the held-out test set after refitting
  selected models on train + validation data.

Outputs:
- results/tables/baseline_validation_results.csv
- results/tables/baseline_test_results.csv
- results/tables/baseline_summary_mean_std.csv
- results/figures/baseline_macro_f1_<dataset>.png
- results/figures/baseline_confusion_<dataset>_<model>.png
"""

import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.utils import SEEDS, K_VALUES, ensure_dirs, FIGURES_DIR, TABLES_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

VALIDATION_SIZE_FROM_POOL = 0.25  # 60% train, 20% validation, 20% test overall
LOGISTIC_C_VALUES = [0.1, 1.0, 10.0]
SVM_C_VALUES = [0.1, 1.0, 10.0]
RANDOM_STATE = 42


def load_datasets():
    """Load the three project datasets as {name: (X, y)}."""
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
    }
    return {
        name: loader(return_X_y=True)
        for name, loader in loaders.items()
    }


def split_data(X, y, seed):
    """Create stratified train/validation/test partitions."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_pipeline(estimator):
    """Fit imputation and scaling only on the current training partition."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def classification_metrics(y_true, y_pred):
    """Return a consistent set of classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def fit_predict_timed(estimator, X_train, y_train, X_eval):
    """Fit estimator and separately measure fit and prediction time."""
    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = estimator.predict(X_eval)
    predict_seconds = time.perf_counter() - start

    return predictions, fit_seconds, predict_seconds


def _validation_select(dataset_name, seed, X_train, y_train, X_val, y_val):
    """Evaluate candidate models on validation data and select configurations."""
    rows = []
    selected = {}

    # Uniform and distance-weighted KNN across all requested K values.
    for weights in ("uniform", "distance"):
        for k in K_VALUES:
            if k > len(y_train):
                continue
            name = f"knn_{weights}_k{k}"
            model = make_pipeline(
                KNeighborsClassifier(n_neighbors=k, weights=weights)
            )
            pred, fit_s, pred_s = fit_predict_timed(
                model, X_train, y_train, X_val
            )
            metrics = classification_metrics(y_val, pred)
            rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "model": name,
                "family": "knn",
                "k": k,
                "weights": weights,
                "C": np.nan,
                **metrics,
                "fit_time_sec": fit_s,
                "predict_time_sec": pred_s,
            })

    # Select best global K and voting strategy by validation macro-F1.
    knn_rows = [r for r in rows if r["family"] == "knn"]
    best_knn = max(knn_rows, key=lambda r: r["macro_f1"])
    selected["best_knn"] = {
        "k": int(best_knn["k"]),
        "weights": best_knn["weights"],
        "name": best_knn["model"],
    }

    # Logistic Regression: tune C on validation macro-F1.
    for c in LOGISTIC_C_VALUES:
        name = f"logistic_regression_c{c:g}"
        model = make_pipeline(
            LogisticRegression(C=c, max_iter=3000, random_state=seed)
        )
        pred, fit_s, pred_s = fit_predict_timed(
            model, X_train, y_train, X_val
        )
        metrics = classification_metrics(y_val, pred)
        rows.append({
            "dataset": dataset_name, "seed": seed, "model": name,
            "family": "logistic_regression", "k": np.nan,
            "weights": "", "C": c, **metrics,
            "fit_time_sec": fit_s, "predict_time_sec": pred_s,
        })

    lr_rows = [r for r in rows if r["family"] == "logistic_regression"]
    best_lr = max(lr_rows, key=lambda r: r["macro_f1"])
    selected["logistic_regression"] = {"C": float(best_lr["C"])}

    # RBF SVM: tune C on validation macro-F1; gamma is fixed to 'scale'.
    for c in SVM_C_VALUES:
        name = f"rbf_svm_c{c:g}"
        model = make_pipeline(SVC(C=c, kernel="rbf", gamma="scale"))
        pred, fit_s, pred_s = fit_predict_timed(
            model, X_train, y_train, X_val
        )
        metrics = classification_metrics(y_val, pred)
        rows.append({
            "dataset": dataset_name, "seed": seed, "model": name,
            "family": "rbf_svm", "k": np.nan,
            "weights": "", "C": c, **metrics,
            "fit_time_sec": fit_s, "predict_time_sec": pred_s,
        })

    svm_rows = [r for r in rows if r["family"] == "rbf_svm"]
    best_svm = max(svm_rows, key=lambda r: r["macro_f1"])
    selected["rbf_svm"] = {"C": float(best_svm["C"])}

    return rows, selected


def _final_test_evaluation(
    dataset_name, seed, X_train, X_val, X_test, y_train, y_val, y_test, selected
):
    """Refit selected models on train+validation and evaluate on test once."""
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    specs = []
    for weights in ("uniform", "distance"):
        for k in K_VALUES:
            if k <= len(y_fit):
                specs.append((
                    f"knn_{weights}_k{k}",
                    KNeighborsClassifier(n_neighbors=k, weights=weights),
                    "knn",
                    k,
                    weights,
                    np.nan,
                ))

    specs.extend([
        (
            f"logistic_regression_c{selected['logistic_regression']['C']:g}",
            LogisticRegression(
                C=selected["logistic_regression"]["C"],
                max_iter=3000,
                random_state=seed,
            ),
            "logistic_regression", np.nan, "",
            selected["logistic_regression"]["C"],
        ),
        (
            f"rbf_svm_c{selected['rbf_svm']['C']:g}",
            SVC(
                C=selected["rbf_svm"]["C"],
                kernel="rbf",
                gamma="scale",
            ),
            "rbf_svm", np.nan, "",
            selected["rbf_svm"]["C"],
        ),
    ])

    rows = []
    predictions_by_model = {}
    for name, estimator, family, k, weights, c in specs:
        model = make_pipeline(estimator)
        pred, fit_s, pred_s = fit_predict_timed(model, X_fit, y_fit, X_test)
        metrics = classification_metrics(y_test, pred)
        rows.append({
            "dataset": dataset_name,
            "seed": seed,
            "model": name,
            "family": family,
            "k": k,
            "weights": weights,
            "C": c,
            **metrics,
            "fit_time_sec": fit_s,
            "predict_time_sec": pred_s,
        })
        predictions_by_model[name] = pred

    # Explicitly mark which KNN configuration was selected on validation.
    for row in rows:
        row["selected_on_validation"] = (
            row["model"] == selected["best_knn"]["name"]
            or row["family"] in ("logistic_regression", "rbf_svm")
        )
    return rows, predictions_by_model


def _save_confusion_matrix(dataset_name, model_name, y_true, y_pred, seed):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{dataset_name} — {model_name} (seed={seed})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(
        FIGURES_DIR,
        f"baseline_confusion_{dataset_name}_{safe_name}_seed{seed}.png",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_summary_plot(test_df):
    """Plot mean test macro-F1 for each model and dataset."""
    for dataset_name in test_df["dataset"].unique():
        sub = test_df[test_df["dataset"] == dataset_name]
        summary = (
            sub.groupby("model", as_index=False)["macro_f1"]
            .mean()
            .sort_values("macro_f1", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=summary, x="model", y="macro_f1", ax=ax)
        ax.set_title(f"{dataset_name} — Baseline Test Macro-F1")
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean test macro-F1 across seeds")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = os.path.join(
            FIGURES_DIR, f"baseline_macro_f1_{dataset_name}.png"
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_baseline_experiments():
    ensure_dirs()
    datasets = load_datasets()
    validation_rows = []
    test_rows = []

    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'=' * 65}\nDataset: {dataset_name}\n{'=' * 65}")

        for seed in SEEDS:
            (
                X_train, X_val, X_test,
                y_train, y_val, y_test,
            ) = split_data(X, y, seed)

            val_rows, selected = _validation_select(
                dataset_name, seed, X_train, y_train, X_val, y_val
            )
            validation_rows.extend(val_rows)

            print(
                f"seed={seed}: selected global KNN = "
                f"{selected['best_knn']['name']}; "
                f"LR C={selected['logistic_regression']['C']}; "
                f"SVM C={selected['rbf_svm']['C']}"
            )

            final_rows, predictions = _final_test_evaluation(
                dataset_name, seed,
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                selected,
            )
            test_rows.extend(final_rows)

            # Save confusion matrices for selected KNN, LR and SVM.
            selected_names = [
                selected["best_knn"]["name"],
                f"logistic_regression_c{selected['logistic_regression']['C']:g}",
                f"rbf_svm_c{selected['rbf_svm']['C']:g}",
            ]
            for model_name in selected_names:
                _save_confusion_matrix(
                    dataset_name, model_name, y_test,
                    predictions[model_name], seed,
                )

    validation_df = pd.DataFrame(validation_rows)
    test_df = pd.DataFrame(test_rows)

    validation_path = os.path.join(
        TABLES_DIR, "baseline_validation_results.csv"
    )
    test_path = os.path.join(TABLES_DIR, "baseline_test_results.csv")
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = (
        test_df.groupby(["dataset", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_macro_mean=("precision_macro", "mean"),
            recall_macro_mean=("recall_macro", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            fit_time_mean_sec=("fit_time_sec", "mean"),
            predict_time_mean_sec=("predict_time_sec", "mean"),
        )
    )
    summary_path = os.path.join(
        TABLES_DIR, "baseline_summary_mean_std.csv"
    )
    summary.to_csv(summary_path, index=False)

    _save_summary_plot(test_df)

    print("\nBaseline experiments complete.")
    print(f"Validation results: {validation_path}")
    print(f"Test results:       {test_path}")
    print(f"Summary:            {summary_path}")
    print("\nMean test macro-F1:")
    print(
        summary[["dataset", "model", "macro_f1_mean", "macro_f1_std"]]
        .sort_values(["dataset", "macro_f1_mean"], ascending=[True, False])
        .to_string(index=False)
    )

    return validation_df, test_df, summary


if __name__ == "__main__":
    run_baseline_experiments()
