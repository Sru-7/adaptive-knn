"""
Class-imbalance and minority-class recall experiments.

For each dataset × imbalance ratio × seed:
  1. Artificially undersample minority classes to the desired ratio.
  2. Train adaptive-K KNN and all fixed-K baselines.
  3. Record macro-F1, balanced accuracy, and per-class recall.
  4. Save plots (grouped bar / heatmap) and a results CSV.

Usage (from project root):
    python -m experiments.robustness_imbalance
"""

import sys, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adaptive_knn import DensityAdaptiveKNN
from experiments.utils import (
    evaluate_model, SEEDS, K_VALUES, ensure_dirs, FIGURES_DIR, TABLES_DIR,
)

IMBALANCE_RATIOS = [1, 2, 5, 10]   # 1:1, 1:2, 1:5, 1:10 (minority:majority)


# -----------------------------------------------------------------------
# Built-in datasets
# -----------------------------------------------------------------------
def _load_builtin_datasets():
    return {
        "iris":          load_iris(return_X_y=True),
        "wine":          load_wine(return_X_y=True),
        "breast_cancer": load_breast_cancer(return_X_y=True),
    }


# -----------------------------------------------------------------------
# Artificial imbalance creator
# -----------------------------------------------------------------------
def create_imbalanced_data(X, y, ratio, rng):
    """
    Keep all samples of the majority class.
    Undersample every other class to  n_majority / ratio  samples.

    Parameters
    ----------
    ratio : int  —  majority / minority target ratio (1 = balanced).
    rng   : np.random.Generator
    """
    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    n_majority = counts.max()
    n_minority_target = max(1, n_majority // ratio)   # at least 1 sample

    indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if cls == majority_class:
            indices.append(cls_idx)
        else:
            if len(cls_idx) <= n_minority_target:
                indices.append(cls_idx)                # already small enough
            else:
                chosen = rng.choice(cls_idx, size=n_minority_target, replace=False)
                indices.append(chosen)

    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


# -----------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------
def run_imbalance_experiments():
    ensure_dirs()
    all_datasets = _load_builtin_datasets()
    all_rows = []

    t_start = time.time()

    for ds_name, (X_raw, y) in all_datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  (n={len(y)}, classes={np.unique(y)})")
        print(f"{'='*60}")

        for seed in SEEDS:
            rng = np.random.default_rng(seed)

            # --- hold out a balanced test set FIRST ---
            X_pool, X_test, y_pool, y_test = train_test_split(
                X_raw, y, test_size=0.25, random_state=seed, stratify=y,
            )

            for ratio in IMBALANCE_RATIOS:
                # Create imbalanced *training* data
                X_train_imb, y_train_imb = create_imbalanced_data(
                    X_pool, y_pool, ratio, np.random.default_rng(seed + ratio)
                )

                # Scale
                scaler = StandardScaler().fit(X_train_imb)
                X_train = scaler.transform(X_train_imb)
                X_test_s = scaler.transform(X_test)

                # Unique classes for per-class recall
                classes = np.unique(y)

                # ---- Adaptive KNN ----
                ada = DensityAdaptiveKNN(m=5, k_min=1, k_max=15)
                ada.fit(X_train, y_train_imb)
                y_pred = ada.predict(X_test_s)
                metrics = evaluate_model(y_test, y_pred)
                per_class_recall = recall_score(y_test, y_pred, labels=classes,
                                                average=None, zero_division=0)
                metrics.update({
                    "model": "adaptive_knn",
                    "dataset": ds_name,
                    "imbalance_ratio": ratio,
                    "seed": seed,
                    "train_size": len(y_train_imb),
                })
                for ci, cls in enumerate(classes):
                    metrics[f"recall_class_{cls}"] = per_class_recall[ci]
                all_rows.append(metrics)

                # ---- Fixed-K baselines ----
                for k in K_VALUES:
                    if k > len(y_train_imb):
                        continue                      # can't have K > n_train
                    fknn = KNeighborsClassifier(n_neighbors=k)
                    fknn.fit(X_train, y_train_imb)
                    y_pred_f = fknn.predict(X_test_s)
                    mf = evaluate_model(y_test, y_pred_f)
                    pcr = recall_score(y_test, y_pred_f, labels=classes,
                                       average=None, zero_division=0)
                    mf.update({
                        "model": f"fixed_k={k}",
                        "dataset": ds_name,
                        "imbalance_ratio": ratio,
                        "seed": seed,
                        "train_size": len(y_train_imb),
                    })
                    for ci, cls in enumerate(classes):
                        mf[f"recall_class_{cls}"] = pcr[ci]
                    all_rows.append(mf)

            print(f"  seed={seed} done")

    elapsed = time.time() - t_start
    print(f"\nAll experiments finished in {elapsed:.1f}s")

    # ---- Save CSV ----
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(TABLES_DIR, "imbalance_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")

    # ---- Plots ----
    _plot_imbalance_macro_f1(df)
    _plot_per_class_recall_heatmap(df)

    return df


# -----------------------------------------------------------------------
# Plot 1: Macro-F1 vs imbalance ratio
# -----------------------------------------------------------------------
def _plot_imbalance_macro_f1(df):
    for ds_name in df["dataset"].unique():
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = df[df["dataset"] == ds_name]
        agg = sub.groupby(["model", "imbalance_ratio"])["macro_f1"].mean().reset_index()

        # Adaptive
        ada = agg[agg["model"] == "adaptive_knn"].sort_values("imbalance_ratio")
        ax.plot(ada["imbalance_ratio"], ada["macro_f1"],
                "o-", linewidth=2.5, markersize=7, color="crimson",
                label="Adaptive KNN", zorder=5)

        # Fixed-K
        cmap_colors = cm.Blues(np.linspace(0.35, 0.9, len(K_VALUES)))
        for idx, k in enumerate(K_VALUES):
            fk = agg[agg["model"] == f"fixed_k={k}"].sort_values("imbalance_ratio")
            if len(fk) == 0:
                continue
            ax.plot(fk["imbalance_ratio"], fk["macro_f1"],
                    "s--", linewidth=1.2, markersize=4, color=cmap_colors[idx],
                    label=f"K={k}", alpha=0.85)

        ax.set_xlabel("Imbalance ratio (majority : minority)")
        ax.set_ylabel("Macro F1")
        ax.set_title(f"{ds_name} — Macro F1 vs Class Imbalance")
        ax.set_xticks(IMBALANCE_RATIOS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"imbalance_f1_{ds_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Plot saved: {path}")


# -----------------------------------------------------------------------
# Plot 2: Per-class recall heatmap (adaptive vs best fixed)
# -----------------------------------------------------------------------
def _plot_per_class_recall_heatmap(df):
    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name]
        classes = sorted([c for c in sub.columns if c.startswith("recall_class_")])
        if not classes:
            continue

        # Select adaptive and best overall fixed-K (highest mean macro-F1)
        agg_model = sub.groupby("model")["macro_f1"].mean()
        fixed_models = [m for m in agg_model.index if m.startswith("fixed_k=")]
        best_fixed = max(fixed_models, key=lambda m: agg_model[m]) if fixed_models else None

        models_to_plot = ["adaptive_knn"]
        if best_fixed:
            models_to_plot.append(best_fixed)

        fig, axes = plt.subplots(1, len(models_to_plot), figsize=(6 * len(models_to_plot), 4))
        if len(models_to_plot) == 1:
            axes = [axes]

        for ax, model in zip(axes, models_to_plot):
            msub = sub[sub["model"] == model]
            # rows = imbalance ratios, cols = classes
            pivot = msub.groupby("imbalance_ratio")[classes].mean()
            heatmap_data = pivot.values
            row_labels = [f"1:{r}" for r in pivot.index]
            col_labels = [c.replace("recall_class_", "Class ") for c in classes]

            im = ax.imshow(heatmap_data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, fontsize=9)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=9)
            ax.set_xlabel("Class")
            ax.set_ylabel("Imbalance ratio")
            ax.set_title(f"{model}", fontsize=10)

            # Annotate cells
            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    val = heatmap_data[i, j]
                    color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

        fig.suptitle(f"{ds_name} — Per-Class Recall Heatmap", fontsize=12)
        plt.colorbar(im, ax=axes, shrink=0.8, label="Recall")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"imbalance_recall_heatmap_{ds_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Heatmap saved: {path}")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    run_imbalance_experiments()

