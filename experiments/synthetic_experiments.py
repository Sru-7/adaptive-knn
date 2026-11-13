"""
Synthetic-data experiments: Adaptive-K vs Fixed-K benchmarks and K-value maps.

For each synthetic dataset:
  1. Train adaptive-K KNN and every fixed-K baseline.
  2. Select the best fixed-K by validation macro-F1.
  3. Report test accuracy, macro-F1, balanced accuracy.
  4. Save K-value heatmap scatter plots.
  5. Export results table to CSV.

Usage (from project root):
    python -m experiments.synthetic_experiments
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless-safe backend
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ---- project imports ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adaptive_knn import DensityAdaptiveKNN
from experiments.synthetic_data import load_all_synthetic
from experiments.utils import (
    predict_with_k_values, evaluate_model,
    SEEDS, K_VALUES, ensure_dirs, FIGURES_DIR, TABLES_DIR,
)


# -----------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------
def run_synthetic_experiments():
    ensure_dirs()
    all_rows = []

    for seed in SEEDS:
        datasets = load_all_synthetic(random_state=seed)

        for ds_name, (X_train, X_val, X_test, y_train, y_val, y_test) in datasets.items():
            # ---- Adaptive-K KNN ----
            ada = DensityAdaptiveKNN(m=5, k_min=1, k_max=15)
            ada.fit(X_train, y_train)
            y_pred_ada, k_vals = predict_with_k_values(ada, X_test)
            metrics_ada = evaluate_model(y_test, y_pred_ada)
            metrics_ada.update({
                "model": "adaptive_knn",
                "dataset": ds_name,
                "seed": seed,
                "k_mean": float(np.mean(k_vals)),
                "k_std":  float(np.std(k_vals)),
            })
            all_rows.append(metrics_ada)

            # ---- Fixed-K baselines (pick best K on validation set) ----
            best_val_f1, best_k = -1, K_VALUES[0]
            for k in K_VALUES:
                fknn = KNeighborsClassifier(n_neighbors=k)
                fknn.fit(X_train, y_train)
                y_val_pred = fknn.predict(X_val)
                val_f1 = evaluate_model(y_val, y_val_pred)["macro_f1"]
                if val_f1 > best_val_f1:
                    best_val_f1, best_k = val_f1, k

            for k in K_VALUES:
                fknn = KNeighborsClassifier(n_neighbors=k)
                fknn.fit(X_train, y_train)
                y_pred_f = fknn.predict(X_test)
                metrics_f = evaluate_model(y_test, y_pred_f)
                metrics_f.update({
                    "model": f"fixed_k={k}" + (" (best)" if k == best_k else ""),
                    "dataset": ds_name,
                    "seed": seed,
                    "k_mean": float(k),
                    "k_std": 0.0,
                })
                all_rows.append(metrics_f)

            # ---- K-value heatmap plot (only for first seed to avoid clutter) ----
            if seed == SEEDS[0]:
                _plot_k_heatmap(X_test, y_test, k_vals, ds_name)

    # ---- Save results CSV ----
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(TABLES_DIR, "synthetic_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")

    # ---- Print summary (mean ± std across seeds) ----
    _print_summary(df)

    return df


# -----------------------------------------------------------------------
# K-value heatmap
# -----------------------------------------------------------------------
def _plot_k_heatmap(X, y_true, k_values, ds_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: true labels
    scatter0 = axes[0].scatter(
        X[:, 0], X[:, 1], c=y_true, cmap="Set1", s=18, alpha=0.75, edgecolors="k", linewidths=0.3
    )
    axes[0].set_title(f"{ds_name} — True Labels")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    legend0 = axes[0].legend(
        *scatter0.legend_elements(), title="Class", loc="upper left"
    )
    axes[0].add_artist(legend0)

    # Right: adaptive K value per query
    sc = axes[1].scatter(
        X[:, 0], X[:, 1], c=k_values, cmap="viridis", s=18, alpha=0.75,
        edgecolors="k", linewidths=0.3, vmin=1, vmax=15,
    )
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label("Adaptive K")
    axes[1].set_title(f"{ds_name} — Chosen K per Query")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"k_map_{ds_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ K-map saved: {path}")


# -----------------------------------------------------------------------
# Summary printer
# -----------------------------------------------------------------------
def _print_summary(df):
    print("\n" + "=" * 80)
    print("SYNTHETIC-DATA EXPERIMENT SUMMARY  (mean ± std over 5 seeds)")
    print("=" * 80)

    for ds_name in df["dataset"].unique():
        print(f"\n--- {ds_name} ---")
        sub = df[df["dataset"] == ds_name]
        for model in sub["model"].unique():
            m = sub[sub["model"] == model]
            acc  = f'{m["accuracy"].mean():.4f} ± {m["accuracy"].std():.4f}'
            f1   = f'{m["macro_f1"].mean():.4f} ± {m["macro_f1"].std():.4f}'
            ba   = f'{m["balanced_accuracy"].mean():.4f} ± {m["balanced_accuracy"].std():.4f}'
            print(f"  {model:25s}  acc={acc}  f1={f1}  bal_acc={ba}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    run_synthetic_experiments()

