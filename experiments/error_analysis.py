"""
Prediction-disagreement and error analysis between adaptive-K and fixed-K baselines.

For each synthetic dataset:
  1. Identify points where adaptive-K and the best fixed-K disagree.
  2. Classify disagreements into:
       • Adaptive corrects fixed   (adaptive right, fixed wrong)
       • Adaptive introduces error (adaptive wrong, fixed right)
       • Both wrong, different prediction
  3. Report per-category counts and average K values.
  4. Save disagreement scatter plots.
  5. Export summary CSV.

Usage (from project root):
    python -m experiments.error_analysis
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adaptive_knn import DensityAdaptiveKNN
from experiments.synthetic_data import load_all_synthetic
from experiments.utils import (
    predict_with_k_values, evaluate_model,
    K_VALUES, ensure_dirs, FIGURES_DIR, TABLES_DIR,
)

SEED = 42                                   # single seed for qualitative plots


# -----------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------
def analyse_disagreements(y_true, y_pred_ada, y_pred_fixed, k_vals):
    """Return masks and summary dict for the three disagreement categories."""
    disagree = y_pred_ada != y_pred_fixed

    ada_corrects  = disagree & (y_pred_ada == y_true) & (y_pred_fixed != y_true)
    ada_errors    = disagree & (y_pred_ada != y_true) & (y_pred_fixed == y_true)
    both_wrong    = disagree & (y_pred_ada != y_true) & (y_pred_fixed != y_true)

    def _stats(mask):
        n = int(mask.sum())
        avg_k = float(np.mean(k_vals[mask])) if n > 0 else float("nan")
        std_k = float(np.std(k_vals[mask])) if n > 0 else float("nan")
        return {"count": n, "avg_k": avg_k, "std_k": std_k}

    summary = {
        "total_queries":        len(y_true),
        "total_disagree":       int(disagree.sum()),
        "adaptive_corrects":    _stats(ada_corrects),
        "adaptive_errors":      _stats(ada_errors),
        "both_wrong_different": _stats(both_wrong),
    }

    masks = {
        "agree":              ~disagree,
        "adaptive_corrects":  ada_corrects,
        "adaptive_errors":    ada_errors,
        "both_wrong":         both_wrong,
    }
    return masks, summary


# -----------------------------------------------------------------------
# Disagreement scatter plot
# -----------------------------------------------------------------------
def _plot_disagreements(X, y_true, masks, k_vals, ds_name):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ---------- Left panel: disagreement category ----------
    ax = axes[0]
    # background: agreeing points in light grey
    agree = masks["agree"]
    ax.scatter(X[agree, 0], X[agree, 1], c="lightgrey", s=12, alpha=0.4, label="Agree")

    colors  = {"adaptive_corrects": "green", "adaptive_errors": "red", "both_wrong": "orange"}
    labels  = {"adaptive_corrects": "Ada corrects fixed",
               "adaptive_errors":   "Ada introduces error",
               "both_wrong":        "Both wrong (different)"}
    markers = {"adaptive_corrects": "^", "adaptive_errors": "v", "both_wrong": "s"}

    for cat in ["adaptive_corrects", "adaptive_errors", "both_wrong"]:
        m = masks[cat]
        if m.sum() == 0:
            continue
        ax.scatter(
            X[m, 0], X[m, 1],
            c=colors[cat], marker=markers[cat], s=50, edgecolors="k",
            linewidths=0.5, alpha=0.85, label=labels[cat],
        )

    ax.set_title(f"{ds_name} — Prediction Disagreements")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(fontsize=8, loc="best")

    # ---------- Right panel: K chosen at disagreement points ----------
    ax2 = axes[1]
    disagree_mask = ~masks["agree"]
    # background
    ax2.scatter(X[agree, 0], X[agree, 1], c="lightgrey", s=12, alpha=0.3)
    if disagree_mask.sum() > 0:
        sc = ax2.scatter(
            X[disagree_mask, 0], X[disagree_mask, 1],
            c=k_vals[disagree_mask], cmap="viridis", s=50,
            edgecolors="k", linewidths=0.5, vmin=1, vmax=15,
        )
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label("Adaptive K at disagreement")
    ax2.set_title(f"{ds_name} — K at Disagreement Points")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"disagreement_{ds_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Disagreement plot saved: {path}")


# -----------------------------------------------------------------------
# K-distribution histogram for disagree vs agree
# -----------------------------------------------------------------------
def _plot_k_distribution(k_vals, masks, ds_name):
    fig, ax = plt.subplots(figsize=(8, 4))

    agree_k    = k_vals[masks["agree"]]
    disagree_k = k_vals[~masks["agree"]]

    bins = np.arange(0.5, 16.5, 1)
    ax.hist(agree_k, bins=bins, alpha=0.6, label="Agree", color="steelblue", edgecolor="k")
    if len(disagree_k) > 0:
        ax.hist(disagree_k, bins=bins, alpha=0.6, label="Disagree", color="tomato", edgecolor="k")

    ax.set_xlabel("Adaptive K")
    ax.set_ylabel("Count")
    ax.set_title(f"{ds_name} — K Distribution (Agree vs Disagree)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"k_dist_{ds_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ K-distribution plot saved: {path}")


# -----------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------
def run_error_analysis():
    ensure_dirs()
    datasets = load_all_synthetic(random_state=SEED)
    summary_rows = []

    for ds_name, (X_train, X_val, X_test, y_train, y_val, y_test) in datasets.items():
        print(f"\n=== {ds_name} ===")

        # ---- Adaptive-K ----
        ada = DensityAdaptiveKNN(m=5, k_min=1, k_max=15)
        ada.fit(X_train, y_train)
        y_pred_ada, k_vals = predict_with_k_values(ada, X_test)

        # ---- Best fixed-K (by validation macro-F1) ----
        best_val_f1, best_k = -1, K_VALUES[0]
        for k in K_VALUES:
            fknn = KNeighborsClassifier(n_neighbors=k)
            fknn.fit(X_train, y_train)
            vf1 = evaluate_model(y_val, fknn.predict(X_val))["macro_f1"]
            if vf1 > best_val_f1:
                best_val_f1, best_k = vf1, k

        fknn_best = KNeighborsClassifier(n_neighbors=best_k)
        fknn_best.fit(X_train, y_train)
        y_pred_fixed = fknn_best.predict(X_test)

        print(f"  Best fixed K = {best_k}")

        # ---- Analyse disagreements ----
        masks, summary = analyse_disagreements(y_test, y_pred_ada, y_pred_fixed, k_vals)

        # Print text summary
        print(f"  Total test points : {summary['total_queries']}")
        print(f"  Disagreements     : {summary['total_disagree']}")
        for cat in ["adaptive_corrects", "adaptive_errors", "both_wrong_different"]:
            s = summary[cat]
            print(f"    {cat:25s}: n={s['count']:3d}   avg_K={s['avg_k']:.1f}  std_K={s['std_k']:.1f}")

        # ---- Metrics comparison ----
        m_ada = evaluate_model(y_test, y_pred_ada)
        m_fix = evaluate_model(y_test, y_pred_fixed)
        print(f"  Adaptive  —  acc={m_ada['accuracy']:.4f}  f1={m_ada['macro_f1']:.4f}")
        print(f"  Fixed K={best_k:2d} —  acc={m_fix['accuracy']:.4f}  f1={m_fix['macro_f1']:.4f}")

        # ---- Plots ----
        _plot_disagreements(X_test, y_test, masks, k_vals, ds_name)
        _plot_k_distribution(k_vals, masks, ds_name)

        # ---- Row for CSV ----
        summary_rows.append({
            "dataset": ds_name,
            "best_fixed_k": best_k,
            "total_queries": summary["total_queries"],
            "disagreements": summary["total_disagree"],
            "ada_corrects":  summary["adaptive_corrects"]["count"],
            "ada_errors":    summary["adaptive_errors"]["count"],
            "both_wrong":    summary["both_wrong_different"]["count"],
            "acc_adaptive":  m_ada["accuracy"],
            "f1_adaptive":   m_ada["macro_f1"],
            "acc_fixed":     m_fix["accuracy"],
            "f1_fixed":      m_fix["macro_f1"],
        })

    # ---- Save CSV ----
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(TABLES_DIR, "error_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Error-analysis summary saved to {csv_path}")

    return df


# -----------------------------------------------------------------------
if __name__ == "__main__":
    run_error_analysis()

