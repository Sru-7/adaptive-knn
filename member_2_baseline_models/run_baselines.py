"""
Member 2: Baseline Comparison Runner & Visualization
===================================================

Runs evaluation across:
1. Breast Cancer Wisconsin
2. UCI Glass Identification
3. UCI Heart Disease

Models compared:
- DensityAdaptiveKNN (Uniform)
- DensityAdaptiveKNN (Distance-Weighted)
- Tuned Fixed KNN (Uniform)
- Tuned Fixed KNN (Distance-Weighted)
- Fixed KNN for K in [1, 3, 5, 7, 9, 11, 15]
- Logistic Regression
- Linear SVM
- RBF SVM
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from member_1_adaptive_algorithm.adaptive_knn import DensityAdaptiveKNN
from member_2_baseline_models.baselines import get_all_models, evaluate_predictions, K_GRID
from member_3_dataset_preprocessing_eda.data_loader import get_all_datasets, get_train_val_test_splits

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def run_baseline_evaluation():
    print("=" * 65)
    print("Member 2: Running Baseline Models & Evaluation Framework")
    print("=" * 65)

    datasets = get_all_datasets()
    all_results = []
    k_curve_data = {k: {} for k in datasets.keys()}

    for d_key, d_info in datasets.items():
        print(f"\nEvaluating Dataset: {d_info['name']} ...")
        splits = get_train_val_test_splits(d_info["X"], d_info["y"], random_state=42)
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        X_test, y_test = splits["X_test"], splits["y_test"]

        # 1. Evaluate DensityAdaptiveKNN (Uniform)
        t0 = time.time()
        ad_knn_unif = DensityAdaptiveKNN(m=5, k_min=1, k_max=15, weights="uniform", mapping_strategy="linear")
        ad_knn_unif.fit(X_train, y_train)
        preds_ad, k_vals_ad = ad_knn_unif.predict_with_k(X_test)
        lat = (time.time() - t0) * 1000.0 / len(X_test)
        m_ad = evaluate_predictions(y_test, preds_ad, latency_ms=lat)
        all_results.append({
            "Dataset": d_info["name"],
            "Model": "Adaptive KNN (Uniform, m=5)",
            "Category": "Adaptive KNN",
            **m_ad,
            "Mean_K": round(float(np.mean(k_vals_ad)), 2),
            "Std_K": round(float(np.std(k_vals_ad)), 2),
        })

        # 2. Evaluate DensityAdaptiveKNN (Distance-weighted)
        t0 = time.time()
        ad_knn_dist = DensityAdaptiveKNN(m=5, k_min=1, k_max=15, weights="distance", mapping_strategy="linear")
        ad_knn_dist.fit(X_train, y_train)
        preds_ad_dist, k_vals_ad_dist = ad_knn_dist.predict_with_k(X_test)
        lat = (time.time() - t0) * 1000.0 / len(X_test)
        m_ad_dist = evaluate_predictions(y_test, preds_ad_dist, latency_ms=lat)
        all_results.append({
            "Dataset": d_info["name"],
            "Model": "Adaptive KNN (Distance-Weighted, m=5)",
            "Category": "Adaptive KNN",
            **m_ad_dist,
            "Mean_K": round(float(np.mean(k_vals_ad_dist)), 2),
            "Std_K": round(float(np.std(k_vals_ad_dist)), 2),
        })

        # 3. Evaluate Baselines
        baselines, best_k_u, best_k_d = get_all_models(X_train, y_train, X_val, y_val)

        for name, model in baselines.items():
            t0 = time.time()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            lat = (time.time() - t0) * 1000.0 / len(X_test)
            m_res = evaluate_predictions(y_test, preds, latency_ms=lat)

            category = "Other Baseline"
            if "Fixed KNN" in name:
                category = "Fixed KNN"
            if "Tuned" in name:
                category = "Tuned KNN"

            # Track K-curve for plotting
            for k in K_GRID:
                if name == f"Fixed KNN (K={k}, uniform)":
                    k_curve_data[d_key][k] = m_res["macro_f1"]

            all_results.append({
                "Dataset": d_info["name"],
                "Model": name,
                "Category": category,
                **m_res,
                "Mean_K": k if "Fixed KNN (K=" in name else "-",
                "Std_K": 0.0 if "Fixed KNN (K=" in name else "-",
            })

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "baseline_comparison_tables.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nAll baseline results saved to: {csv_path}")

    # Display key highlights
    highlight_models = [
        "Adaptive KNN (Distance-Weighted, m=5)",
        "Adaptive KNN (Uniform, m=5)",
        f"Tuned Fixed KNN (K={best_k_u}, uniform)",
        "Logistic Regression",
        "RBF SVM",
    ]
    print("\nSample Comparisons across Models:")
    print(results_df[results_df["Model"].isin(highlight_models)][["Dataset", "Model", "accuracy", "macro_f1", "balanced_acc"]])

    # -------------------------------------------------------------
    # Plot 1: Macro-F1 Comparison of Key Models
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    key_models = [
        "Adaptive KNN (Distance-Weighted, m=5)",
        "Adaptive KNN (Uniform, m=5)",
        "Tuned Fixed KNN (Uniform)",
        "Logistic Regression",
        "RBF SVM"
    ]
    for ax, (d_key, d_info) in zip(axes, datasets.items()):
        sub_df = results_df[results_df["Dataset"] == d_info["name"]]
        
        # Grab rows
        plot_names = []
        plot_f1s = []
        colors = []
        for km in key_models:
            match = sub_df[sub_df["Model"].str.startswith(km[:15])]
            if not match.empty:
                row = match.iloc[0]
                label = row["Model"]
                if "Tuned" in label:
                    label = f"Tuned KNN ({label.split('(')[1].split(',')[0]})"
                elif "Adaptive KNN (Distance" in label:
                    label = "Adaptive KNN (Dist)"
                elif "Adaptive KNN (Uniform" in label:
                    label = "Adaptive KNN (Unif)"
                plot_names.append(label)
                plot_f1s.append(row["macro_f1"])
                colors.append("#2ca02c" if "Adaptive" in label else "#1f77b4")

        bars = ax.barh(plot_names, plot_f1s, color=colors, alpha=0.85, edgecolor="black")
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel("Test Macro-F1 Score", fontweight="bold")
        ax.set_title(f"{d_info['name']}", fontweight="bold", fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        for bar in bars:
            w = bar.get_width()
            ax.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    f1_plot_path = os.path.join(FIG_DIR, "baseline_macro_f1_comparison.png")
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()
    print(f"Saved baseline Macro-F1 comparison plot: {f1_plot_path}")

    # -------------------------------------------------------------
    # Plot 2: K-Sensitivity Curves (Fixed K vs Adaptive Horizontal Line)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (d_key, d_info) in zip(axes, datasets.items()):
        ks = sorted(k_curve_data[d_key].keys())
        f1_vals = [k_curve_data[d_key][k] for k in ks]
        ax.plot(ks, f1_vals, marker="o", color="#1f77b4", linewidth=2, label="Fixed-K KNN (Uniform)")

        # Adaptive baseline line
        ad_sub = results_df[(results_df["Dataset"] == d_info["name"]) & (results_df["Model"].str.contains("Adaptive KNN (Distance", regex=False))]
        if not ad_sub.empty:
            ad_score = ad_sub.iloc[0]["macro_f1"]
            ax.axhline(ad_score, color="#2ca02c", linestyle="--", linewidth=2, label=f"Adaptive KNN ({ad_score:.3f})")

        ax.set_title(f"{d_info['name']}: K vs Performance", fontweight="bold", fontsize=11)
        ax.set_xlabel("Neighborhood Size K")
        ax.set_ylabel("Test Macro-F1")
        ax.set_xticks(ks)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9)

    plt.tight_layout()
    curve_plot_path = os.path.join(FIG_DIR, "fixed_k_vs_adaptive_curve.png")
    plt.savefig(curve_plot_path, dpi=300)
    plt.close()
    print(f"Saved Fixed-K vs Adaptive curve plot: {curve_plot_path}")

    # Write documentation
    doc_path = os.path.join(DOCS_DIR, "baselines_documentation.md")
    with open(doc_path, "w") as f:
        f.write(f"""# Member 2 — Baseline Models & Evaluation Framework

## 1. Overview of Evaluated Models

We benchmarked the proposed `DensityAdaptiveKNN` against a full suite of competitive baseline classifiers:
1. **Fixed-K KNN (Uniform Voting):** $K \\in [1, 3, 5, 7, 9, 11, 15]$
2. **Fixed-K KNN (Distance-Weighted Voting):** $K \\in [1, 3, 5, 7, 9, 11, 15]$ with weight $w_i = 1 / (d_i + \\epsilon)$
3. **Tuned Fixed-K KNN:** Optimal global $K$ selected using the validation partition.
4. **Logistic Regression:** $L_2$-regularized linear decision baseline.
5. **Support Vector Machines (SVM):**
   - Linear SVM (convex linear separator)
   - RBF Kernel SVM (non-linear Gaussian kernel mapping)

---

## 2. Evaluation Protocol

- **Data Partitions:** 60% Train, 20% Validation, 20% Test (stratified).
- **Hyperparameter Tuning:** Tuned on the 20% validation split, strictly evaluated on the unseen 20% test split.
- **Metrics Computed:**
  - Accuracy
  - Macro-F1 (unweighted mean across classes)
  - Balanced Accuracy (mean recall across classes)
  - Macro Precision & Macro Recall
  - Per-sample Inference Latency (milliseconds)
  - Adaptive $K$ distribution statistics (Mean and Std of $K(x)$)

---

## 3. Results Summary Table

Detailed test metrics are saved in:
`member_2_baseline_models/results/baseline_comparison_tables.csv`

Generated figures:
- `member_2_baseline_models/results/figures/baseline_macro_f1_comparison.png`
- `member_2_baseline_models/results/figures/fixed_k_vs_adaptive_curve.png`
""")
    print("Baseline documentation written.")


if __name__ == "__main__":
    run_baseline_evaluation()
