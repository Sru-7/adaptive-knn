"""
Feature-noise robustness experiments under varying perturbation levels.

For each dataset × noise level × seed:
  1. Inject Gaussian noise  N(0, σ · std_j)  into every feature j.
  2. Train and evaluate adaptive-K KNN and all fixed-K baselines.
  3. Record accuracy, macro-F1, balanced accuracy.
  4. Save line plots (metric vs noise level) and a results CSV.

Usage (from project root):
    python -m experiments.robustness_noise
"""

import sys, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adaptive_knn import DensityAdaptiveKNN
from experiments.utils import (
    evaluate_model, SEEDS, K_VALUES, ensure_dirs, FIGURES_DIR, TABLES_DIR,
)

NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]


# -----------------------------------------------------------------------
# Benchmark datasets for noise robustness evaluation
# -----------------------------------------------------------------------
def _load_builtin_datasets():
    """Return dict of { name: (X, y) } using scikit-learn toy datasets."""
    loaders = {
        "iris":           load_iris,
        "wine":           load_wine,
        "breast_cancer":  load_breast_cancer,
    }
    return {name: loader(return_X_y=True) for name, loader in loaders.items()}


# -----------------------------------------------------------------------
# Noise injection
# -----------------------------------------------------------------------
def add_feature_noise(X, noise_scale, rng):
    """
    Add Gaussian noise:  X_noisy = X + N(0, noise_scale * std_j)
    where std_j is the per-feature standard deviation of X.
    """
    if noise_scale == 0.0:
        return X.copy()
    stds = X.std(axis=0)
    noise = rng.normal(0, 1, X.shape) * stds * noise_scale
    return X + noise


# -----------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------
def run_noise_experiments():
    ensure_dirs()
    all_datasets = _load_builtin_datasets()
    all_rows = []

    t_start = time.time()

    for ds_name, (X_raw, y) in all_datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  (n={len(y)}, features={X_raw.shape[1]})")
        print(f"{'='*60}")

        for seed in SEEDS:
            # Reproducible split
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_raw, y, test_size=0.25, random_state=seed, stratify=y,
            )

            # Scale based on training data
            scaler = StandardScaler().fit(X_train_raw)
            X_train_scaled = scaler.transform(X_train_raw)
            X_test_scaled  = scaler.transform(X_test_raw)

            for noise in NOISE_LEVELS:
                rng = np.random.default_rng(seed + int(noise * 1000))
                X_train_noisy = add_feature_noise(X_train_scaled, noise, rng)
                X_test_noisy  = add_feature_noise(X_test_scaled, noise, rng)

                # ---- Adaptive KNN ----
                ada = DensityAdaptiveKNN(m=5, k_min=1, k_max=15)
                ada.fit(X_train_noisy, y_train)
                y_pred = ada.predict(X_test_noisy)
                metrics = evaluate_model(y_test, y_pred)
                metrics.update({
                    "model": "adaptive_knn",
                    "dataset": ds_name,
                    "noise_level": noise,
                    "seed": seed,
                })
                all_rows.append(metrics)

                # ---- Fixed-K baselines ----
                for k in K_VALUES:
                    fknn = KNeighborsClassifier(n_neighbors=k)
                    fknn.fit(X_train_noisy, y_train)
                    y_pred_f = fknn.predict(X_test_noisy)
                    mf = evaluate_model(y_test, y_pred_f)
                    mf.update({
                        "model": f"fixed_k={k}",
                        "dataset": ds_name,
                        "noise_level": noise,
                        "seed": seed,
                    })
                    all_rows.append(mf)

            print(f"  seed={seed} done")

    elapsed = time.time() - t_start
    print(f"\nAll experiments finished in {elapsed:.1f}s")

    # ---- Save CSV ----
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(TABLES_DIR, "noise_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")

    # ---- Plots ----
    _plot_noise_robustness(df)

    return df


# -----------------------------------------------------------------------
# Line plots: metric vs noise level
# -----------------------------------------------------------------------
def _plot_noise_robustness(df):
    """One figure per dataset with subplots for each metric."""
    metrics = ["accuracy", "macro_f1", "balanced_accuracy"]
    metric_labels = {"accuracy": "Accuracy", "macro_f1": "Macro F1",
                     "balanced_accuracy": "Balanced Accuracy"}

    for ds_name in df["dataset"].unique():
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
        sub = df[df["dataset"] == ds_name]

        # Aggregate: mean across seeds
        agg = sub.groupby(["model", "noise_level"])[metrics].mean().reset_index()

        for ax, metric in zip(axes, metrics):
            # Adaptive KNN — thick line
            ada_data = agg[agg["model"] == "adaptive_knn"].sort_values("noise_level")
            ax.plot(ada_data["noise_level"], ada_data[metric],
                    "o-", linewidth=2.5, markersize=7, color="crimson",
                    label="Adaptive KNN", zorder=5)

            # Fixed-K baselines — thinner lines
            cmap = plt.cm.Blues(np.linspace(0.35, 0.9, len(K_VALUES)))
            for idx, k in enumerate(K_VALUES):
                fk_data = agg[agg["model"] == f"fixed_k={k}"].sort_values("noise_level")
                ax.plot(fk_data["noise_level"], fk_data[metric],
                        "s--", linewidth=1.2, markersize=4, color=cmap[idx],
                        label=f"K={k}", alpha=0.85)

            ax.set_xlabel("Noise level (σ × feature std)")
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(f"{ds_name} — {metric_labels[metric]}")
            ax.set_xticks(NOISE_LEVELS)
            ax.grid(alpha=0.3)

        # Single legend for the figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
                   bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        path = os.path.join(FIGURES_DIR, f"noise_robustness_{ds_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Plot saved: {path}")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    run_noise_experiments()

