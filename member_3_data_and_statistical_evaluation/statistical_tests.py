"""
Member 6: Statistical Significance Testing & Reproducibility Suite
==================================================================

Provides rigorous statistical hypothesis testing across repeated runs:
- Paired Student's t-test (parametric)
- Wilcoxon signed-rank test (non-parametric)
- Effect size (Cohen's d)
- Latency and runtime profiling
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_paired_tests(scores_a, scores_b, model_a_name="Adaptive KNN", model_b_name="Baseline"):
    """
    Computes Paired t-test and Wilcoxon signed-rank test between two paired arrays of scores.
    """
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    diff = a - b
    n = len(diff)

    mean_a = float(np.mean(a))
    std_a = float(np.std(a, ddof=1)) if n > 1 else 0.0
    mean_b = float(np.mean(b))
    std_b = float(np.std(b, ddof=1)) if n > 1 else 0.0
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    # Cohen's d effect size
    cohen_d = float(mean_diff / (std_diff + 1e-8)) if std_diff > 1e-8 else 0.0

    # Paired t-test
    if np.allclose(diff, 0.0):
        t_stat, p_val_t = 0.0, 1.0
    else:
        t_stat, p_val_t = stats.ttest_rel(a, b)

    # Wilcoxon signed rank test
    if np.allclose(diff, 0.0) or np.all(diff == 0):
        w_stat, p_val_w = 0.0, 1.0
    else:
        try:
            w_res = stats.wilcoxon(a, b, zero_method="wilcox")
            w_stat, p_val_w = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, p_val_w = np.nan, np.nan

    return {
        "Model_A": model_a_name,
        "Model_B": model_b_name,
        "Mean_A": round(mean_a, 4),
        "Std_A": round(std_a, 4),
        "Mean_B": round(mean_b, 4),
        "Std_B": round(std_b, 4),
        "Mean_Diff": round(mean_diff, 4),
        "Cohen_d": round(cohen_d, 3),
        "t_stat": round(float(t_stat), 3),
        "p_val_paired_t": round(float(p_val_t), 4),
        "p_val_wilcoxon": round(float(p_val_w), 4) if not np.isnan(p_val_w) else "-",
        "Significant (p<0.05)": bool(p_val_t < 0.05),
    }
