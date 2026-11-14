"""
Master Orchestration Runner: All Members 1 through 7
===================================================

Executes the complete experimental lifecycle across all 7 members:
  - Member 1: Adaptive algorithm unit tests & integrity verification
  - Member 3: Dataset preprocessing & exploratory data analysis (Breast Cancer, Glass, Heart Disease)
  - Member 2: Baseline model evaluation (Fixed KNN, Tuned KNN, LogReg, SVM)
  - Member 4: Robustness to feature noise & class imbalance experiments
  - Member 5: Synthetic manifold benchmarks & prediction disagreement analysis
  - Member 7: Algorithmic mapping strategies & hyperparameter sensitivity analysis
  - Member 6: Statistical significance testing & integrated reproducibility report
"""

import sys
import os
import time
import subprocess


def run_step(step_num, title, module_or_cmd, is_pytest=False):
    print("\n" + "=" * 75)
    print(f"  STEP [{step_num}/7]: {title.upper()}")
    print("=" * 75)
    t0 = time.time()

    if is_pytest:
        cmd = [sys.executable, "-m", "pytest", module_or_cmd]
    else:
        cmd = [sys.executable, "-m", module_or_cmd]

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"✓ Step {step_num} completed successfully in {elapsed:.1f}s")
        # Print snippet of output
        lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
        for ln in lines[-6:]:
            print(f"  | {ln}")
    else:
        print(f"✗ Step {step_num} failed with exit code {result.returncode}")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError(f"Step {step_num} failed: {title}")


def main():
    start_time = time.time()
    print("*" * 75)
    print("  ADAPTIVE KNN RESEARCH PIPELINE — MASTER MEMBER ORCHESTRATION")
    print("  Datasets: Breast Cancer Wisconsin, UCI Glass, UCI Heart Disease")
    print("*" * 75)

    # 1. Member 1: Unit tests
    run_step(1, "Member 1 — Adaptive KNN Unit Tests & Bound Validation", "member_1_adaptive_algorithm/test_adaptive_knn.py", is_pytest=True)

    # 2. Member 3: Dataset preprocessing & EDA
    run_step(2, "Member 3 — Dataset Preprocessing & Exploratory Data Analysis", "member_3_dataset_preprocessing_eda.run_eda")

    # 3. Member 2: Baseline models
    run_step(3, "Member 2 — Baseline Models & Global K Evaluation", "member_2_baseline_models.run_baselines")

    # 4. Member 4: Robustness and class imbalance
    run_step(4, "Member 4 — Noise Robustness & Class Imbalance Experiments", "member_4_robustness_imbalance.run_robustness")

    # 5. Member 5: Synthetic data and error analysis
    run_step(5, "Member 5 — Synthetic Manifolds & Error Disagreement Analysis", "member_5_synthetic_error_analysis.run_synthetic_and_error")

    # 6. Member 7: Mapping strategies & sensitivity
    run_step(6, "Member 7 — Algorithmic Mapping Strategies & Sensitivity Grid", "member_7_mapping_strategies.run_mapping_sensitivity")

    # 7. Member 6: Statistical evaluation & reproducibility
    run_step(7, "Member 6 — Statistical Hypothesis Testing & Integration", "member_6_statistical_evaluation.run_statistical_evaluation")

    total_elapsed = time.time() - start_time
    print("\n" + "*" * 75)
    print(f"  ALL 7 MEMBERS COMPLETED SUCCESSFULLY!  ({total_elapsed:.1f}s total)")
    print("*" * 75)
    print("\nMember Deliverable Directories:")
    print("  - member_1_adaptive_algorithm/          (Algorithm, docs, unit tests)")
    print("  - member_2_baseline_models/             (Fixed-K, Tuned KNN, LogReg, SVM)")
    print("  - member_3_dataset_preprocessing_eda/   (Loaders, EDA plots, summaries)")
    print("  - member_4_robustness_imbalance/        (Noise curves, imbalance heatmaps)")
    print("  - member_5_synthetic_error_analysis/   (Synthetic K maps, error analysis)")
    print("  - member_6_statistical_evaluation/     (Paired tests, boxplots, latency)")
    print("  - member_7_mapping_strategies/          (Linear/Quantile/Sigmoid mappings)")
    print("\nFinal Master Comprehensive Report:")
    print("  - COMPREHENSIVE_EVALUATION_REPORT.md")


if __name__ == "__main__":
    main()
