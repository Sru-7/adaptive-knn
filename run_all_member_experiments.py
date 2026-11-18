"""
Master Orchestration Runner: All Members 1 through 5
===================================================

Executes the complete experimental lifecycle across all 5 members:
  - Member 1: Adaptive algorithm unit tests & bound validation
  - Member 2: Baseline model evaluation (Fixed KNN, Tuned KNN, LogReg, SVM)
  - Member 3: Dataset preprocessing, exploratory data analysis & statistical significance evaluation (combining former 3 + 6)
  - Member 4: Robustness, class imbalance, synthetic manifolds & prediction disagreement analysis (combining former 4 + 5)
  - Member 5: Algorithmic mapping strategies & hyperparameter sensitivity grid (former 7)
"""

import sys
import os
import time
import subprocess


def run_step(step_num, title, module_or_cmd, is_pytest=False):
    print("\n" + "=" * 75)
    print(f"  STEP [{step_num}/5]: {title.upper()}")
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
    print("  ADAPTIVE KNN RESEARCH PIPELINE — 5-MEMBER MASTER ORCHESTRATION")
    print("  Datasets: Breast Cancer Wisconsin, UCI Glass, UCI Heart Disease")
    print("*" * 75)

    # 1. Member 1: Unit tests & Algorithm bounds
    run_step(1, "Member 1 — Adaptive KNN Unit Tests & Bound Validation", "member_1_adaptive_algorithm/test_adaptive_knn.py", is_pytest=True)

    # 2. Member 2: Baseline models
    run_step(2, "Member 2 — Baseline Models & Global K Evaluation", "member_2_baseline_models.run_baselines")

    # 3. Member 3: Dataset preprocessing, EDA & Statistical Hypothesis Testing (3 + 6)
    run_step(3, "Member 3 — Dataset Preprocessing, EDA & Statistical Significance", "member_3_data_and_statistical_evaluation.run_member_3")

    # 4. Member 4: Robustness, Imbalance, Synthetic Manifolds & Error Analysis (4 + 5)
    run_step(4, "Member 4 — Robustness, Imbalance, Synthetic Data & Error Analysis", "member_4_robustness_and_synthetic_analysis.run_member_4")

    # 5. Member 5: Mapping strategies & sensitivity (former 7)
    run_step(5, "Member 5 — Algorithmic Mapping Strategies & Sensitivity Grid", "member_5_mapping_strategies.run_member_5")

    total_elapsed = time.time() - start_time
    print("\n" + "*" * 75)
    print(f"  ALL 5 MEMBERS COMPLETED SUCCESSFULLY!  ({total_elapsed:.1f}s total)")
    print("*" * 75)
    print("\nMember Deliverable Directories:")
    print("  - member_1_adaptive_algorithm/              (Algorithm, docs, unit tests)")
    print("  - member_2_baseline_models/                 (Fixed-K, Tuned KNN, LogReg, SVM)")
    print("  - member_3_data_and_statistical_evaluation/ (Loaders, EDA, Paired tests, latency)")
    print("  - member_4_robustness_and_synthetic_analysis/ (Noise curves, imbalance, synthetic maps, error analysis)")
    print("  - member_5_mapping_strategies/              (Linear/Quantile/Sigmoid/Threshold mappings & sensitivity)")
    print("\nFinal Master Comprehensive Report:")
    print("  - COMPREHENSIVE_EVALUATION_REPORT.md")


if __name__ == "__main__":
    main()
