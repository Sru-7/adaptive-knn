"""
Member 3 Master Runner: Dataset Preparation, EDA & Statistical Significance
===========================================================================

Combines:
1. Dataset ingestion, missing value imputation, scaling, PCA, and density heterogeneity EDA.
2. Multi-seed repeated evaluations (Seeds: 42, 123, 456, 789, 1024), Paired t-tests,
   Wilcoxon signed-rank tests, Cohen's d effect sizes, and runtime latency profiling.
"""

import sys
import os
import time

def main():
    print("=" * 75)
    print("MEMBER 3: DATASET PREPROCESSING, EDA & STATISTICAL EVALUATION")
    print("=" * 75)
    
    t0 = time.time()
    
    print("\n--- Subtask 3.1: Exploratory Data Analysis & Density Histograms ---")
    from member_3_data_and_statistical_evaluation.run_eda import run_eda
    run_eda()
    
    print("\n--- Subtask 3.2: Multi-Seed Statistical Significance & Profiling ---")
    from member_3_data_and_statistical_evaluation.run_statistical_evaluation import main as run_stats_main
    run_stats_main()
    
    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"✓ Member 3 complete in {elapsed:.1f}s")
    print("=" * 75)

if __name__ == "__main__":
    main()
