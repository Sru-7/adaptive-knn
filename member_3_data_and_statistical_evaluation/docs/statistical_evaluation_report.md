# Member 3 — Statistical Evaluation & Reproducibility Report

## 1. Experimental Integrity & Protocol

To eliminate random splitting artifacts and establish rigorous statistical validity:
- **Repetitions:** All experiments were evaluated across **5 distinct random seeds** (`[42, 123, 456, 789, 1024]`).
- **Data Partitions:** 60% Train, 20% Validation, 20% Test (stratified by class labels).
- **Leakage Prevention:** Standard scaling and density statistics were strictly precomputed on training partitions.
- **Statistical Tests:**
  1. Paired Student's t-test (`ttest_rel`)
  2. Non-parametric Wilcoxon signed-rank test (`wilcoxon`)
  3. Effect size: Cohen's $d = \frac{\mu_{\Delta}}{\sigma_{\Delta}}$

---

## 2. Integrated Multi-Seed Results (Mean ± Std)

Detailed metrics saved to:
`member_3_data_and_statistical_evaluation/results/tables/integrated_statistical_summary.csv`

---

## 3. Statistical Significance Findings

Full test statistics saved to:
`member_3_data_and_statistical_evaluation/results/tables/statistical_significance_tests.csv`

### Key Takeaways:
1. **UCI Glass Identification:**
   `Adaptive KNN (Distance-Weighted)` achieves a higher mean Macro-F1 compared to Fixed KNN baselines with positive effect sizes, showing strong advantages on datasets with diverse density profiles.
2. **Computational Overhead:**
   Predict latency for `DensityAdaptiveKNN` is under **0.15 ms per query sample**, making it fully practical for real-time inference while delivering local adaptivity.

---

## 4. Deliverables
- Tables:
  - `member_3_data_and_statistical_evaluation/results/tables/repeated_seed_raw_metrics.csv`
  - `member_3_data_and_statistical_evaluation/results/tables/integrated_statistical_summary.csv`
  - `member_3_data_and_statistical_evaluation/results/tables/statistical_significance_tests.csv`
- Figures:
  - `member_3_data_and_statistical_evaluation/results/figures/metric_variance_boxplots.png`
  - `member_3_data_and_statistical_evaluation/results/figures/runtime_latency_comparison.png`
