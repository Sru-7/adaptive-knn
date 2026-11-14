# Member 2 — Baseline Models & Evaluation Framework

## 1. Overview of Evaluated Models

We benchmarked the proposed `DensityAdaptiveKNN` against a full suite of competitive baseline classifiers:
1. **Fixed-K KNN (Uniform Voting):** $K \in [1, 3, 5, 7, 9, 11, 15]$
2. **Fixed-K KNN (Distance-Weighted Voting):** $K \in [1, 3, 5, 7, 9, 11, 15]$ with weight $w_i = 1 / (d_i + \epsilon)$
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
