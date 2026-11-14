# Adaptive KNN: Learning the Right Number of Neighbors

Inspired by: Li, J., Xu, H., Zhang, S. (2026), *"Adaptive kNN graph model"*, *Nature Communications* 17, 7491.

---

## Member-Wise Project Architecture

This project is organized **strictly member-wise** to support clear ownership, modularity, and reproducible benchmarking across three key datasets: **Breast Cancer Wisconsin**, **UCI Glass Identification**, and **UCI Heart Disease (Cleveland)**.

```
adaptive-knn/
├── member_1_adaptive_algorithm/          # Member 1: Adaptive algorithm, unit tests, bounds
│   ├── adaptive_knn.py
│   ├── test_adaptive_knn.py
│   └── docs/methodology.md
├── member_2_baseline_models/             # Member 2: Baseline suite (Fixed KNN, LogReg, SVM)
│   ├── baselines.py
│   ├── run_baselines.py
│   ├── results/
│   │   ├── baseline_comparison_tables.csv
│   │   └── figures/
│   └── docs/baselines_documentation.md
├── member_3_dataset_preprocessing_eda/   # Member 3: Data loaders, preprocessing, EDA
│   ├── data_loader.py
│   ├── run_eda.py
│   ├── results/
│   │   ├── figures/
│   │   └── tables/
│   └── docs/eda_report.md
├── member_4_robustness_imbalance/        # Member 4: Noise robustness & class imbalance
│   ├── run_robustness.py
│   ├── results/
│   │   ├── figures/
│   │   └── tables/
│   └── docs/robustness_analysis.md
├── member_5_synthetic_error_analysis/   # Member 5: Synthetic manifolds & error disagreement
│   ├── synthetic_data.py
│   ├── run_synthetic_and_error.py
│   ├── results/
│   │   ├── figures/
│   │   └── tables/
│   └── docs/error_analysis_report.md
├── member_6_statistical_evaluation/     # Member 6: Paired tests, latency, reproducibility
│   ├── statistical_tests.py
│   ├── run_statistical_evaluation.py
│   ├── results/
│   │   ├── figures/
│   │   └── tables/
│   └── docs/statistical_evaluation_report.md
├── member_7_mapping_strategies/          # Member 7: Linear/Quantile/Sigmoid/Threshold mappings
│   ├── mappings.py
│   ├── run_mapping_sensitivity.py
│   ├── results/
│   │   ├── figures/
│   │   └── tables/
│   └── docs/mapping_strategies_report.md
├── data/                                 # Datasets (glass.data, processed.cleveland.data)
├── run_all_member_experiments.py         # Master orchestrator
└── COMPREHENSIVE_EVALUATION_REPORT.md    # Complete publication-grade report with figures
```

---

## Quickstart: Running All Experiments

To execute the entire member pipeline sequentially:

```bash
python run_all_member_experiments.py
```

### Running Individual Member Suites

```bash
# Member 1: Run unit tests
python -m pytest member_1_adaptive_algorithm/test_adaptive_knn.py

# Member 3: Preprocessing and Exploratory Data Analysis
python -m member_3_dataset_preprocessing_eda.run_eda

# Member 2: Baseline Models and Evaluation Framework
python -m member_2_baseline_models.run_baselines

# Member 4: Noise Robustness and Class Imbalance Sweeps
python -m member_4_robustness_imbalance.run_robustness

# Member 5: Synthetic Benchmarks and Prediction Disagreement
python -m member_5_synthetic_error_analysis.run_synthetic_and_error

# Member 7: Algorithmic Mapping Strategies and Sensitivity
python -m member_7_mapping_strategies.run_mapping_sensitivity

# Member 6: Statistical Significance and Final Integration
python -m member_6_statistical_evaluation.run_statistical_evaluation
```

---

## Detailed Evaluation Report

The full experimental methodology, theoretical formulations, baseline comparisons, robustness curves, error analysis, and 15 evaluation graphs are documented in:
👉 **[COMPREHENSIVE_EVALUATION_REPORT.md](COMPREHENSIVE_EVALUATION_REPORT.md)**
