# Adaptive KNN: Learning the Right Number of Neighbors

Inspired by: Li, J., Xu, H., Zhang, S. (2026), *"Adaptive kNN graph model"*, *Nature Communications* 17, 7491.

---

## 🚀 Live Interactive Streamlit Demo (For Project Presentation)

An interactive web application is included to demonstrate the advantages of Density-Adaptive KNN over conventional Fixed-K KNN in real time.

```bash
# Launch the interactive demonstration website:
streamlit run app.py
```

### Features included in the Streamlit App:
1. **🌟 2D Manifolds & Spatial $K(x)$ Heatmaps:** Visually inspect dense clusters (where $K \approx 2-3$ preserves crisp boundaries) versus sparse clusters (where $K \approx 10-12$ stabilizes noise).
2. **📊 Interactive Benchmark Dashboard:** Compare Adaptive KNN against Fixed-K ($K \in [1, 3, 5, 7, 9, 11, 15]$) across Real UCI benchmarks and complex synthetic manifolds with **satisfying +8% to +18% gains**.
3. **🎯 Live Point Inspector & Disagreement Explorer:** Click or slide any coordinate $(X_1, X_2)$ to visualize its $K$ neighbors, distance weights, and see exactly where Fixed KNN fails and Adaptive KNN succeeds.
4. **🛡️ Real-Time Noise & Imbalance Stress-Testing:** Real-time sliders for Gaussian feature noise ($\sigma \in [0, 1]$) and class imbalance ($1:1$ to $1:10$), watching minority class recall stay protected.
5. **📑 Member-Wise Academic Architecture:** Full documentation of Members 1 through 5 methodologies, mathematical formulas, and statistical significance tests ($p < 0.05$).

---

## 📁 Member-Wise Project Architecture

This project is organized **strictly member-wise** to support clear ownership, modularity, and reproducible benchmarking across **Breast Cancer Wisconsin**, **UCI Glass Identification**, and **UCI Heart Disease (Cleveland)**:

```
adaptive-knn/
├── app.py                                # Streamlit Interactive Web Application
├── streamlit_app/app.py                  # Streamlit module entrypoint
├── member_1_adaptive_algorithm/          # Member 1: Core DensityAdaptiveKNN, bounds, unit tests
│   ├── adaptive_knn.py
│   ├── test_adaptive_knn.py
│   └── complete_contribution.md
├── member_2_baseline_models/             # Member 2: Baseline suite (Fixed KNN, LogReg, SVM)
│   ├── baselines.py
│   ├── run_baselines.py
│   ├── results/baseline_comparison_tables.csv
│   └── docs/baselines_documentation.md
├── member_3_data_and_statistical_evaluation/   # Member 3: Data loaders, preprocessing, EDA
│   ├── data_loader.py
│   ├── run_eda.py
│   ├── results/ (figures & tables)
│   └── docs/eda_report.md
├── member_4_robustness_and_synthetic_analysis/        # Member 4: Noise robustness & class imbalance
│   ├── run_robustness.py
│   ├── results/ (figures & tables)
│   └── docs/robustness_analysis.md
├── member_4_robustness_and_synthetic_analysis/   # Member 5: Synthetic manifolds & error disagreement
│   ├── synthetic_data.py
│   ├── run_synthetic_and_error.py
│   ├── results/ (figures & tables)
│   └── docs/error_analysis_report.md
├── member_3_data_and_statistical_evaluation/     # Member 6: Paired tests, latency, reproducibility
│   ├── statistical_tests.py
│   ├── run_statistical_evaluation.py
│   ├── results/ (figures & tables)
│   └── docs/statistical_evaluation_report.md
├── member_5_mapping_strategies/          # Member 7: Linear, Quantile, Sigmoidal mappings
│   ├── mappings.py
│   ├── run_mapping_sensitivity.py
│   ├── results/ (figures & tables)
│   └── docs/mapping_strategies_report.md
├── data/                                 # Datasets (glass.data, processed.cleveland.data)
├── run_all_member_experiments.py         # Master pipeline orchestrator
└── COMPREHENSIVE_EVALUATION_REPORT.md    # Master publication-grade evaluation report
```

---

## 🧪 Running All Member Experiments

To execute the entire member pipeline sequentially:

```bash
python run_all_member_experiments.py
```

### Running Individual Member Suites

```bash
# Member 1: Run unit tests
python -m pytest member_1_adaptive_algorithm/test_adaptive_knn.py

# Member 3: Preprocessing and Exploratory Data Analysis
python -m member_3_data_and_statistical_evaluation.run_eda

# Member 2: Baseline Models and Evaluation Framework
python -m member_2_baseline_models.run_baselines

# Member 4: Noise Robustness and Class Imbalance Sweeps
python -m member_4_robustness_and_synthetic_analysis.run_robustness

# Member 5: Synthetic Benchmarks and Prediction Disagreement
python -m member_4_robustness_and_synthetic_analysis.run_synthetic_and_error

# Member 7: Algorithmic Mapping Strategies and Sensitivity
python -m member_5_mapping_strategies.run_mapping_sensitivity

# Member 6: Statistical Significance and Final Integration
python -m member_3_data_and_statistical_evaluation.run_statistical_evaluation
```

---

## 📈 Key Improvement Summary

* **UCI Glass Identification:** Macro-F1 improves from **$0.4710$ (Fixed $K=5$) to $0.6007$ (Adaptive KNN with Sigmoidal Mapping)** — a **$+12.97\%$ absolute gain / $+27.5\%$ relative gain**; in 5-fold CV, Macro-F1 reaches **$0.6344$ vs $0.5054$ (+25.5% relative gain)**!
* **Multi-Density Manifold:** Macro-F1 jumps from **$78.4\%$ to $89.6\%$** (**$+11.2\%$ absolute gain**).
* **UCI Heart Disease:** Accuracy reaches **$85.25\%$** (beating validation-tuned KNN $81.6\%$ and RBF SVM $80.3\%$).
* **Noise Robustness:** Under $\sigma = 1.0$ feature noise, Fixed $K=1$ crashes to $84\%$, while Adaptive KNN maintains $93\%$ (**$+9.0\%$ gain**).
* **Minority Recall:** Preserves up to **$+25\%$ higher recall** under $1:10$ imbalance.

Detailed findings and 15 evaluation graphs are documented in: **[COMPREHENSIVE_EVALUATION_REPORT.md](COMPREHENSIVE_EVALUATION_REPORT.md)**
