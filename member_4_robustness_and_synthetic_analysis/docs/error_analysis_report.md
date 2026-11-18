# Member 4 (Subtask 4.2) — Synthetic Manifolds & Error Disagreement Analysis

## 1. Controlled Synthetic Manifold Experiments

Synthetic manifolds isolate density disparities from extraneous confounding variables:
1. **Two-Density Gaussians:** Dense cluster ($\sigma=0.35$) vs sparse cluster ($\sigma=1.40$).
   - *Fixed KNN ($K=5$):* Accuracy 95.56%, Macro-F1 0.9555
   - *Adaptive KNN:* Accuracy **97.78%**, Macro-F1 **0.9777** (+2.2% gain)
2. **Noisy Moons:** Non-linear decision boundary with dense cores and overlapping boundary noise.
   - *Fixed KNN ($K=5$):* Accuracy 95.00%, Macro-F1 0.9499
   - *Adaptive KNN:* Accuracy **96.67%**, Macro-F1 **0.9665** (+1.7% gain)
3. **Multi-Density Blobs:** Four clusters with disparate spreads $\sigma \in [0.5, 1.6, 0.8, 2.2]$.
   - *Fixed KNN ($K=5$):* Accuracy 91.67%, Macro-F1 0.9166
   - *Adaptive KNN:* Accuracy **93.75%**, Macro-F1 **0.9374** (+2.1% gain)

### Spatial K Maps (`synthetic_decision_boundaries_and_k_maps.png`):
The spatial heatmaps confirm the central theoretical thesis of this research:
- Inside dense cluster cores, `DensityAdaptiveKNN` naturally assigns **$K \approx 1-3$**, protecting fine local structure without contamination.
- In the sparse periphery, the surface smoothly expands to **$K \approx 12-15$**, aggregating sufficient evidence to prevent noise overfitting.

---

## 2. Prediction Disagreement & Error Decomposition on Real Datasets

We decomposed test predictions between `DensityAdaptiveKNN` and the tuned fixed-K baseline into four mutually exclusive categories:
- **Both Correct:** Concordant consensus.
- **Adaptive Wins:** Samples correctly classified by Adaptive KNN but misclassified by Fixed KNN.
- **Fixed Wins:** Samples correctly classified by Fixed KNN but misclassified by Adaptive KNN.
- **Both Wrong:** High-entropy ambiguous or mislabeled samples.

### Key Conclusions:
- **UCI Glass Identification:** Adaptive KNN achieved a decisive net advantage of **+4 sample(s)** (5 wins vs 1 loss), yielding a **+9.3% accuracy gain**. Disagreement rate was **13.95%**, concentrated in sparse minority glass types (Classes 5 and 7).
- **UCI Heart Disease:** Net advantage of **+3 sample(s)** (4 wins vs 1 loss), yielding a **+4.9% accuracy gain**.
- **Breast Cancer:** Net advantage of **+2 sample(s)** (2 wins vs 0 losses), with **1.75%** disagreement rate.

---

## 3. Generated Deliverables
- Tables:
  - `member_4_robustness_and_synthetic_analysis/results/tables/synthetic_benchmark_results.csv`
  - `member_4_robustness_and_synthetic_analysis/results/tables/error_disagreement_analysis.csv`
- Figures:
  - `member_4_robustness_and_synthetic_analysis/results/figures/synthetic_decision_boundaries_and_k_maps.png`
  - `member_4_robustness_and_synthetic_analysis/results/figures/disagreement_breakdown_real_datasets.png`
