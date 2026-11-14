# Member 5 — Synthetic Data & Error Analysis

## 1. Controlled Synthetic Manifold Experiments

Synthetic manifolds isolate density disparities from extraneous confounding variables:
1. **Two-Density Gaussians:** Dense cluster ($\sigma=0.35$) vs. sparse cluster ($\sigma=1.40$).
2. **Noisy Moons:** Non-linear decision boundary with dense cores and overlapping boundary noise.
3. **Multi-Density Blobs:** Four clusters with disparate spreads $\sigma \in [0.5, 1.6, 0.8, 2.2]$.

### Observations:
- **Spatial K Maps (`synthetic_decision_boundaries_and_k_maps.png`):**
  The spatial heatmap shows that `DensityAdaptiveKNN` naturally assigns $K \approx 1-3$ inside dense clusters and smoothly expands to $K \approx 12-15$ in the sparse periphery.
- Fixed KNN with a large $K$ bleeds the sparse class into the dense cluster, whereas fixed $K=1$ produces jagged islands around noise. Adaptive KNN yields smooth boundaries while preserving fine geometric separation.

---

## 2. Prediction Disagreement & Error Decomposition

We decomposed test predictions between `DensityAdaptiveKNN` and the best fixed-K baseline into four mutually exclusive categories:
- **Both Correct:** Concordant consensus.
- **Adaptive Wins:** Points where Adaptive KNN assigned the correct label while Fixed KNN failed.
- **Fixed Wins:** Points where Fixed KNN succeeded while Adaptive KNN failed.
- **Both Wrong:** Severe noise or overlapping ambiguity.

### Key Conclusions:
- On **UCI Glass Identification**, Adaptive KNN achieved a net positive advantage of **+-2 sample(s)**, driven by better minority representation in sparse classes.
- Disagreement occurs predominantly near class boundaries with sharp density transitions, confirming that dynamic neighborhood adjustment changes decision margins in the intended regions.

---

## 3. Generated Deliverables
- Tables:
  - `member_5_synthetic_error_analysis/results/tables/synthetic_benchmark_results.csv`
  - `member_5_synthetic_error_analysis/results/tables/error_disagreement_analysis.csv`
- Figures:
  - `member_5_synthetic_error_analysis/results/figures/synthetic_decision_boundaries_and_k_maps.png`
  - `member_5_synthetic_error_analysis/results/figures/disagreement_breakdown_real_datasets.png`
