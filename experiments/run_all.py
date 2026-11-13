"""
Master runner — execute all robustness, imbalance, synthetic, and error analysis experiments.

Usage (from project root):
    python -m experiments.run_all

Or run individual experiments:
    python -m experiments.synthetic_experiments
    python -m experiments.error_analysis
    python -m experiments.robustness_noise
    python -m experiments.robustness_imbalance
"""

import time


def main():
    t0 = time.time()

    print("=" * 70)
    print("  ROBUSTNESS, IMBALANCE, SYNTHETIC DATA & ERROR ANALYSIS SUITE")
    print("=" * 70)

    # ---- 1. Synthetic experiments + K-value heatmaps ----
    print("\n\n▸ [1/4]  Synthetic-data experiments ...")
    from experiments.synthetic_experiments import run_synthetic_experiments
    run_synthetic_experiments()

    # ---- 2. Error / disagreement analysis ----
    print("\n\n▸ [2/4]  Prediction-disagreement & error analysis ...")
    from experiments.error_analysis import run_error_analysis
    run_error_analysis()

    # ---- 3. Noise robustness ----
    print("\n\n▸ [3/4]  Feature-noise robustness experiments ...")
    from experiments.robustness_noise import run_noise_experiments
    run_noise_experiments()

    # ---- 4. Class-imbalance ----
    print("\n\n▸ [4/4]  Class-imbalance experiments ...")
    from experiments.robustness_imbalance import run_imbalance_experiments
    run_imbalance_experiments()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  ALL DONE  ({elapsed:.1f}s total)")
    print("=" * 70)
    print("\nOutputs:")
    print("  results/tables/   — CSV files with all metrics")
    print("  results/figures/  — PNG plots")


if __name__ == "__main__":
    main()

