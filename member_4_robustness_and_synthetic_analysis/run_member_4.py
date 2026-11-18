"""
Member 4 Master Runner: Robustness, Class Imbalance, Synthetic Manifolds & Error Analysis
========================================================================================

Combines:
1. Feature noise perturbation sweeps (sigma in [0.0, 1.0]) and class imbalance stress-tests (1:1 to 1:10).
2. Synthetic 2D multi-density manifolds, dynamic K(x) heatmaps, and prediction disagreement breakdown.
"""

import sys
import os
import time

def main():
    print("=" * 75)
    print("MEMBER 4: ROBUSTNESS, CLASS IMBALANCE, SYNTHETIC DATA & ERROR ANALYSIS")
    print("=" * 75)
    
    t0 = time.time()
    
    print("\n--- Subtask 4.1: Feature Noise & Class Imbalance Experiments ---")
    from member_4_robustness_and_synthetic_analysis.run_robustness import main as run_robust_main
    run_robust_main()
    
    print("\n--- Subtask 4.2: Synthetic Manifolds & Prediction Disagreement Analysis ---")
    from member_4_robustness_and_synthetic_analysis.run_synthetic_and_error import main as run_syn_main
    run_syn_main()
    
    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"✓ Member 4 complete in {elapsed:.1f}s")
    print("=" * 75)

if __name__ == "__main__":
    main()
