"""
Member 5 Master Runner: Adaptive-K Mapping Strategies & Algorithmic Sensitivity
==============================================================================

Evaluates:
1. Linear, Quantile (Rank-Based), Sigmoidal Logistic, and Step Threshold mappings.
2. Sensitivity grid search over m (density reference window), k_min, and k_max.
3. Cross-dataset generalization and parameter robustness.
"""

import sys
import os
import time

def main():
    print("=" * 75)
    print("MEMBER 5: ADAPTIVE-K MAPPING STRATEGIES & SENSITIVITY GRID SEARCH")
    print("=" * 75)
    
    t0 = time.time()
    
    from member_5_mapping_strategies.run_mapping_sensitivity import main as run_map_main
    run_map_main()
    
    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"✓ Member 5 complete in {elapsed:.1f}s")
    print("=" * 75)

if __name__ == "__main__":
    main()
