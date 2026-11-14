"""
Backward-compatible wrapper re-exporting Member 1's DensityAdaptiveKNN.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from member_1_adaptive_algorithm.adaptive_knn import DensityAdaptiveKNN, linear_mapping, quantile_mapping, sigmoid_mapping

__all__ = ["DensityAdaptiveKNN", "linear_mapping", "quantile_mapping", "sigmoid_mapping"]
