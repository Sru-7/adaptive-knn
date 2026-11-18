"""
Backward-compatible wrapper re-exporting Member 7's mapping functions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from member_5_mapping_strategies.mappings import (
    linear_mapping,
    quantile_mapping,
    sigmoid_mapping,
    threshold_mapping,
)

# Alias for legacy compatibility
inverse_density_mapping = linear_mapping

__all__ = [
    "linear_mapping",
    "quantile_mapping",
    "sigmoid_mapping",
    "threshold_mapping",
    "inverse_density_mapping",
]
