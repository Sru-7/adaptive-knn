import numpy as np

def inverse_density_mapping(local_distance, min_dist, max_dist, k_min, k_max):
    """
    Maps local distance to K. 
    Dense regions (small distance) -> K approaches k_min
    Sparse regions (large distance) -> K approaches k_max
    """
    # Prevent division by zero if all distances are identical
    if max_dist == min_dist:
        return k_min
    
    # Normalize the local distance to a [0, 1] scale
    ratio = (local_distance - min_dist) / (max_dist - min_dist)
    
    # Scale this ratio to fit within the [k_min, k_max] range
    k_adaptive = k_min + ratio * (k_max - k_min)
    
    # Round to an integer and enforce strict bounds in case of test set outliers
    k_final = int(np.round(k_adaptive))
    return max(k_min, min(k_final, k_max))