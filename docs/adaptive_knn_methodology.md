# Adaptive-K Algorithm Methodology

## Overview
This document outlines the core algorithm for the density-based adaptive-K classifier.

## Algorithm Pseudocode
**Input:** 
- Training data `X_train`, labels `y_train`
- Query instance `x_q`
- Parameters: `m` (density neighbors), `k_min` (lower bound), `k_max` (upper bound)
- Mapping function `f(density)`

**Procedure:**
1. Calculate distances from `x_q` to all points in `X_train`.
2. Find the `m` nearest neighbors to `x_q`.
3. Compute local density estimate `d_local` = Average distance to the `m` nearest neighbors.
4. Calculate adaptive neighborhood size: `K_adaptive` = `f(d_local, k_min, k_max)`.
5. Constrain `K_adaptive` so that `k_min <= K_adaptive <= k_max`.
6. Extract the `K_adaptive` nearest neighbors to `x_q`.
7. **Return:** The majority class label among the `K_adaptive` neighbors.