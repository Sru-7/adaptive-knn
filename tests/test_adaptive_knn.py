import pytest
import numpy as np
from src.adaptive_knn import DensityAdaptiveKNN

def test_adaptive_knn_bounds_and_predictions():
    # Generate random synthetic data for testing
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(10, 5)
    
    # Initialize the classifier with specific bounds
    clf = DensityAdaptiveKNN(m=3, k_min=3, k_max=11)
    clf.fit(X_train, y_train)
    
    # Predict to ensure the fit and predict pipeline runs without crashing
    preds = clf.predict(X_test)
    
    # Check that predictions map to the correct classes
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})
    
    # Manually test that the mapping function respects the K bounds
    for i in range(len(X_test)):
        dists, _ = clf.nn_.kneighbors([X_test[i]], n_neighbors=clf.m)
        local_dist = clf._estimate_density(dists[0])
        
        k = clf.mapping_func(
            local_dist, 
            clf.global_min_dist_, 
            clf.global_max_dist_, 
            clf.k_min, 
            clf.k_max
        )
        
        # Ensure the selected K is within the predefined limits
        assert clf.k_min <= k <= clf.k_max