import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from .mappings import inverse_density_mapping

class DensityAdaptiveKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, m=5, k_min=1, k_max=15, mapping_func=None):
        """
        m: Number of neighbors used to estimate local density.
        k_min: Lower bound for K.
        k_max: Upper bound for K.
        mapping_func: Strategy to convert density into K.
        """
        self.m = m
        self.k_min = k_min
        self.k_max = k_max
        self.mapping_func = mapping_func if mapping_func is not None else inverse_density_mapping

    def fit(self, X, y):
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        self.classes_ = np.unique(self.y_train_)
        
        # We need max(m, k_max) neighbors total for both density estimation and the final vote
        self.max_neighbors = max(self.m, self.k_max)
        self.nn_ = NearestNeighbors(n_neighbors=self.max_neighbors)
        self.nn_.fit(self.X_train_)
        
        # Precompute global min and max distances across the training set to help scale the mapping function
        distances, _ = self.nn_.kneighbors(self.X_train_, n_neighbors=self.m)
        mean_distances = np.mean(distances, axis=1)
        self.global_min_dist_ = np.min(mean_distances)
        self.global_max_dist_ = np.max(mean_distances)
        
        return self

    def _estimate_density(self, distances_to_m):
        """ Density is represented inversely by the mean distance to the m nearest neighbors. """
        return np.mean(distances_to_m)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        # Retrieve distances and indices up to max_neighbors for all query points
        distances, indices = self.nn_.kneighbors(X, n_neighbors=self.max_neighbors)
        
        for i in range(len(X)):
            # 1. Estimate local distance using the top m neighbors
            local_distance = self._estimate_density(distances[i, :self.m])
            
            # 2. Map this distance to an adaptive K value using the assigned mapping strategy
            adaptive_k = self.mapping_func(
                local_distance, 
                self.global_min_dist_, 
                self.global_max_dist_, 
                self.k_min, 
                self.k_max
            )
            
            # 3. Enforce the K boundaries
            adaptive_k = max(self.k_min, min(adaptive_k, self.k_max))
            
            # 4. Perform majority voting among the dynamically selected adaptive_k neighbors
            neighbor_classes = self.y_train_[indices[i, :adaptive_k]]
            most_common = Counter(neighbor_classes).most_common(1)[0][0]
            predictions.append(most_common)
            
        return np.array(predictions)