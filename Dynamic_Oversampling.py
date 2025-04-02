import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from imblearn.base import BaseSampler

class CustomOversampler(BaseSampler):
    def __init__(self, minority_class=1, n_neighbors=5, noise_scale=0.1, contamination=0.05, random_state=None):
        self.minority_class = minority_class
        self.n_neighbors = n_neighbors
        self.noise_scale = noise_scale
        self.contamination = contamination
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        Resamples the dataset by oversampling the minority class with density-based synthetic sample generation.
        """
        rng = np.random.default_rng(self.random_state)

        # Convert to NumPy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Separate majority and minority classes
        minority_data = X[y == self.minority_class]
        majority_data = X[y != self.minority_class]

        # Step 1: Density estimation using Nearest Neighbors
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(minority_data)
        distances, _ = nbrs.kneighbors(minority_data)
        density_scores = np.mean(distances, axis=1)

        # Step 2: Feature importance scaling
        feature_importances = mutual_info_classif(X, y, random_state=self.random_state)
        feature_weights = feature_importances / feature_importances.sum()

        # Step 3: Outlier detection using Isolation Forest
        isolation_forest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        isolation_forest.fit(minority_data)
        outlier_scores = -isolation_forest.decision_function(minority_data)
        sampling_weights = 1 / (1 + outlier_scores)
        sampling_weights /= sampling_weights.sum()

        # Step 4: Synthetic sample generation
        synthetic_data = []
        target_count = len(majority_data) - len(minority_data)

        for _ in range(target_count):
            combined_weights = sampling_weights / (density_scores + 1e-5)
            combined_weights /= combined_weights.sum()
            point_idx = rng.choice(len(minority_data), p=combined_weights)
            point = minority_data[point_idx]

            # Select a neighbor based on local density
            local_k = max(2, int(self.n_neighbors * (1 - density_scores[point_idx])))
            distances, indices = nbrs.kneighbors([point], n_neighbors=local_k)
            neighbor_idx = rng.choice(indices[0])
            neighbor = minority_data[neighbor_idx]

            # Generate a synthetic sample
            interpolation = rng.uniform(0, 1)
            synthetic_point = point + interpolation * (neighbor - point)

            # Add weighted noise
            noise = rng.normal(scale=self.noise_scale, size=synthetic_point.shape) * feature_weights
            synthetic_point += noise

            synthetic_data.append(synthetic_point)

        # Combine original data with synthetic data
        synthetic_data = np.array(synthetic_data)
        oversampled_X = np.vstack((X, synthetic_data))
        oversampled_y = np.hstack((y, np.full(len(synthetic_data), self.minority_class)))

        return oversampled_X, oversampled_y
