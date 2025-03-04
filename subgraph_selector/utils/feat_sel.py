import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureSelector:
    """
    Identifies the most important features across the top N principal components (PCs).
    """

    def __init__(self, top_n_pcs=3, top_n_features_per_pc=2, standardize=True):
        """
        Initializes the PCA feature selector.
        
        :param top_n_pcs: Number of top principal components to consider.
        :param top_n_features_per_pc: Number of most significant features to extract per PCA component.
        :param standardize: Whether to standardize the feature matrix before applying PCA.
        """
        self.top_n_pcs = top_n_pcs
        self.top_n_features_per_pc = top_n_features_per_pc
        self.standardize = standardize
        self.pca = None
        self.scaler = StandardScaler() if standardize else None
        self.top_features_per_pc = None  # Store selected features for each PCA
        self.common_features = None  # Store the intersection of top features across PCs

    def fit(self, feature_matrix):
        """
        Fits PCA on the given feature matrix and finds the most important features.
        """

        # Optional standardization
        if self.standardize:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
            print("Feature matrix scaled and standardized.")

        # Apply PCA
        self.pca = PCA(n_components=self.top_n_pcs, svd_solver="randomized", random_state=42)
        self.pca.fit(feature_matrix)

        # Get loadings (absolute values to ignore direction)
        loadings = np.abs(self.pca.components_)

        # Identify the top N features for each PCA component
        self.top_features_per_pc = {}
        feature_sets = []

        for i in range(self.top_n_pcs):
            top_indices = np.argsort(loadings[i])[-self.top_n_features_per_pc:][::-1]  # Get top N features
            self.top_features_per_pc[f"PCA {i+1}"] = top_indices.tolist()
            feature_sets.append(set(top_indices.tolist()))  # Store as set for intersection

        # Print top features per PCA component
        for pc, features in self.top_features_per_pc.items():
            print(f"{pc}: Top {self.top_n_features_per_pc} Features → {features}")

        # Compute union of top features across PCs
        self.common_features = list(set.union(*feature_sets)) if feature_sets else []

    def get_top_features(self):
        """
        Returns the union of the most important features across the selected PCs.
        """
        if self.common_features is None:
            raise ValueError("PCA has not been fitted. Please call `fit()` first.")
        return self.common_features


# Example
if __name__ == "__main__":
    feature_matrix = np.array([
        [0, 2, 0, 0, 1, 3],  
        [0, 2, 0, 1, 1, 2],
        [0, 2, 0, 2, 1, 3],
        [0, 2, 0, 3, 1, 2],
        [0, 2, 0, 3, 1, 3],
        [1, 0, 1, 2, 2, 1],
        [1, 0, 1, 3, 2, 3],
        [1, 0, 1, 0, 2, 1],
        [1, 0, 1, 0, 2, 3],
        [1, 1, 1, 1, 3, 1]
    ])  

    # Initialize and fit PCA feature selector
    pca_selector = FeatureSelector(top_n_pcs=3, top_n_features_per_pc=2, standardize=True)  # 取前 3 個 PCA，每個取 2 個重要特徵
    pca_selector.fit(feature_matrix)

    # Get the union of the most important features
    common_features = pca_selector.get_top_features()
    print("Union of Top Features Across PCs:", common_features)
