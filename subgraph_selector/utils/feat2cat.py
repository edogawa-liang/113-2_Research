import numpy as np
from sklearn.mixture import GaussianMixture

class Feat2Vec:
    """
    A class for transforming continuous numerical features into categorical values.
    Uses Gaussian Mixture Model (GMM) to find the best cut-off points.
    """

    def __init__(self, min_components=2, max_components=5, penalty=2):
        """
        Initializes the Feat2Vec transformer.

        :param min_components: Minimum number of clusters for Gaussian Mixture Model (GMM).
        :param max_components: Maximum number of clusters for GMM to prevent overfitting.
        :param penalty: Multiplier for the complexity term in BIC: BIC = -2 log L + penalty * k log N.
        """
        self.min_components = min_components
        self.max_components = max_components
        self.penalty = penalty
        self.best_k = None  # Store the best K value


    def _gmm_binning(self, feature_values):
        """
        Uses Gaussian Mixture Model (GMM) to categorize continuous values.
        Automatically selects the optimal number of components (k) using BIC.
        """

        feature_values = np.asarray(feature_values).reshape(-1, 1)  # Reshape for GMM
        best_k = self.min_components
        best_bic = float("inf")

        # Find the best number of components
        for k in range(self.min_components, self.max_components + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(feature_values)
            bic = gmm.bic(feature_values)  # Compute BIC score
            modified_bic = bic + self.penalty * k * np.log(len(feature_values))  # Apply extra penalty

            if modified_bic < best_bic:
                best_bic = modified_bic
                best_k = k  # Update the best k

        self.best_k = best_k

        print(f"Using GMM binning, best k: {best_k}")
        gmm = GaussianMixture(n_components=best_k, random_state=42)
        labels = gmm.fit_predict(feature_values)

        return labels


    def transform(self, feature_values):
        """
        Transforms continuous feature values into categorical values using GMM clustering.
        """
        feature_values = np.asarray(feature_values)  # Ensure input is a NumPy array
        return self._gmm_binning(feature_values)
        
    
    def get_best_k(self):
        """
        Returns the best number of clusters found by GMM.
        """
        return self.best_k
    

if __name__ == "__main__":
    # Example feature vectors
    feature_vectors = {
        "Feature Vector 1": np.array([1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]),  # Expect k ≈ 2
        "Feature Vector 2": np.array([1, 2, 3, 4, 5, 10, 11, 11, 11, 11, 12, 13, 19, 20, 21, 22, 23]),  # Expect k ≈ 3
        "Feature Vector 3": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # Expect k ≈ 2
    }

    feat2vec = Feat2Vec()

    for name, feature_vector in feature_vectors.items():
        categorical_values = feat2vec.transform(feature_vector)
        best_k = feat2vec.get_best_k()

        print(f"\n{name}:")
        print("Original Values:", feature_vector)
        print("Transformed Categorical Values:", categorical_values)
        print("Best k:", best_k)
