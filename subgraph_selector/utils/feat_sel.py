import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

class FeatureSelector:
    """
    Identifies the most important features using PCA or Tree-based methods.
    """

    def __init__(self, method="pca", top_n=6, standardize=True, top_n_features_per_pc=2):   
        """
        Initializes the feature selector.

        :param method: "pca" for PCA-based feature selection, "tree" for tree-based feature selection.
        :param top_n_pcs: Number of top principal components to consider (if using PCA).
        :param top_n_features_per_pc: Number of most significant features per PCA component.
        :param top_n_features_tree: Number of top important features to select for tree-based method.
        :param standardize: Whether to standardize the feature matrix before applying PCA.
        """
        self.method = method
        self.top_n = top_n
        self.standardize = standardize
        self.pca = None
        self.tree_model = None
        self.svd = None
        self.scaler = StandardScaler() if standardize else None 
        self.top_features = None  # Store selected features
        self.top_n_features_per_pc = top_n_features_per_pc
        self.top_n_pcs = top_n // top_n_features_per_pc

    
    def fit(self, feature_matrix, labels=None):
        """
        Fits the feature selection model based on the chosen method.
        """

        if self.method == "pca":
            self._fit_pca(feature_matrix)
        elif self.method == "tree":
            if labels is None:
                raise ValueError("Tree-based method requires labels (y).")
            self._fit_tree(feature_matrix, labels)
        elif self.method == "mutual_info":
            if labels is None:
                raise ValueError("Mutual Information method requires labels (y).")
            self._fit_mutual_info(feature_matrix, labels)
        elif self.method == "svd":
            self._fit_svd(feature_matrix)
        else:
            raise ValueError("Unsupported method. Choose 'pca', 'tree', 'mutual_info', or 'svd'.")



    def _fit_pca(self, feature_matrix):
        """Applies PCA for feature selection."""
        if self.standardize:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
            print("Feature matrix scaled and standardized.")

        # Apply PCA
        self.pca = PCA(n_components=self.top_n_pcs, svd_solver="randomized", random_state=42)
        self.pca.fit(feature_matrix)

        # Get loadings (absolute values to ignore direction)
        loadings = np.abs(self.pca.components_)

        # Identify the top N features for each PCA component
        feature_sets = []
        for i in range(self.top_n_pcs):
            top_indices = np.argsort(-loadings[i])[:self.top_n_features_per_pc]
            feature_sets.append(set(top_indices.tolist()))

        # Compute union of top features across PCs
        self.top_features = list(set.union(*feature_sets)) if feature_sets else []
        print(f"PCA Selected Features: {self.top_features}")
    
    # MCA: row 數可能比原始 One-Hot 特徵數還多，因為它基於「模態」(可能值) 來計算，而不是基於變數。

    # 檢查SVD
    def _fit_svd(self, feature_matrix):
        """Applies Truncated SVD for feature selection."""
        self.svd = TruncatedSVD(n_components=self.top_n_pcs, random_state=42)
        self.svd.fit(feature_matrix)

        # Get loadings (absolute values)
        loadings = np.abs(self.svd.components_)

        # Identify the top N features for each SVD component
        feature_sets = []
        for i in range(self.top_n_pcs):
            top_indices = np.argsort(-loadings[i])[:self.top_n_features_per_pc]
            feature_sets.append(set(top_indices.tolist()))

        # Compute union of top features across components
        self.top_features = list(set.union(*feature_sets)) if feature_sets else []
        print(f"SVD Selected Features: {self.top_features}")



    # 以防禦方的角度，可以看到訓練集的y
    def _fit_tree(self, feature_matrix, labels):
        """Applies a tree-based model (Random Forest) for feature selection."""

        self.tree_model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        self.tree_model.fit(feature_matrix, labels)

        # Get feature importance scores
        feature_importance = self.tree_model.feature_importances_

        # Select top N important features
        self.top_features = np.argsort(-feature_importance)[:self.top_n].tolist()
        print(f"Tree-Based Selected Features: {self.top_features}")


    # 以防禦方的角度，可以看到訓練集的y
    def _fit_mutual_info(self, feature_matrix, labels):
        """Applies Mutual Information for feature selection."""
        mi_scores = mutual_info_classif(feature_matrix, labels, discrete_features='auto')
        self.top_features = np.argsort(-mi_scores)[:self.top_n].tolist()
        print(f"Mutual Information Selected Features: {self.top_features}")


    def get_top_features(self):
        """
        Returns the most important features based on the selected method.
        """
        if self.top_features is None or len(self.top_features) == 0:  # 檢查 `None` 和空列表
            raise ValueError("Feature selection has not been fitted or no important features were found.")
        return self.top_features
