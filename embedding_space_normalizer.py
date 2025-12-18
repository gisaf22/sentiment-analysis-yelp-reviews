# embedding_space.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class EmbeddingSpaceNormalizer:
    """
    Handles geometry correction for transformer embeddings:
    mean-centering, PCA, and optional whitening.
    """

    def __init__(self, n_components=150, whiten=True, random_state=42):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.pca = None
        self.mean_vec = None
        self.scaler = None

    def fit_transform(self, X):
        """
        X must be L2-normalized embeddings
        """
        # ---- Mean centering ----------------
        self.scaler = StandardScaler(with_std=False)
        X_centered = self.scaler.fit_transform(X)

        # ---- PCA (+ optional whitening) ----
        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver="full",
            random_state=self.random_state
        )

        X_proj = self.pca.fit_transform(X_centered)

        return X_proj

    def transform(self, X):
        """
        Apply same transformation to new data
        """
        # Mean centering
        X_centered = self.scaler.fit_transform(X)
        return self.pca.transform(X_centered)

    def variance_retained(self):
        return self.pca.explained_variance_ratio_.sum()
