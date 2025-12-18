import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import spacy
from collections import Counter


class BusinessContextClusterer:
    """
    Groups reviews of a single business into shared customer contexts
    (e.g., general dining, delivery issues, special events).
    """
    def __init__(self):
        pass

    def get_dendrogram(self, embeddings, method="average", metric="cosine"):
        """
        embeddings: normalized embedding matrix (n_samples × dim)
        method: linkage method ('average' recommended for cosine)
        metric: distance metric ('cosine' recommended)
        """
    
        # SciPy linkage needs a condensed distance matrix or raw vectors.
        # For cosine distance + average linkage, we can pass raw embeddings.
        Z = linkage(embeddings, method=method, metric=metric)
    
        return Z

    def plot_dendrogram(self, Z, truncate=100):
        """
        Z: linkage matrix, contains merge history
        truncate: how many leaf nodes to show (prevents huge trees)
        """
        dendrogram(Z, truncate_mode="lastp", p=truncate, leaf_rotation=90., leaf_font_size=10., show_contracted=True)
        plt.title(f"Hierarchical Clustering Dendrogram")
        plt.xlabel("Merged clusters(truncated)")
        plt.ylabel("Cosine Distance")

    def plot_merge_distance_jumps(self, Z, top_n=50):
        """
        Diagnostic plot showing merge-distance jumps
        near the top of the dendrogram.
        """
        heights = Z[:, 2]
        diffs = np.diff(heights)
    
        top_diffs = diffs[-top_n:]
        x = range(len(diffs) - top_n, len(diffs))
    
        plt.figure(figsize=(10, 4))
        plt.plot(x, top_diffs)
        plt.title("Top-Level Merge Distance Jumps")
        plt.xlabel("Merge index")
        plt.ylabel("Cosine distance change")
        plt.grid(True)
        plt.show()

    def detect_cut_height(self, Z, top_n=200):
        """
        Detect the most meaningful semantic cut near the top of the dendrogram.
    
        Returns:
            cut_height: cosine distance threshold for AgglomerativeClustering
        """
        heights = Z[:, 2]
        diffs = np.diff(heights)
    
        # focus on top-level merges only
        top_diffs = diffs[-top_n:]
    
        # largest semantic jump
        idx = np.argmax(top_diffs)
    
        # map local index to global
        global_idx = len(diffs) - top_n + idx
    
        # cut BEFORE the big jump
        cut_height = heights[max(global_idx - 1, 0)]
    
        return cut_height

    def cut_clusters(self, embeddings, cut_height, metric="cosine", linkage="average"):
        """
        Apply a semantic distance cut to produce business context labels.

        embeddings: normalized embedding matrix (n_samples × dim)
        cut_height: cosine distance threshold
        """
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cut_height,
            linkage=linkage,
            metric=metric,
        )
        return model.fit_predict(embeddings)

    def plot_dendrogram_with_spikes(self, Z, cut_height, truncate=100):
        plt.figure(figsize=(14, 6), dpi=150)
        self.plot_dendrogram(Z, truncate=truncate)
    
        plt.axhline(y=cut_height, color="red", linestyle="--", linewidth=2)
    
        plt.title("Dendrogram with Spike-Based Semantic Cuts")
        plt.xlabel("Merged clusters (truncated)")
        plt.ylabel("Cosine Distance")
        plt.show()