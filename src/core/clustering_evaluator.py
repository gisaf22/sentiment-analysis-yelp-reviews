import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


class ClusteringEvaluator:
    """
    Evaluates clustering quality using:
      - Intra-cluster cosine cohesion
      - Inter-centroid cosine separation
      - Cluster size balance

    Supports:
      - KMeans
      - Agglomerative Clustering
    """

    def __init__(self, random_state=42, n_init=20):
        self.random_state = random_state
        self.n_init = n_init

    # ------------------------------------------------------------------
    # CORE PUBLIC METHOD
    # ------------------------------------------------------------------

    def probe_k_range(
        self,
        X_raw,
        X_cluster,
        k_values,
        method="kmeans",
        linkage="average",
        metric="cosine",
    ):
        results = []

        for k in k_values:
            print(f"Evaluating {method} | k={k}...")

            labels = self._fit_clustering(
                X_cluster,
                k,
                method=method,
                linkage=linkage,
                metric=metric,
            )

            intra = self._compute_intra_cluster_cosine(X_raw, labels, k)
            inter = self._compute_inter_cluster_cosine(X_raw, labels, k)
            size_std = self._compute_cluster_size_std(labels, k)

            results.append({
                "k": k,
                "method": method,
                "linkage": linkage if method == "agglo" else None,
                "avg_intra_cosine": intra,
                "inter_centroid_cosine": inter,
                "cluster_size_std": size_std,
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # CLUSTERING BACKENDS
    # ------------------------------------------------------------------

    def _fit_clustering(self, X_cluster, k, method, linkage, metric):
        if method == "kmeans":
            return self._fit_kmeans(X_cluster, k)

        elif method == "agglo":
            return self._fit_agglo(X_cluster, k, linkage, metric)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _fit_kmeans(self, X_cluster, k):
        model = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        return model.fit_predict(X_cluster)

    def _fit_agglo(self, X_cluster, k, linkage, metric, cut=None):
        """
        Agglomerative clustering.
        NOTE:
          - Ward only works with Euclidean
          - Average / complete / single work with cosine
        """
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric=metric,
            cut=None
        )
        return model.fit_predict(X_cluster)

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    def _compute_intra_cluster_cosine(self, X_raw, labels, k):
        intra_scores = []
        cluster_sizes = []

        for cid in range(k):
            cluster_embs = X_raw[labels == cid]
            n = len(cluster_embs)

            if n > 1:
                sims = cosine_similarity(cluster_embs)
                upper = sims[np.triu_indices_from(sims, k=1)]
                intra_scores.append(np.mean(upper))
                cluster_sizes.append(n)

        if not intra_scores:
            return np.nan

        return np.average(intra_scores, weights=cluster_sizes)

    def _compute_inter_cluster_cosine(self, X_raw, labels, k):
        centroids = []

        for cid in range(k):
            cluster_embs = X_raw[labels == cid]
            if len(cluster_embs):
                centroids.append(cluster_embs.mean(axis=0))

        if len(centroids) < 2:
            return np.nan

        centroids = np.vstack(centroids)
        sim_mat = cosine_similarity(centroids)
        upper = sim_mat[np.triu_indices_from(sim_mat, k=1)]

        return upper.mean()

    def _compute_cluster_size_std(self, labels, k):
        sizes = [np.sum(labels == cid) for cid in range(k)]
        return np.std(sizes)

    # ------------------------------------------------------------------
    # FINAL CLUSTERING
    # ------------------------------------------------------------------

    def fit_final_clustering(
        self,
        X_cluster,
        k,
        method="kmeans",
        linkage="average",
        metric="cosine",
    ):
        if method == "kmeans":
            model = KMeans(
                n_clusters=k,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            labels = model.fit_predict(X_cluster)
            return model, labels

        elif method == "agglo":
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=linkage,
                metric=metric,
                cut=None
            )
            labels = model.fit_predict(X_cluster)
            return model, labels

    @staticmethod
    def extract_representative_examples(
        X_cluster,
        df,
        labels,
        kmeans_model,
        text_col="text_cleaned",
        top_n=10,
    ):
        """
        Select representative examples per cluster based on
        cosine similarity to the KMeans centroid.
    
        Valid ONLY for KMeans.
        """
    
        cluster_examples = {}
        centroids = kmeans_model.cluster_centers_
    
        for cid in np.unique(labels):
            idx = np.where(labels == cid)[0]
    
            if len(idx) == 0:
                continue
    
            cluster_vecs = X_cluster[idx]
            centroid = centroids[cid].reshape(1, -1)
    
            sims = cosine_similarity(cluster_vecs, centroid).ravel()
            top_idx = np.argsort(sims)[-top_n:]
    
            cluster_examples[int(cid)] = (
                df.iloc[idx[top_idx]][text_col].tolist()
            )
    
        return cluster_examples


    @staticmethod
    def extract_representative_examples_medoid(
        X_cluster,
        df,
        labels,
        text_col="text_cleaned",
        top_n=10,
    ):
        """
        Select representative examples per cluster using medoids
        (minimum mean cosine distance).
    
        Works for:
          - Agglomerative
          - DBSCAN / HDBSCAN
          - Any non-centroid clustering
        """
    
        cluster_examples = {}
    
        for cid in np.unique(labels):
            if cid == -1:
                continue  # skip noise if present
    
            idx = np.where(labels == cid)[0]
    
            if len(idx) == 0:
                continue
    
            cluster_vecs = X_cluster[idx]
    
            # Pairwise cosine distance within cluster
            dist_matrix = cosine_distances(cluster_vecs)
    
            # Mean distance per point (medoid score)
            mean_dist = dist_matrix.mean(axis=1)
    
            # Closest points to medoid
            top_idx = np.argsort(mean_dist)[:top_n]
    
            cluster_examples[int(cid)] = (
                df.iloc[idx[top_idx]][text_col].tolist()
            )
    
        return cluster_examples


