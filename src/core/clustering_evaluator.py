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

    def find_elbow_k(
        self,
        X_raw,
        X_cluster,
        k_range,
        method="kmeans",
        linkage="average",
        metric="cosine",
    ):
        """
        Find optimal k using elbow method on inter-centroid cosine distance.
        
        Uses the "maximum distance from line" approach to detect the elbow point
        where adding more clusters yields diminishing returns.
        
        Parameters:
        -----------
        X_raw : np.ndarray
            Raw embeddings for metric computation
        X_cluster : np.ndarray
            Normalized embeddings for clustering
        k_range : range or list
            Range of k values to evaluate
        method : str
            Clustering method ("kmeans" or "agglo")
            
        Returns:
        --------
        int : Optimal k value
        """
        results = self.probe_k_range(
            X_raw=X_raw,
            X_cluster=X_cluster,
            k_values=k_range,
            method=method,
            linkage=linkage,
            metric=metric,
        )
        
        k = np.array(results["k"])
        y = np.array(results["inter_centroid_cosine"])
        
        # Normalize for elbow detection
        k_norm = (k - k.min()) / (k.max() - k.min() + 1e-9)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
        
        # Find point with max distance from line connecting endpoints
        p1 = np.array([k_norm[0], y_norm[0]])
        p2 = np.array([k_norm[-1], y_norm[-1]])
        
        distances = []
        for i in range(len(k_norm)):
            p = np.array([k_norm[i], y_norm[i]])
            distance = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-9)
            distances.append(distance)
        
        elbow_k = k[np.argmax(distances)]
        return int(elbow_k)

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
    def extract_representative_examples_diverse(
        X_cluster,
        df,
        labels,
        kmeans_model,
        text_col="text_cleaned",
        top_n=10,
        n_closest=2,
        boundary_percentile=80,
    ):
        """
        Select representative examples using hybrid strategy:
        1. Include n_closest reviews nearest to centroid (most typical)
        2. Stratified sample from remaining reviews within safe boundary
        
        Parameters:
        -----------
        top_n : int
            Total number of examples to select
        n_closest : int  
            Number of closest-to-centroid examples to always include
        boundary_percentile : int
            Only sample from reviews within this percentile of distance
            (avoids outliers near cluster boundaries)
        
        Strategy:
        ---------
        - Reviews closest to centroid = "core" examples (typical)
        - Reviews from middle distance = "edge" examples (variety)
        - Excludes outer 20% to avoid boundary confusion
        """
        cluster_examples = {}
        centroids = kmeans_model.cluster_centers_

        for cid in np.unique(labels):
            idx = np.where(labels == cid)[0]

            if len(idx) == 0:
                continue

            cluster_vecs = X_cluster[idx]
            centroid = centroids[cid].reshape(1, -1)

            # Calculate similarities to centroid
            sims = cosine_similarity(cluster_vecs, centroid).ravel()
            sorted_order = np.argsort(sims)[::-1]  # highest similarity first

            # Determine safe boundary (exclude outer reviews)
            boundary_idx = int(len(sorted_order) * boundary_percentile / 100)
            safe_indices = sorted_order[:boundary_idx]

            if len(safe_indices) < top_n:
                # Small cluster: just take what we have
                selected = sorted_order[:top_n]
            else:
                # Select n_closest from top
                closest = safe_indices[:n_closest]
                
                # Stratified sample from remaining safe zone
                remaining_safe = safe_indices[n_closest:]
                n_diverse = min(top_n - n_closest, len(remaining_safe))
                
                if n_diverse > 0 and len(remaining_safe) > 0:
                    # Sample evenly across distance quantiles
                    step = len(remaining_safe) // n_diverse
                    if step == 0:
                        step = 1
                    diverse_picks = remaining_safe[::step][:n_diverse]
                    selected = np.concatenate([closest, diverse_picks])
                else:
                    selected = closest

            # Convert to original indices and get texts
            original_idx = idx[selected]
            cluster_examples[int(cid)] = (
                df.iloc[original_idx][text_col].tolist()
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


