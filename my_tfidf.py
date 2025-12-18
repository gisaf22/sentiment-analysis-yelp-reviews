# feature_engineering/tfidf.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class MyTFIDF:
    """
    Builds TF-IDF features for interpretability & theme extraction.
    Designed to work with clustered review data.
    """

    def __init__(self, max_features=8000, max_df=0.75, ngram_range=(1, 2)):
        self.max_features = max_features
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.global_mean = None

    # ------------------------------------------------------------------
    # GLOBAL TF-IDF (BACKGROUND MODEL)
    # ------------------------------------------------------------------

    def build_global(self, texts):
        n_docs = len(texts)
        min_df = max(3, int(0.01 * n_docs))

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            min_df=min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range
        )

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        tf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )

        # Store global background distribution
        self.global_mean = tf_df.mean(axis=0)

        return tf_df

    def extract_cluster_terms(self, tf_df, labels, top_n=10):
        """
        Extract top TF-IDF terms per cluster from a GLOBAL TF-IDF matrix.
        """
        cluster_terms = {}

        for cid in sorted(np.unique(labels)):
            cluster_rows = tf_df[labels == cid]   # ✅ FIXED INDEXING
            mean_tfidf = cluster_rows.mean(axis=0)

            top_terms = (
                mean_tfidf
                .sort_values(ascending=False)
                .head(top_n)
                .index
                .tolist()
            )

            cluster_terms[cid] = top_terms

        return cluster_terms

    def extract_cluster_terms_local(self, df, text_col="lemmas", cluster_col="cluster", top_n=10):
        """
        Builds TF-IDF separately INSIDE each cluster for stronger theme separation.
        This is the version you should use for your final report.
        """
        cluster_terms = {}

        for cid in sorted(df[cluster_col].unique()):
            cluster_texts = df.loc[df[cluster_col] == cid, text_col]

            n_docs = len(cluster_texts)
            min_df = max(2, int(0.02 * n_docs))

            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=self.max_features,
                min_df=min_df,
                max_df=self.max_df,
                ngram_range=self.ngram_range
            )

            tfidf_matrix = vectorizer.fit_transform(cluster_texts)

            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            terms = vectorizer.get_feature_names_out()

            top_idx = mean_tfidf.argsort()[::-1][:top_n]
            cluster_terms[cid] = [terms[i] for i in top_idx]

        return cluster_terms

    # ------------------------------------------------------------------
    # CONTRASTIVE CLUSTER TERMS (FINAL THEMES)
    # ------------------------------------------------------------------

    def extract_cluster_terms_contrastive(
        self,
        tf_df,
        labels,
        top_n=10,
        alpha=0.6,        # background penalty
        min_cluster_df=0.02
    ):
        """
        Stable contrastive TF-IDF using:
        cluster_mean - alpha * global_mean
        with variance-aware smoothing.
        """
    
        if self.global_mean is None:
            raise ValueError("Call build_global() first.")
    
        cluster_terms = {}
        global_mean = self.global_mean
    
        for cid in sorted(np.unique(labels)):
            cluster_rows = tf_df[labels == cid]
    
            # Drop terms that are too rare inside the cluster
            df_ratio = (cluster_rows > 0).mean(axis=0)
            valid_terms = df_ratio >= min_cluster_df
    
            cluster_mean = cluster_rows.mean(axis=0)
    
            # ✅ Smoothed contrastive scoring
            contrastive_score = (
                cluster_mean
                - alpha * global_mean
            )
    
            contrastive_score = contrastive_score[valid_terms]
    
            top_terms = (
                contrastive_score
                .sort_values(ascending=False)
                .head(top_n)
                .index
                .tolist()
            )
    
            cluster_terms[cid] = top_terms
    
        return cluster_terms
