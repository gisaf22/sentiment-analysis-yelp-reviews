# pipeline.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data.my_reviews_loader import BusinessReviewLoader
from src.preprocessing.my_preprocessor import MyPreProcessor
from src.preprocessing.feature_engineering import MyFeatureEngineering
from src.embeddings.my_sentence_transformer import MySentenceTransformer
from src.core.business_context_clusterer import BusinessContextClusterer
from src.core.theme_discovery_engine import ThemeDiscoveryEngine
from src.core.theme_labeler import ThemeLabeler
from src.core.clustering_evaluator import ClusteringEvaluator
from src.embeddings.embedding_space_normalizer import EmbeddingSpaceNormalizer


class ReviewThemePipeline:
    """
    End-to-end pipeline for discovering and labeling customer themes
    for a single business.

    Pipeline stages:
    1. Load reviews
    2. Clean text
    3. Embed reviews
    4. Cluster into business contexts (hierarchical)
    5. Focus on main context (largest cluster)
    6. KMeans within main context to find themes
    7. Discover + label each theme using LLM
    """

    def __init__(
        self,
        review_loader: BusinessReviewLoader,
        preprocessor: MyPreProcessor,
        embedder: MySentenceTransformer,
        context_clusterer: BusinessContextClusterer,
        theme_discovery: ThemeDiscoveryEngine,
        theme_labeler: ThemeLabeler,
    ):
        self.review_loader = review_loader
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.context_clusterer = context_clusterer
        self.theme_discovery = theme_discovery
        self.theme_labeler = theme_labeler
        
        # Additional components for KMeans sub-clustering
        self.evaluator = ClusteringEvaluator()
        self.normalizer = EmbeddingSpaceNormalizer()
        self.feature_engineer = MyFeatureEngineering()

    # ------------------------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------------------------

    def run(self, business_id, k_range=range(6, 15), business_name=None):
        """
        Run full theme discovery + labeling pipeline for a business.
        
        Follows the notebook workflow:
        1. Load and preprocess reviews
        2. Embed reviews
        3. Hierarchical clustering to find business contexts
        4. Focus on main (largest) context
        5. KMeans within main context to find sub-themes
        6. Label each theme with LLM

        Returns:
            dict with structure:
            {
                theme_id: {
                    "theme_name": ...,
                    "theme_category": ...,
                    "summary": ...,
                    "likes": [...],
                    "dislikes": [...],
                    "business_impact": ...,
                    "keywords": [...],           # TF-IDF terms
                    "noun_phrases": [...],       # Extracted phrases
                    "representative_reviews": [...],  # Example reviews
                    "review_count": int          # Number of reviews in theme
                }
            }
        """

        # -----------------------------
        # 1. Load reviews
        # -----------------------------
        df = self.review_loader.get_reviews_by_business_id(business_id)

        if df.empty:
            return {}

        # -----------------------------
        # 2. Preprocess text
        # -----------------------------
        df = self.preprocessor.prepare(df)
        
        # Add lemmas for TF-IDF modeling
        df = self.feature_engineer.add_features(df)
        
        # -----------------------------
        # 3. Embed reviews
        # -----------------------------
        texts = df["text_cleaned"].tolist()
        embeddings = self.embedder.transform_sentences(texts)

        # -----------------------------
        # 4. Hierarchical clustering (business contexts)
        # -----------------------------
        Z = self.context_clusterer.get_dendrogram(embeddings)
        cut_height = self.context_clusterer.detect_cut_height(Z)
        context_labels = self.context_clusterer.cut_clusters(
            embeddings=embeddings,
            cut_height=cut_height,
        )

        # -----------------------------
        # 5. Focus on main context (largest cluster)
        # -----------------------------
        cluster_sizes = pd.Series(context_labels).value_counts()
        primary_context_id = cluster_sizes.idxmax()
        
        main_mask = context_labels == primary_context_id
        main_reviews = df[main_mask].reset_index(drop=True)
        X_main = embeddings[main_mask]
        
        print(f"Main context has {len(main_reviews)} reviews (out of {len(df)} total)")
        
        if len(main_reviews) < 20:
            print("Too few reviews in main context for theme discovery")
            return {}

        # -----------------------------
        # 6. KMeans within main context
        # -----------------------------
        # Normalize embeddings for clustering
        X_main_cluster = self.normalizer.fit_transform(X_main)
        
        # Find optimal k using elbow method
        elbow_k = self.evaluator.find_elbow_k(
            X_raw=X_main,
            X_cluster=X_main_cluster,
            k_range=k_range,
        )
        print(f"Optimal k from elbow method: {elbow_k}")
        
        # Fit final KMeans
        kmeans_model, theme_labels = self.evaluator.fit_final_clustering(
            X_cluster=X_main_cluster,
            k=elbow_k,
            method="kmeans"
        )
        
        main_reviews = main_reviews.copy()
        main_reviews["theme_id"] = theme_labels
        
        theme_sizes = main_reviews.groupby("theme_id").size().sort_values(ascending=False)
        print(f"Theme sizes: {dict(theme_sizes)}")

        # -----------------------------
        # 7. Discover + label each theme
        # -----------------------------
        # Fit global TF-IDF on all main context reviews
        texts_modeling = main_reviews["lemmas"].tolist()
        texts_display = main_reviews["text"].tolist()
        self.theme_discovery.fit_global_tfidf(texts_modeling)
        
        output = {}
        
        for theme_id in theme_sizes.index:
            theme_mask = main_reviews["theme_id"] == theme_id
            theme_texts_modeling = main_reviews.loc[theme_mask, "lemmas"].tolist()
            theme_texts_display = main_reviews.loc[theme_mask, "text"].tolist()
            
            if len(theme_texts_modeling) < 5:
                print(f"Skipping theme {theme_id} (only {len(theme_texts_modeling)} reviews)")
                continue
            
            try:
                # Extract theme signals
                signals = self.theme_discovery.describe_cluster(
                    texts_for_modeling=theme_texts_modeling,
                    texts_for_display=theme_texts_display
                )
                
                # Label the theme with LLM
                labeled = self.theme_labeler.label_theme(
                    cluster_id=f"theme_{theme_id}",
                    terms=signals["terms"],
                    noun_phrases=signals["noun_phrases"],
                    examples=signals["examples"],
                )
                
                # Add intermediate data to output
                labeled["keywords"] = signals["terms"]
                labeled["noun_phrases"] = signals["noun_phrases"]
                labeled["representative_reviews"] = signals["examples"]
                labeled["review_count"] = len(theme_texts_modeling)
                
                output[theme_id] = labeled
                print(f"✓ Theme {theme_id}: {labeled.get('theme_name', 'unnamed')}")
                
            except Exception as e:
                print(f"Error processing theme {theme_id}: {e}")
                continue

        # -----------------------------
        # 8. Compute inter-theme similarity matrix
        # -----------------------------
        unique_themes = sorted(output.keys())
        centroids = []
        for tid in unique_themes:
            theme_mask = theme_labels == tid
            theme_embeddings = X_main[theme_mask]
            if len(theme_embeddings) > 0:
                centroids.append(theme_embeddings.mean(axis=0))
        
        if len(centroids) >= 2:
            centroids = np.vstack(centroids)
            similarity_matrix = cosine_similarity(centroids).tolist()
        else:
            similarity_matrix = None
        
        # -----------------------------
        # 9. Build review assignments for downstream analysis
        # -----------------------------
        # Include date if available, review_id if available
        review_assignments = []
        for idx, row in main_reviews.iterrows():
            assignment = {
                "theme_id": int(row["theme_id"]),
            }
            # Include date if present
            if "date" in row and pd.notna(row["date"]):
                assignment["date"] = str(row["date"])
            # Include review_id if present
            if "review_id" in row and pd.notna(row["review_id"]):
                assignment["review_id"] = str(row["review_id"])
            review_assignments.append(assignment)
        
        # Build final output with metadata
        final_output = {
            "metadata": {
                "business_id": business_id,
                "business_name": business_name,
                "total_reviews": len(df),
                "main_context_reviews": len(main_reviews),
                "k": elbow_k,
                "theme_ids": [int(t) for t in unique_themes]
            },
            "inter_theme_similarity": similarity_matrix,
            "themes": {int(k): v for k, v in output.items()},
            "review_assignments": review_assignments
        }

        return final_output
