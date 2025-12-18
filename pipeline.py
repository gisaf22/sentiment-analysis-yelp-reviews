# pipeline.py

from collections import defaultdict

from data.my_reviews_loader import BusinessReviewLoader
from preprocessing.my_preprocessor import MyPreProcessor
from embeddings.my_sentence_transformer import MySentenceTransformer
from clustering.business_context_clusterer import BusinessContextClusterer
from themes.theme_discovery import ThemeDiscoveryEngine
from themes.theme_labeler import ThemeLabeler


class ReviewThemePipeline:
    """
    End-to-end pipeline for discovering and labeling customer themes
    for a single business.

    Pipeline stages:
    1. Load reviews
    2. Clean text
    3. Embed reviews
    4. Cluster into business contexts
    5. Discover themes within each context
    6. Label themes using an LLM
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

    # ------------------------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------------------------

    def run(self, business_id):
        """
        Run full theme discovery + labeling pipeline for a business.

        Returns:
            dict with structure:
            {
                context_id: {
                    "themes": {
                        theme_id: {
                            "theme_name": ...,
                            "summary": ...,
                            "likes": [...],
                            "dislikes": [...],
                            "business_impact": ...
                        }
                    }
                }
            }
        """

        # -----------------------------
        # 1. Load reviews
        # -----------------------------
        df = self.review_loader.load(business_id)

        if df.empty:
            return {}

        # -----------------------------
        # 2. Preprocess text
        # -----------------------------
        df = self.preprocessor.prepare(df)
        texts = df["text_cleaned"].tolist()

        # -----------------------------
        # 3. Embed reviews
        # -----------------------------
        embeddings = self.embedder.encode(texts)

        # -----------------------------
        # 4. Discover business contexts
        # -----------------------------
        Z = self.context_clusterer.get_dendrogram(embeddings)
        cut_height = self.context_clusterer.detect_cut_height(Z)

        labels = self.context_clusterer.cluster(
            embeddings=embeddings,
            cut_height=cut_height,
        )

        # group reviews by context
        context_texts = defaultdict(list)
        for label, text in zip(labels, texts):
            context_texts[label].append(text)

        # -----------------------------
        # 5. Discover + label themes
        # -----------------------------
        output = {}

        for context_id, context_reviews in context_texts.items():
            signals = self.theme_discovery.discover(context_reviews)

            labeled_themes = {}

            for theme in signals["themes"]:
                labeled = self.theme_labeler.label_theme(
                    cluster_id=f"context_{context_id}_theme_{theme['theme_id']}",
                    terms=theme["terms"],
                    noun_phrases=signals["noun_phrases"],
                    examples=theme["examples"],
                )

                labeled_themes[theme["theme_id"]] = labeled

            output[context_id] = {
                "themes": labeled_themes
            }

        return output
