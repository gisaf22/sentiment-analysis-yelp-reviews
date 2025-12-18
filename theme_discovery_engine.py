# theme_discovery_engine.py

import numpy as np
from collections import Counter
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


class ThemeDiscoveryEngine:
    """
    Deterministic discovery of customer themes within a business context.

    Pipeline:
        texts
          → TF-IDF
          → NMF topics
          → representative documents
          → theme-specific noun phrases
    """

    def __init__(self, n_topics=3, n_top_words=10, max_features=3000, min_df=2, max_df=0.95, random_state=42):
        self.n_topics = n_topics
        self.n_top_words = n_top_words
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state

        # Required for noun_chunks
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # ==============================================================
    # PUBLIC API
    # ==============================================================

    def discover(self, texts_for_modeling, texts_for_display):
        """
        Discover themes inside a single semantic context.
    
        Parameters:
            texts_for_modeling : List[str]
                Lemmatized texts used for TF-IDF + NMF
            texts_for_display : List[str]
                Clean or raw texts used for noun phrases and examples
    
        Returns:
            {
                "themes": [
                    {
                        "theme_id": int,
                        "terms": List[str],
                        "noun_phrases": List[str],
                        "examples": List[str]
                    }
                ]
            }
        """
    
        # --------------------------------------------------
        # Guard: not enough data
        # --------------------------------------------------
        if len(texts_for_modeling) < 10:
            return {"themes": []}
    
        # --------------------------------------------------
        # 1. TF-IDF (LEXICAL, LEMMATIZED)
        # --------------------------------------------------
        tfidf, feature_names = self._build_tfidf(texts_for_modeling)
    
        # --------------------------------------------------
        # 2. NMF (TOPIC DISCOVERY)
        # --------------------------------------------------
        W, H = self._fit_nmf(tfidf)
        if W is None or H is None:
            return {"themes": []}
    
        themes = []
    
        # --------------------------------------------------
        # 3. Extract themes
        # --------------------------------------------------
        for topic_id in range(H.shape[0]):
    
            # Top discriminative words for this topic
            terms = self._top_terms(H, feature_names, topic_id)
    
            # Indices of documents most associated with this topic
            top_doc_idx = np.argsort(W[:, topic_id])[-5:]
    
            # Human-readable texts (NOT lemmatized)
            theme_texts = [texts_for_display[i] for i in top_doc_idx]
    
            # Linguistic noun phrases from surface text
            noun_phrases = self.extract_noun_phrases(
                theme_texts,
                limit=10
            )
    
            themes.append({
                "theme_id": topic_id,
                "terms": terms,
                "noun_phrases": noun_phrases,
                "examples": theme_texts[:3],
            })
    
        return {"themes": themes}


    # ==============================================================
    # TF-IDF (Lexical Signal Extraction)
    #
    # - Build a corpus-level vocabulary from all documents
    # - Keep up to `max_features` most discriminative terms (by TF-IDF)
    # - Term must appear in at least `min_df` documents
    # - Term must appear in no more than `max_df` fraction of documents
    # - Tokens must be alphabetic and length >= 3 characters
    # - Remove English stop words
    #
    # Purpose:
    #   Retain words that are frequent enough to matter,
    #   rare enough to be informative,
    #   and suitable for interpretable theme discovery.
    #
    # Words like
    # “the”, “food”, “place”, “good”
    # → appear everywhere → not discriminative
    #
    # Words like
    # “chargrilled”, “gumbo”, “wait line”, “raw oysters”
    # → appear frequently in some reviews but not others
    # → highly discriminative
    #   
    # These words help answer:
    #  
    # “What is this review specifically about?”
    # ==============================================================

    def _build_tfidf(self, texts):
        min_df = min(self.min_df, max(1, int(0.01 * len(texts))))

        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        )

        tfidf = vectorizer.fit_transform(texts)
        return tfidf, vectorizer.get_feature_names_out()

    # ==============================================================
    # NMF
    # ==============================================================

    def _fit_nmf(self, tfidf):
        n_topics = min(self.n_topics, tfidf.shape[0] - 1)
        if n_topics <= 0:
            return None, None

        nmf = NMF(
            n_components=n_topics,
            init="nndsvd",
            random_state=self.random_state,
            max_iter=500,
        )

        W = nmf.fit_transform(tfidf)
        H = nmf.components_

        return W, H

    # ==============================================================
    # THEME COMPONENTS
    # ==============================================================

    def _top_terms(self, H, feature_names, topic_id):
        """
        Return top n_terms that best describe a topic/component in the H matrix
        """
        weights = H[topic_id]
        top_idx = weights.argsort()[-self.n_top_words:][::-1]
        return [feature_names[i] for i in top_idx]

    def _top_documents(self, W, texts, topic_id, n_docs=5):
        """
        Return the top n_docs representative for a topic
        """
        top_idx = np.argsort(W[:, topic_id])[-n_docs:]
        return [texts[i] for i in top_idx]

    # ==============================================================
    # NOUN PHRASES
    # ==============================================================

    def extract_noun_phrases(self, texts, limit=20, max_doc_frac=0.5):
        """
        Extract informative noun phrases using linguistic + DF filtering.
        """
        phrase_counts = Counter()
        doc_freq = Counter()

        for text in texts:
            doc = self.nlp(text)

            # Ensures each phrase is counted once per document
            # Prevents long rants from dominating phrase importance
            seen = set()

            # For each review's noun chunks
            for chunk in doc.noun_chunks:

                # ignore chunks that are just one word, belong to tfidf
                if len(chunk) < 2:
                    continue

                # ignore chunk if all words are stop words
                if all(tok.is_stop for tok in chunk):
                    continue

                # enforce chunk to start with NOUN or PROPNOUN    
                if chunk.root.pos_ not in {"NOUN", "PROPN"}:
                    continue

                phrase = chunk.text.lower().strip()
                seen.add(phrase)

            phrase_counts.update(seen)
            doc_freq.update(seen)

        n_docs = len(texts)

        filtered = {
            p: c
            for p, c in phrase_counts.items()
            if doc_freq[p] / n_docs <= max_doc_frac
        }

        return [p for p, _ in Counter(filtered).most_common(limit)]
