# theme_discovery_engine.py
from __future__ import annotations

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


# ==============================
# Configuration
# ==============================

@dataclass
class ThemeDiscoveryConfig:
    """
    Configuration for lexical theme description.

    This engine assumes semantic clusters already exist.
    It provides contrastive lexical explanations using GLOBAL TF-IDF.
    """
    n_top_words: int = 10

    # TF-IDF controls
    max_features: int = 3000
    min_df: int = 15
    max_df: float = 0.80
    sublinear_tf: bool = True

    # Automatic generic-term suppression
    auto_stopwords: bool = True
    auto_stopwords_df_frac: float = 0.25
    auto_stopwords_max_terms: int = 60
    auto_stopwords_min_len: int = 3

    preserve_negation: bool = True
    token_pattern: str = r"(?u)\b[a-zA-Z_]{2,}\b"

    # Surface text extraction
    spacy_model: str = "en_core_web_sm"
    n_repr_docs: int = 5
    n_example_docs: int = 3

    min_docs: int = 10


# ==============================
# Engine
# ==============================

class ThemeDiscoveryEngine:
    """
    Lexical theme description engine.

    Lifecycle (IMPORTANT):
      1) fit_global_tfidf(all_texts)
      2) describe_cluster(cluster_texts)

    This separation enforces:
      - global IDF (contrastive signal)
      - local TF (cluster-specific salience)
    """

    def __init__(self, **kwargs):
        self.cfg = ThemeDiscoveryConfig(**kwargs)

        # stopwords
        self.negators = {"not", "no", "nor", "never"}
        self.stop_words = self._build_base_stop_words()
        self.last_auto_stopwords_: List[str] = []

        # TF-IDF state
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._feature_names: Optional[np.ndarray] = None

        # spaCy
        self._nlp = None

    # ==============================================================
    # GLOBAL FIT (IDF)
    # ==============================================================

    def fit_global_tfidf(self, texts_for_modeling: List[str]) -> None:
        """
        Fit GLOBAL TF-IDF.

        Must be called once before describing clusters.
        """
        if len(texts_for_modeling) < self.cfg.min_docs:
            raise ValueError("Not enough documents to fit global TF-IDF.")

        self._vectorizer, self._feature_names = self._fit_tfidf_vectorizer(texts_for_modeling)

    # ==============================================================
    # CLUSTER DESCRIPTION
    # ==============================================================

    def describe_cluster(
        self,
        texts_for_modeling: List[str],
        texts_for_display: List[str],
    ) -> Dict[str, Any]:
        """
        Produce lexical explanation for ONE semantic cluster.
        """
        self._validate_inputs(texts_for_modeling, texts_for_display)

        if self._vectorizer is None:
            raise RuntimeError("Global TF-IDF not fit. Call fit_global_tfidf first.")

        X = self._vectorizer.transform(texts_for_modeling)

        centroid = np.asarray(X.mean(axis=0)).ravel()
        terms = self._rank_terms_by_cluster_centroid(centroid)

        repr_idx = self._select_representative_docs(X)
        repr_texts = [texts_for_display[i] for i in repr_idx]

        noun_phrases = self._extract_noun_phrases(repr_texts)

        return {
            "terms": terms,
            "noun_phrases": noun_phrases,
            "examples": repr_texts[: self.cfg.n_example_docs],
        }

    # ==============================================================
    # TF-IDF internals
    # ==============================================================

    def _fit_tfidf_vectorizer(self, texts: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
        stop_words = self.stop_words
        self.last_auto_stopwords_ = []

        if self.cfg.auto_stopwords:
            learned = self._learn_auto_stopwords(texts, stop_words)
            if learned:
                stop_words = sorted(set(stop_words) | set(learned))
                self.last_auto_stopwords_ = learned

        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            min_df=max(1, min(self.cfg.min_df, len(texts))),
            max_df=self.cfg.max_df,
            max_features=self.cfg.max_features,
            token_pattern=self.cfg.token_pattern,
            lowercase=True,
            sublinear_tf=self.cfg.sublinear_tf,
        )
        vectorizer.fit(texts)
        return vectorizer, vectorizer.get_feature_names_out()

    def _rank_terms_by_cluster_centroid(self, centroid: np.ndarray) -> List[str]:
        if np.allclose(centroid, 0):
            return []

        idx = np.argsort(centroid)[-self.cfg.n_top_words:][::-1]
        return [self._feature_names[i] for i in idx if centroid[i] > 0]

    def _select_representative_docs(self, X) -> np.ndarray:
        strength = np.asarray(X.sum(axis=1)).ravel()
        n = min(self.cfg.n_repr_docs, len(strength))
        return np.argsort(strength)[-n:]

    # ==============================================================
    # Auto stopwords
    # ==============================================================

    def _learn_auto_stopwords(self, texts: List[str], base_stop_words: List[str]) -> List[str]:
        probe = TfidfVectorizer(
            stop_words=base_stop_words,
            min_df=1,
            max_df=1.0,
            max_features=self.cfg.max_features,
            token_pattern=self.cfg.token_pattern,
            lowercase=True,
        )
        X = probe.fit_transform(texts)
        feats = probe.get_feature_names_out()

        df = np.asarray((X > 0).sum(axis=0)).ravel()
        frac = df / max(1, X.shape[0])

        idx = np.where(frac >= self.cfg.auto_stopwords_df_frac)[0]
        terms = []
        for i in idx:
            t = feats[i]
            if len(t) < self.cfg.auto_stopwords_min_len:
                continue
            if self.cfg.preserve_negation and t in self.negators:
                continue
            terms.append((t, frac[i]))

        terms.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in terms[: self.cfg.auto_stopwords_max_terms]]

    # ==============================================================
    # Validation
    # ==============================================================

    def _validate_inputs(self, texts_for_modeling, texts_for_display):
        if len(texts_for_modeling) != len(texts_for_display):
            raise ValueError("Modeling and display texts must align.")
        if len(texts_for_modeling) < self.cfg.min_docs:
            raise ValueError("Too few documents in cluster.")

    def _build_base_stop_words(self) -> List[str]:
        base = set(ENGLISH_STOP_WORDS)
        if self.cfg.preserve_negation:
            base -= self.negators
        return sorted(base)

    # ==============================================================
    # Noun phrase extraction
    # ==============================================================

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load(self.cfg.spacy_model, disable=["ner"])
        return self._nlp

    def _extract_noun_phrases(self, texts: List[str]) -> List[str]:
        counts = Counter()
        doc_freq = Counter()
    
        for text in texts:
            seen = set()
            doc = self.nlp(text)
            for chunk in doc.noun_chunks:
                if len(chunk) < 2:
                    continue
                if all(tok.is_stop for tok in chunk):
                    continue
                phrase = chunk.text.lower()
                counts[phrase] += 1
                seen.add(phrase)
            doc_freq.update(seen)
    
        max_df = 0.8 * len(texts)
        return [
            p for p, _ in counts.most_common(10)
            if doc_freq[p] <= max_df
        ]

