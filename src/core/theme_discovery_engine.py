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
    n_repr_docs: int = 15
    n_example_docs: int = 10

    # Diverse sampling strategy
    sampling_strategy: str = "diverse"  # "closest" or "diverse"
    n_closest: int = 2  # Number of closest-to-centroid examples (anchors)
    boundary_percentile: int = 80  # Exclude outer 20% (near cluster boundaries)

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
        """
        Select representative documents using configured strategy.
        
        Strategies:
        - "closest": Select docs with highest TF-IDF similarity to centroid
        - "diverse": Hybrid approach - some closest + stratified sample from safe zone
        """
        n_docs = X.shape[0]
        n = min(self.cfg.n_repr_docs, n_docs)
        
        if n_docs == 0:
            return np.array([], dtype=int)
        
        # Calculate centroid similarity for each doc
        centroid = np.asarray(X.mean(axis=0)).ravel()
        
        # Cosine similarity to centroid
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        norms = np.linalg.norm(X_dense, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            centroid_norm = 1
        
        sims = (X_dense @ centroid) / (norms * centroid_norm)
        sorted_order = np.argsort(sims)[::-1]  # Highest similarity first
        
        if self.cfg.sampling_strategy == "closest":
            # Original behavior: just pick closest to centroid
            return sorted_order[:n]
        
        elif self.cfg.sampling_strategy == "diverse":
            # Hybrid: closest + stratified from safe zone
            boundary_idx = int(n_docs * self.cfg.boundary_percentile / 100)
            safe_indices = sorted_order[:max(boundary_idx, n)]
            
            if len(safe_indices) <= n:
                return safe_indices
            
            # Take n_closest from top
            n_closest = min(self.cfg.n_closest, n)
            closest = safe_indices[:n_closest]
            
            # Stratified sample from remaining safe zone
            remaining_safe = safe_indices[n_closest:]
            n_diverse = n - n_closest
            
            if n_diverse > 0 and len(remaining_safe) > 0:
                step = max(1, len(remaining_safe) // n_diverse)
                diverse_picks = remaining_safe[::step][:n_diverse]
                return np.concatenate([closest, diverse_picks])
            else:
                return closest
        
        else:
            # Fallback to TF-IDF strength (original behavior)
            strength = np.asarray(X.sum(axis=1)).ravel()
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
        """
        Extract meaningful noun phrases from representative texts.
        
        Pipeline:
        1. Parse texts with spaCy
        2. Extract noun chunks
        3. Clean each chunk (remove determiners, pronouns, etc.)
        4. Rank by frequency, filter by document frequency
        
        Returns:
            List of top 10 cleaned noun phrases
        """
        counts = Counter()
        doc_freq = Counter()
    
        for text in texts:
            phrases_in_doc = self._extract_phrases_from_text(text)
            counts.update(phrases_in_doc)
            doc_freq.update(set(phrases_in_doc))
    
        return self._rank_and_filter_phrases(counts, doc_freq, n_texts=len(texts))
    
    def _extract_phrases_from_text(self, text: str) -> List[str]:
        """Extract cleaned noun phrases from a single text."""
        phrases = []
        doc = self.nlp(text)
        
        for chunk in doc.noun_chunks:
            cleaned = self._clean_noun_chunk(chunk)
            if cleaned:
                phrases.append(cleaned)
        
        return phrases
    
    def _clean_noun_chunk(self, chunk) -> Optional[str]:
        """
        Clean a spaCy noun chunk by removing uninformative tokens.
        
        Removes:
        - Determiners (a, an, the)
        - Pronouns (my, your, his, her, its, our, their)
        - Demonstratives (this, that, these, those)
        
        Returns:
            Cleaned phrase string, or None if invalid
        """
        # Skip very short chunks
        if len(chunk) < 2:
            return None
            
        # Skip if all tokens are stop words
        if all(tok.is_stop for tok in chunk):
            return None
        
        # Keep only meaningful POS tags
        meaningful_pos = {"NOUN", "PROPN", "ADJ", "NUM"}
        tokens = [tok for tok in chunk if tok.pos_ in meaningful_pos or not tok.is_stop]
        
        if len(tokens) < 2:
            return None
        
        # Build phrase from remaining tokens
        phrase = " ".join(tok.text.lower() for tok in tokens)
        
        # Final validation
        if len(phrase) < 5:
            return None
            
        return phrase
    
    def _rank_and_filter_phrases(
        self, 
        counts: Counter, 
        doc_freq: Counter, 
        n_texts: int,
        top_n: int = 10,
        max_df_ratio: float = 0.8
    ) -> List[str]:
        """
        Rank phrases by frequency and filter out overly common ones.
        
        Args:
            counts: Phrase frequency counts
            doc_freq: Document frequency per phrase
            n_texts: Total number of texts
            top_n: Number of phrases to return
            max_df_ratio: Max document frequency ratio (filter too-common phrases)
        
        Returns:
            Top N filtered phrases
        """
        max_df = max_df_ratio * n_texts
        return [
            phrase for phrase, _ in counts.most_common(top_n)
            if doc_freq[phrase] <= max_df
        ]

