import numpy as np
import pandas as pd

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


class MyFeatureEngineering:
    """
    Adds linguistic and statistical text features (POS-aware NLTK lemmatization).

    Designed to work at different granularities:
      - Review-level: text_cleaned -> lemmas + numeric stats
      - Sentence-level: sentence_cleaned -> lemmas + numeric stats

    Negation preservation
    ---------------------
    By default, NLTK stopwords include "not", "no", "nor" which would erase negation
    (e.g., "not too long" -> "long"). This class preserves negation by:
      1) Removing negators from the stopword set
      2) Optionally marking negation scope as NOT_<token> for the next N tokens
         to keep "not long" distinct from "long" in TF-IDF/NMF.
    """

    def __init__(self, negation_scope: int = 0):
        """
        Parameters
        ----------
        negation_scope : int
            0  -> keep negation words only (recommended baseline)
            >0 -> additionally mark the next N tokens after a negator as NOT_<token>
                  (stronger separation for lexical topic models)
        """
        base_stopwords = set(stopwords.words("english"))

        # Preserve negation tokens (critical for sentiment & theme direction)
        self.negators = {"not", "no", "nor", "never"}
        self.stopwords = base_stopwords - self.negators

        self.lemmatizer = WordNetLemmatizer()

        # Negation handling mode
        self.negation_scope = int(negation_scope) if negation_scope else 0

    # --------------------------------------------------
    # POS-AWARE LEMMATIZER
    # --------------------------------------------------
    def _nltk_pos_to_wordnet(self, tag: str):
        """Map NLTK POS tags to WordNet POS tags used by the lemmatizer."""
        if tag.startswith("J"):
            return wordnet.ADJ

        # Treat VBG/VBN as adjectives in review language ("grilled", "cooked", "waiting")
        if tag in {"VBG", "VBN"}:
            return wordnet.ADJ

        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV

        return wordnet.NOUN

    # --------------------------------------------------
    # NEGATION MARKING (OPTIONAL)
    # --------------------------------------------------
    def _mark_negation(self, tokens: list[str]) -> list[str]:
        """
        Optionally mark negation scope to prevent collapsing opposite meanings.

        Example (scope=2):
            ["wait", "was", "not", "too", "long"] ->
            ["wait", "was", "not", "NOT_too", "NOT_long"]

        Notes
        - This is primarily helpful for TF-IDF/NMF where tokens are the feature space.
        - For embeddings, you typically would not use these marked tokens.
        """
        if self.negation_scope <= 0:
            return tokens

        out: list[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            out.append(tok)

            if tok in self.negators:
                # mark next N alpha tokens (stop if punctuation appears, but we filter isalpha() anyway)
                marked = 0
                j = i + 1
                while j < len(tokens) and marked < self.negation_scope:
                    nxt = tokens[j]
                    # do not mark other negators
                    if nxt not in self.negators:
                        out.append(f"NOT_{nxt}")
                        marked += 1
                    j += 1

            i += 1

        return out

    def tokenize_and_lemmatize(self, text: str) -> str:
        """
        Tokenize, POS-tag, preserve negation, remove stopwords, and lemmatize.

        Returns
        -------
        str
            A whitespace-joined lemma string suitable for TF-IDF / NMF.
        """
        # Tokenize; keep alphabetic tokens only (consistent with your prior logic)
        tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]

        # Optional: add NOT_<token> markers for lexical topic modeling
        tokens = self._mark_negation(tokens)

        # POS tag AFTER marking. Tagged tokens like "NOT_long" won't be isalpha(),
        # so we will treat them separately below.
        # Split tokens into normal alpha tokens and NOT_ markers.
        alpha_tokens = [t for t in tokens if t.isalpha()]
        tagged = pos_tag(alpha_tokens)

        # Lemmatize alpha tokens; keep negators (since we removed them from stopwords)
        lemmas = []
        for token, pos in tagged:
            if token in self.stopwords:
                continue
            lemma = self.lemmatizer.lemmatize(token, self._nltk_pos_to_wordnet(pos))
            lemmas.append(lemma)

        # Append NOT_ markers as-is (they carry meaning for TF-IDF/NMF)
        # Keep them only if scope mode is enabled.
        if self.negation_scope > 0:
            not_markers = [t for t in tokens if t.startswith("NOT_")]
            # normalize marker case and remove any accidental duplicates in sequence
            lemmas.extend(not_markers)

        return " ".join(lemmas)

    def lemmatize_corpus(self, texts):
        """Lemmatize a list of texts and return lemma strings (POS-aware)."""
        return [self.tokenize_and_lemmatize(t) for t in texts]

    # --------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------
    def compute_basic_stats(self, texts):
        """
        Compute basic numeric text stats.

        Returns lists aligned with `texts`:
        - word_count, char_count, avg_word_length, sentence_count
        """
        word_counts, char_counts, avg_word_len, sentence_counts = [], [], [], []

        for txt in texts:
            words = txt.split()
            wc = len(words)
            cc = len(txt)
            awl = np.mean([len(w) for w in words]) if wc else 0
            sc = max(1, txt.count(".") + txt.count("!") + txt.count("?"))

            word_counts.append(wc)
            char_counts.append(cc)
            avg_word_len.append(awl)
            sentence_counts.append(sc)

        return word_counts, char_counts, avg_word_len, sentence_counts

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def add_features(
        self,
        df: pd.DataFrame,
        text_col: str = "text_cleaned",
        lemma_col: str = "lemmas",
        add_stats: bool = True,
        prefix: str | None = None,
    ) -> pd.DataFrame:
        """
        Add lemma text (negation-preserving) and optional numeric stats for a specified text column.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        text_col : str
            Column containing cleaned text to featurize (e.g., "text_cleaned" or "sentence_cleaned").
        lemma_col : str
            Output column name for lemma strings.
        add_stats : bool
            Whether to compute numeric stats.
        prefix : str | None
            If provided, numeric stat columns are prefixed (useful to avoid collisions when
            adding features at multiple granularities).

        Returns
        -------
        pd.DataFrame
            Copy of df with new columns added.
        """
        df = df.copy()

        if text_col not in df.columns:
            raise ValueError(f"MyFeatureEngineering.add_features: '{text_col}' not found in df columns.")

        texts = df[text_col].astype(str).tolist()

        # Lemmas (POS-aware + negation-preserving)
        df[lemma_col] = self.lemmatize_corpus(texts)

        if add_stats:
            wc, cc, awl, sc = self.compute_basic_stats(texts)

            p = f"{prefix}_" if prefix else ""
            df[f"{p}word_count"] = wc
            df[f"{p}char_count"] = cc
            df[f"{p}avg_word_length"] = awl
            df[f"{p}sentence_count"] = sc

        return df