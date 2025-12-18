import numpy as np
import pandas as pd
import string

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


class MyFeatureEngineering:
    """
    Adds linguistic and statistical text features.
    Uses POS-aware NLTK lemmatization.
    Assumes input already ran through MyPreProcessor.
    """

    def __init__(self):
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    # --------------------------------------------------
    # POS-AWARE LEMMATIZER (FIXED)
    # --------------------------------------------------
    def _nltk_pos_to_wordnet(self, tag):
        if tag.startswith("J"):
            return wordnet.ADJ
    
        # IMPORTANT: treat VBG/VBN as adjectives in reviews
        if tag in {"VBG", "VBN"}:
            return wordnet.ADJ
    
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV

        return wordnet.NOUN

    def tokenize_and_lemmatize(self, text):
        tokens = [
            t.lower()
            for t in word_tokenize(text)
            if t.isalpha()
        ]

        tagged = pos_tag(tokens)

        lemmas = [
            self.lemmatizer.lemmatize(token, self._nltk_pos_to_wordnet(pos))
            for token, pos in tagged
            if token not in self.stopwords
        ]

        return " ".join(lemmas)

    def lemmatize_corpus(self, texts):
        """
        Lemmatizes a list of texts and returns lemma strings
        (POS-aware NLTK – correct for TF-IDF & NMF)
        """
        return [self.tokenize_and_lemmatize(t) for t in texts]

    # --------------------------------------------------
    # NUMERIC FEATURES (UNCHANGED)
    # --------------------------------------------------
    def compute_basic_stats(self, texts):
        word_counts = []
        char_counts = []
        avg_word_len = []
        sentence_counts = []

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
    # PUBLIC API (UNCHANGED)
    # --------------------------------------------------
    def add_features(self, df):
        df = df.copy()

        texts = df["text_cleaned"].astype(str).tolist()

        # POS-aware lemmatization
        df["lemmas"] = self.lemmatize_corpus(texts)

        # Basic stats
        wc, cc, awl, sc = self.compute_basic_stats(texts)

        df["word_count"] = wc
        df["char_count"] = cc
        df["avg_word_length"] = awl
        df["sentence_count"] = sc

        return df
