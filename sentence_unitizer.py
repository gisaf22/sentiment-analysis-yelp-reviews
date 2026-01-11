"""
sentence_unitizer.py

Purpose
-------
Transform review-level text into sentence-level "aspect units" (fast proxy for aspect-based analysis).

Why this exists
---------------
Full reviews often contain multiple aspects (food, service, wait time). Treating each review as one document
creates blended embeddings and overlapping themes. Sentence-level units better match reality and yield cleaner,
more actionable theme assignment and trend analysis.

What it does
------------
- Splits each review into sentences
- Explodes to one row per sentence
- Adds time keys (month), sentence index, and a stable sentence_id
- Designed to be called from a notebook (no CLI / __main__ required)

Typical notebook usage
----------------------
    from sentence_unitizer import SentenceUnitizer

    unitizer = SentenceUnitizer(
        text_col="text_cleaned",
        id_col="review_id",
        date_col="date",
        keep_cols=["review_id", "business_id", "stars", "date", "text_cleaned"]
    )

    df_aspects = unitizer.transform(clean_reviews)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class SentenceUnitizer:
    """
    Review → sentence-level units.

    This class is intentionally narrow:
    it changes the *granularity* of the dataset from one row per review to one row per sentence,
    while preserving keys needed for later theme assignment and trend aggregation.
    """

    # Column configuration
    text_col: str = "text_cleaned"
    id_col: str = "review_id"
    date_col: str = "date"
    month_col: str = "month"

    # Quality controls
    min_chars: int = 10  # drop tiny sentence fragments
    keep_cols: Optional[List[str]] = None  # restrict columns passed through (performance/memory)

    # NOTE: We are not using regex lookbehind for abbreviations because Python re lookbehind
    # requires fixed-width patterns. If you need abbreviation protection, enable the optional
    # protection step in _split().
    ABBREV_LIST: List[str] = None  # will be set in __post_init__

    def __post_init__(self) -> None:
        """
        Post-constructor hook (dataclass pattern).

        Compiles the sentence-splitting regex once so we don't recompile for every row.
        We split only when the next token looks like a new sentence start to reduce junk splits.
        """
        # Simple + robust: split after .,!,? when next char looks like a new sentence start
        self._sent_split_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

        # Optional: abbreviation protection list (used in _split if enabled)
        self.ABBREV_LIST = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Mt.", "vs.", "etc.", "e.g.", "i.e."]

    # ---------------------------
    # Public API
    # ---------------------------
    def fit(self, df: pd.DataFrame, y=None) -> "SentenceUnitizer":
        """
        Pipeline-compatibility method.

        No learned parameters are required for sentence splitting, but we validate required columns here.
        Returns self to enable sklearn-style chaining.
        """
        self._validate_required_columns(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform review-level dataframe → sentence-level dataframe.

        Steps:
          1) Optionally restrict columns (keep_cols)
          2) Validate required columns exist
          3) Add time keys (month)
          4) Split + explode sentences
          5) Add sentence_idx and sentence_id for traceability
        """
        out = self._select_columns(df.copy())
        self._validate_required_columns(out)

        out = self._add_time_keys(out)
        out = self._explode_sentences(out)
        out = self._add_sentence_ids(out)
        return out

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Convenience method: fit + transform in one call.
        Useful when you want sklearn-like ergonomics.
        """
        return self.fit(df, y=y).transform(df)

    # ---------------------------
    # Helpers (internal)
    # ---------------------------
    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If keep_cols is provided, restrict the dataframe to those columns only.
        Reduces memory footprint before explode() multiplies rows.
        """
        if self.keep_cols is None:
            return df

        missing = [c for c in self.keep_cols if c not in df.columns]
        if missing:
            raise ValueError(f"SentenceUnitizer: missing keep_cols in input df: {missing}")

        return df[self.keep_cols].copy()

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Ensures the dataframe contains minimum columns needed to:
          - split text (text_col)
          - group by review (id_col)
          - build time windows (date_col)
        """
        required = [self.text_col, self.id_col, self.date_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"SentenceUnitizer: required column(s) missing: {missing}")

    def _add_time_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds month key used later for trend analysis.
        """
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.month_col] = df[self.date_col].dt.to_period("M").astype(str)
        return df

    def _split(self, text: str) -> List[str]:
        """
        Split a single review text into sentence strings.

        Notes:
        - Light normalization of whitespace
        - Filters out very short fragments (min_chars)
        - Returns list; empty list if text missing/blank

        Optional abbreviation protection:
        - Enabled by default below (safe, no lookbehind). If you dislike it,
          remove the protection block.
        """
        if pd.isna(text) or not str(text).strip():
            return []

        t = re.sub(r"\s+", " ", str(text)).strip()

        # --- OPTIONAL: protect common abbreviations so we don't split after "Dr." etc. ---
        for ab in self.ABBREV_LIST:
            t = t.replace(ab, ab.replace(".", "<DOT>"))

        sents = self._sent_split_re.split(t)

        # Restore periods
        sents = [s.replace("<DOT>", ".").strip() for s in sents]
        return [s for s in sents if len(s) >= self.min_chars]

    def _explode_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies _split to produce list of sentences per review and explodes to one row per sentence.
        """
        df["sentence_list"] = df[self.text_col].apply(self._split)
        df = df.explode("sentence_list", ignore_index=True).rename(columns={"sentence_list": "sentence"})

        df["sentence"] = df["sentence"].fillna("").astype(str).str.strip()
        df = df[df["sentence"].str.len() >= self.min_chars].copy()
        return df

    def _add_sentence_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds sentence_idx and stable sentence_id for traceability and evidence extraction.
        """
        df["sentence_idx"] = df.groupby(self.id_col).cumcount()
        df["sentence_id"] = df[self.id_col].astype(str) + "_s" + df["sentence_idx"].astype(str)
        return df

    # ---------------------------
    # Notebook convenience helper
    # ---------------------------
    def preview_review(self, df_aspects: pd.DataFrame, review_id: str, n: int = 50) -> pd.DataFrame:
        """
        Shows the sentence splits for a single review_id to verify splitter quality.
        """
        sub = df_aspects[df_aspects[self.id_col] == review_id].copy()
        cols = [self.id_col, "sentence_idx", "sentence_id", "sentence"]
        cols = [c for c in cols if c in sub.columns]
        return sub.sort_values("sentence_idx")[cols].head(n)
