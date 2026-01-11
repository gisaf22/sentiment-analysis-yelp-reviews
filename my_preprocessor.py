from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import contractions
import re
from unidecode import unidecode
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class MyPreProcessor:
    """
    Generic text preprocessing for either reviews or sentence units.

    Designed for reuse across:
      - review-level preprocessing (text -> text_cleaned)
      - sentence-level preprocessing (sentence -> sentence_cleaned)

    Key idea:
      Configure which column to read (text_col) and which column to write (out_col).
    """

    def __init__(self, min_words=3):
        self.min_words = min_words
        self.URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

    def remove_html_tags(self, text: str) -> str:
        """
        Removes HTML, scripts, links, URLs, and normalizes line breaks.
        Safe on non-string inputs.
        """
        try:
            if not isinstance(text, str):
                return ""

            soup = BeautifulSoup(text, "html.parser")

            for tag in soup.find_all(["iframe", "script", "a"]):
                tag.decompose()

            stripped_text = soup.get_text()
            stripped_text = self.URL_PATTERN.sub("", stripped_text)
            stripped_text = re.sub(r"\r\n?|\n", " ", stripped_text)

            return stripped_text

        except Exception as e:
            print(f"HTML parse error: {e}")
            return ""

    def prepare(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        out_col: str = "text_cleaned",
        dedupe_on: str | None = None,
        lowercase: bool = True,
        expand_contractions: bool = True,
        normalize_unicode: bool = True,
        remove_html: bool = True,
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline, configurable by column.

        Parameters
        ----------
        text_col : str
            Input text column name ("text" for reviews, "sentence" for sentence units).
        out_col : str
            Output cleaned text column name ("text_cleaned", "sentence_cleaned", etc.).
        dedupe_on : str | None
            Column to drop exact duplicates on. If None, no dedupe.
            - For reviews: "text"
            - For sentences: "sentence" or the cleaned output column (after creation)
        lowercase : bool
            Whether to lowercase output.
        """
        df = df.copy()

        if text_col not in df.columns:
            raise ValueError(f"prepare(): text_col '{text_col}' not found in df columns.")

        # 1) Optional: Drop exact duplicates (before heavy work)
        if dedupe_on is not None:
            if dedupe_on not in df.columns:
                raise ValueError(f"prepare(): dedupe_on '{dedupe_on}' not found in df columns.")
            df = df.drop_duplicates(subset=dedupe_on, keep="first").reset_index(drop=True)

        # 2) Clean base text
        series = df[text_col].astype(str)

        # 3) Remove HTML + URLs (helpful for raw review HTML; usually unnecessary for sentence units)
        if remove_html:
            series = series.apply(self.remove_html_tags)
        else:
            series = series.apply(lambda x: self.URL_PATTERN.sub("", x))
            series = series.str.replace(r"\r\n?|\n", " ", regex=True)

        # 4) Expand contractions
        if expand_contractions:
            series = series.apply(contractions.fix)

        # 5) Unicode normalization + lowercase
        if normalize_unicode:
            series = series.apply(unidecode)

        if lowercase:
            series = series.str.lower()

        # 6) Normalize whitespace
        series = series.str.replace(r"\s+", " ", regex=True).str.strip()

        df[out_col] = series

        # 7) Drop short texts
        mask = df[out_col].str.split().str.len() >= self.min_words
        df = df[mask].reset_index(drop=True)

        return df