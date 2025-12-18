from bs4 import BeautifulSoup
import contractions
import re
from unidecode import unidecode
from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class MyPreProcessor:
    """
    Handles all text preprocessing before lemmatization.
    Designed for reuse across businesses and pipelines.
    """

    def __init__(self, min_words=3):
        self.min_words = min_words
        self.URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

    def remove_html_tags(self, text):
        """
        Removes HTML, scripts, links, URLs, and normalizes line breaks.
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

    def prepare(self, df):
        """
        Full preprocessing pipeline:
        - Exact deduplication
        - HTML + URL removal
        - Contraction expansion
        - Unicode normalization
        - Lowercasing
        - Whitespace normalization
        - Short-review removal
        """
        df = df.copy()

        # 1. Drop exact duplicate raw reviews
        df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

        # 2. Remove HTML + URLs
        df["text_cleaned"] = df["text"].astype(str).apply(self.remove_html_tags)

        # 3. Expand contractions
        df["text_cleaned"] = df["text_cleaned"].apply(contractions.fix)

        # 4. Normalize unicode + lowercase
        df["text_cleaned"] = df["text_cleaned"].apply(unidecode).str.lower()

        # 5. Normalize whitespace
        df["text_cleaned"] = (
            df["text_cleaned"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # 6. Drop short reviews
        mask = df["text_cleaned"].str.split().str.len() >= self.min_words
        df = df[mask].reset_index(drop=True)

        return df
