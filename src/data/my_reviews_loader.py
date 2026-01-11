import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


class BusinessReviewLoader:
    """
    Loads Yelp businesses using pandas, and large Yelp reviews using Spark.
    Filters reviews by business safely before converting to pandas.
    """

    def __init__(
        self,
        business_path,
        reviews_path,
        id_col="business_id",
        name_col="name",
        text_col="text",
        spark_driver_memory="8g"
    ):
        self.id_col = id_col
        self.name_col = name_col
        self.text_col = text_col

        # --- Load businesses in pandas (small table) ---
        self.business_df = self._load_pandas_file(business_path)

        # --- Start Spark ---
        self.spark = SparkSession.builder \
            .appName("YelpReviewLoader") \
            .config("spark.driver.memory", spark_driver_memory) \
            .getOrCreate()

        # --- Load reviews in Spark (large table) ---
        self.reviews_df = self._load_spark_file(reviews_path)

        self._validate_columns()

    # ------------------------------------------------------------------
    # FILE LOADERS
    # ------------------------------------------------------------------

    def _load_pandas_file(self, path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".json"):
            return pd.read_json(path, lines=True)
        else:
            raise ValueError("Business file must be .parquet, .csv, or .json")

    def _load_spark_file(self, path):
        if not path.endswith(".parquet"):
            raise ValueError("Spark reviews loader requires PARQUET file")
        return self.spark.read.parquet(path)

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_columns(self):
        if self.id_col not in self.business_df.columns:
            raise ValueError(f"{self.id_col} missing from business file")

        if self.id_col not in self.reviews_df.columns:
            raise ValueError(f"{self.id_col} missing from reviews parquet")

        if self.name_col not in self.business_df.columns:
            raise ValueError(f"{self.name_col} missing from business file")

        if self.text_col not in self.reviews_df.columns:
            raise ValueError(f"{self.text_col} missing from reviews parquet")

    # ------------------------------------------------------------------
    # BUSINESS LOOKUP
    # ------------------------------------------------------------------

    def get_reviews_by_business_name(self, business_name, exact=True, require_open=True):
        name = business_name.strip().lower()

        df = self.business_df.copy()
        df["name_clean"] = df[self.name_col].astype(str).str.lower().str.strip()

        # --- Name matching ---
        if exact:
            matches = df[df["name_clean"] == name]
        else:
            matches = df[df["name_clean"].str.contains(name, regex=False)]

        # --- Enforce open businesses only ---
        if require_open and "is_open" in matches.columns:
            matches = matches[matches["is_open"] == 1]

        if matches.empty:
            raise ValueError(
                f"No OPEN business found with name: '{business_name}' "
                f"(exact={exact})"
            )

        # --- Disambiguate using highest review count ---
        if "review_count" in matches.columns:
            matches = matches.sort_values("review_count", ascending=False)

        chosen = matches.iloc[0]
        business_id = chosen[self.id_col]

        # --- Explicit verification (no silent mismatch ever again) ---
        print("\n Selected Business:")
        for col in [self.id_col, self.name_col, "city", "state", "review_count", "is_open"]:
            if col in chosen:
                print(f"{col}: {chosen[col]}")

        return self.get_reviews_by_business_id(business_id)

    # ------------------------------------------------------------------
    # REVIEW EXTRACTION (SAFE SPARK FILTER → PANDAS)
    # ------------------------------------------------------------------

    def get_reviews_by_business_id(self, business_id):
        spark_filtered = self.reviews_df.filter(
            col(self.id_col) == business_id
        )

        pdf = spark_filtered.toPandas()

        # IMPORTANT: stop Spark immediately after conversion
        self.spark.stop()

        return pdf

