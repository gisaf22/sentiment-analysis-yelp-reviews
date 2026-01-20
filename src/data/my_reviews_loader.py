import pandas as pd
import pyarrow.parquet as pq


class BusinessReviewLoader:
    """
    Loads Yelp businesses using pandas.
    Loads reviews using PyArrow with pushdown predicate filtering.
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
        self.reviews_path = reviews_path

        # --- Load businesses in pandas (small table) ---
        self.business_df = self._load_pandas_file(business_path)

        # Don't load full reviews here - load on demand per business with predicate pushdown
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
            raise ValueError("File must be .parquet, .csv, or .json")

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_columns(self):
        if self.id_col not in self.business_df.columns:
            raise ValueError(f"{self.id_col} missing from business file")

        if self.name_col not in self.business_df.columns:
            raise ValueError(f"{self.name_col} missing from business file")

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

        # --- Explicit verification ---
        print("\nSelected Business:")
        for col in [self.id_col, self.name_col, "city", "state", "review_count", "is_open"]:
            if col in chosen:
                print(f"{col}: {chosen[col]}")

        return self.get_reviews_by_business_id(business_id)

    # ------------------------------------------------------------------
    # REVIEW EXTRACTION (PYARROW PUSHDOWN PREDICATE FILTERING)
    # ------------------------------------------------------------------

    def get_reviews_by_business_id(self, business_id):
        """
        Load only reviews for this business using PyArrow pushdown predicates.
        Filter is applied at the parquet file level - only matching row groups loaded.
        This is the most efficient approach for large parquet files.
        """
        # Use filters parameter for predicate pushdown with read_table
        # PyArrow uses row group metadata to skip non-matching row groups
        
        # Build filter: [[(column, operator, value)]]
        # Double-nested list required for proper OR/AND logic
        filters = [[(self.id_col, '==', business_id)]]
        
        # Read with pushdown predicate - only matching row groups loaded from disk
        table = pq.read_table(self.reviews_path, filters=filters)
        
        # Convert to pandas
        df = table.to_pandas()
        
        return df.reset_index(drop=True)


