import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CorpusDenoiser:
    """
    Removes low-signal and near-duplicate texts after embedding.
    Algorithm-agnostic and clustering-independent.
    """

    def __init__(self, min_words=6, dup_thresh=0.985):
        self.min_words = min_words
        self.dup_thresh = dup_thresh

    def fit_transform(self, df, X_embed, text_col="text_cleaned"):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Text data
        X_embed : np.ndarray
            Embeddings aligned with df
        text_col : str
            Column used for length filtering
        """

        # --------------------------------------------------
        # 1. Filter short texts
        # --------------------------------------------------
        mask_len = df[text_col].str.split().str.len() >= self.min_words
        df = df[mask_len].reset_index(drop=True)
        X_embed = X_embed[mask_len.values]

        if len(df) == 0:
            return df, X_embed

        # --------------------------------------------------
        # 2. Remove near-duplicates (cosine similarity)
        # --------------------------------------------------
        sim = cosine_similarity(X_embed)
        np.fill_diagonal(sim, 0)

        keep = np.ones(len(df), dtype=bool)

        for i in range(len(df)):
            if not keep[i]:
                continue
            dup_idx = np.where(sim[i] > self.dup_thresh)[0]
            keep[dup_idx] = False

        return (
            df[keep].reset_index(drop=True),
            X_embed[keep],
        )
