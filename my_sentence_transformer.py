import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


class MySentenceTransformer:
    """
    High-quality sentence embedding wrapper for:
      - Clustering
      - Theme extraction
      - Semantic deduplication
      - Retrieval

    Supports MiniLM and MPNet with automatic device selection.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        """
        Parameters
        ----------
        model_name : str
            Example:
              - "all-MiniLM-L6-v2"   (fast, good)
              - "all-mpnet-base-v2" (slow, best quality)
        device : str | None
            "cuda", "cpu", or None for auto-detect
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name

    # ---------------------------------------------------------
    # CORE ENCODING
    # ---------------------------------------------------------

    def encode(
        self,
        texts,
        batch_size=64,
        normalize_output=True,
        show_progress=True
    ):
        """
        Encode a list of texts into dense embeddings.

        Parameters
        ----------
        texts : List[str]
        batch_size : int
        normalize_output : bool
            MUST be True for cosine clustering.
        show_progress : bool

        Returns
        -------
        np.ndarray
            Shape: (n_samples, embedding_dim)
        """

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        if normalize_output:
            embeddings = self.normalize_embeddings(embeddings)

        return embeddings

    # ---------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def normalize_embeddings(embeddings):
        """
        L2-normalize embeddings for:
          - Cosine similarity
          - Whitening
          - KMeans on spherical space
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return normalize(embeddings)

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------

    def embedding_dim(self):
        """
        Returns embedding dimension without encoding.
        """
        return self.model.get_sentence_embedding_dimension()

    def info(self):
        """
        Quick human-readable model diagnostics.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": self.embedding_dim(),
        }

