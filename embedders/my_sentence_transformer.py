from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

class MySentenceTransformer:

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def transform_sentences(self, sents):
        """
        Encodes a single sentence (str) or a list of sentences (List[str])
        into embeddings (NumPy array)
        """
        return self.model.encode(sents, convert_to_numpy=True)

    def normalize_embeddings(self, embeddings):
        """
        L2-normalizes embeddings along the last axis.
        Useful for cosine similarity or clustering.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return normalize(embeddings)