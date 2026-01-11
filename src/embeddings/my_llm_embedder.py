import numpy as np
from InstructorEmbedding import INSTRUCTOR

class MyHuggingFaceEmbedder:
    def __init__(self, model_name="hkunlp/instructor-xl"):
        self.model = INSTRUCTOR(model_name)

    def transform_sentences(self, texts, instruction="Represent the Yelp review for sentiment analysis:"):
        inputs = [[instruction, t] for t in texts]
        return np.array(self.model.encode(inputs))

    def normalize_embeddings(self, embeddings):
        from sklearn.preprocessing import normalize
        return normalize(embeddings, axis=1)