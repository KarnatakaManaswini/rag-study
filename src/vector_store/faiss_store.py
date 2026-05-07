import faiss
import numpy as np


class FAISSStore:
    def __init__(self, embeddings: np.ndarray, texts: list[str]):
        self.texts = texts
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k=3):
        query_embedding = query_embedding.reshape(1, -1)
        _, indices = self.index.search(query_embedding, top_k)

        return [self.texts[i] for i in indices[0]]
