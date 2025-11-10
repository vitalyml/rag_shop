import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Embedder:
    """Класс для векторизации текстов с использованием ru-en-RoSBERTa"""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode_passages(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Кодирование документов батчами с нормализацией для косинусного сходства        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embs = self.model.encode(batch, convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(embs)
            all_embeddings.append(embs)

        return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Кодирование запросов с нормализацией для косинусного сходства        """
        embs = self.model.encode(queries, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embs)
        return embs

    @property
    def dimension(self) -> int:
        """Размерность эмбеддингов"""
        return self.model.get_sentence_embedding_dimension()
