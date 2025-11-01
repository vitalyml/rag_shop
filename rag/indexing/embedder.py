import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Класс для векторизации текстов с использованием E5 модели"""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """
        Кодирование документов с префиксом 'passage:'

        Args:
            texts: список текстов документов

        Returns:
            массив эмбеддингов shape (N, D)
        """
        prefixed = [f"passage: {t if isinstance(t, str) else ''}" for t in texts]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True
        ).astype("float32")

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """
        Кодирование запросов с префиксом 'query:'

        Args:
            queries: список поисковых запросов

        Returns:
            массив эмбеддингов shape (N, D)
        """
        prefixed = [f"query: {q}" for q in queries]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True
        ).astype("float32")

    @property
    def dimension(self) -> int:
        """Размерность эмбеддингов"""
        return self.model.get_sentence_embedding_dimension()
