from collections import defaultdict

from rag.indexing.dense import DenseRetriever
from rag.indexing.sparse import BM25Retriever


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_c: int = 60
    ):
        self.dense = dense
        self.sparse = sparse
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_c = rrf_c

    def search(self, query: str, k: int = 20) -> list[dict]:
        dense_results = self.dense.search(query, k=k*2)
        bm25_results = self.sparse.search(query, k=k*2)

        # RRF слияние
        scores = defaultdict(float)
        first_occurrence = {}

        # Dense результаты
        for rank, item in enumerate(dense_results, 1):
            doc_id = item["pid"]
            scores[doc_id] += self.dense_weight / (self.rrf_c + rank)
            first_occurrence.setdefault(doc_id, item)

        # BM25 результаты
        for rank, item in enumerate(bm25_results, 1):
            doc_id = item["pid"]
            scores[doc_id] += self.bm25_weight / (self.rrf_c + rank)
            first_occurrence.setdefault(doc_id, item)

        # Сортируем и возвращаем топ-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        fused = []
        for doc_id, score in ranked:
            item = dict(first_occurrence[doc_id])
            item["score"] = float(score)  # RRF score
            fused.append(item)

        return fused
