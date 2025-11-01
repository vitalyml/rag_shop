import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from rag.preprocessing.text import simple_tokenize


class BM25Retriever:
    """BM25-based sparse retrieval"""

    def __init__(
        self,
        bm25_path: Path,
        df: pd.DataFrame
    ):
        with open(bm25_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']  # SimpleBM25 or BM25Okapi
            self.pids = data['pids']

        self.df = df

    def search(self, query: str, k: int = 20) -> list[dict]:
        """
        BM25 поиск по ключевым словам

        Args:
            query: поисковый запрос
            k: количество результатов

        Returns:
            список словарей с результатами
        """
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for doc_idx in top_indices:
            pid = int(self.pids[doc_idx])
            row = self.df.loc[pid, ['title', 'brand', 'price_num', 'url']].to_dict()
            results.append({
                'pid': pid,
                'score': float(scores[doc_idx]),
                **row
            })

        return results
