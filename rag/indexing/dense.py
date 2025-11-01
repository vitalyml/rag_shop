import pickle
from pathlib import Path
import faiss
import pandas as pd

from .embedder import Embedder


class DenseRetriever:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedder: Embedder,
        df: pd.DataFrame
    ):
        self.index = faiss.read_index(str(index_path))
        self.embedder = embedder
        self.df = df

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.pids = metadata['pids']
            self.dimension = metadata['dimension']

    def search(self, query: str, k: int = 20) -> list[dict]:
        """
        Поиск через FAISS (cosine similarity)
        """
        q_emb = self.embedder.encode_queries([query])
        D, I = self.index.search(q_emb, k)

        results = []
        for doc_idx, score in zip(I[0].tolist(), D[0].tolist()):
            pid = int(self.pids[doc_idx])
            row = self.df.loc[pid, ['title', 'brand', 'price_num', 'url']].to_dict()
            results.append({
                'pid': pid,
                'score': float(score),
                **row
            })

        return results
