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

    def search(self, query: str, k: int = 20) -> list[dict]:
        """
        Поиск через FAISS
        """
        q_emb = self.embedder.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, k)
        results = []
        for idx_row, score in zip(indices[0], scores[0]):
            if idx_row < 0:
                continue
            row = self.df.iloc[idx_row]
            result = {
                "pid": int(idx_row),
                "title": str(row["title"]) if "title" in row else "",
                "url": str(row["url"]) if "url" in row else "#",
                "score": float(score),
                "brand": str(row["brand"]) if "brand" in row and pd.notna(row["brand"]) else "",
                "price_num": float(row["price_num"]) if "price_num" in row and pd.notna(row["price_num"]) else 0,
                "image_url": str(row["image_url"]) if "image_url" in row and pd.notna(row["image_url"]) else "",
                "price": str(row["price"]) if "price" in row and pd.notna(row["price"]) else "",
                "old_price": str(row["old_price"]) if "old_price" in row and pd.notna(row["old_price"]) else "",
                "description": str(row["description"]) if "description" in row and pd.notna(row["description"]) else ""
            }
            results.append(result)

        return results
