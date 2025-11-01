import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import pandas as pd
import faiss

from rag.indexing.embedder import Embedder
from config import settings


def main():
    print("Построение FAISS индекса...")

    if not settings.PROCESSED_RAG_CSV.exists():
        print(f"Ошибка: {settings.PROCESSED_RAG_CSV} не найден")
        sys.exit(1)

    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Загрузка
    df = pd.read_csv(settings.PROCESSED_RAG_CSV)
    docs = df['doc_text_rag'].fillna('').tolist()
    pids = df.index.values.astype(int)
    print(f"Загружено {len(df)} товаров")
    print("Векторизация...")
    embedder = Embedder(settings.EMBEDDING_MODEL)
    doc_emb = embedder.encode_passages(docs)
    dim = embedder.dimension
    print("Создание FAISS индекса...")
    index = faiss.IndexFlatIP(dim)

    # Добавляем по батчам
    batch_size = 100
    for i in range(0, len(doc_emb), batch_size):
        batch = doc_emb[i:i+batch_size]
        index.add(batch)
        if (i + batch_size) % 500 == 0:
            print(f"  Обработано {min(i+batch_size, len(doc_emb))}/{len(doc_emb)}")

    # Сохранение
    faiss.write_index(index, str(settings.FAISS_INDEX_PATH))
    with open(settings.METADATA_PATH, 'wb') as f:
        pickle.dump({
            'pids': pids,
            'model_name': settings.EMBEDDING_MODEL,
            'dimension': dim
        }, f)

    print(f"✓ Готово: {index.ntotal} векторов, размерность {dim}")


if __name__ == "__main__":
    main()
