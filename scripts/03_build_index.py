import sys
import os
from pathlib import Path

# Fix для macOS: отключаем многопоточность ДО импорта библиотек
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import settings


def main():
    print("Построение FAISS IndexFlatIP индекса...")

    if not settings.PROCESSED_RAG_CSV.exists():
        print(f"Ошибка: {settings.PROCESSED_RAG_CSV} не найден")
        sys.exit(1)

    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(settings.PROCESSED_RAG_CSV)
    texts = df['doc_text_rag'].fillna('').tolist()
    print(f"Загружено {len(df)} товаров")
    print("Инициализация модели и индекса...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dimension)
    print("Векторизация и индексация...")
    batch_size = 128
    pbar = tqdm(total=len(texts), desc="Индексация", unit="doc")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embs)
        index.add(embs)
        pbar.update(len(batch))

    pbar.close()

    faiss.write_index(index, str(settings.FAISS_INDEX_PATH))

    print(f"Готово, индекс: {index.ntotal} векторов, размерность {dimension}")


if __name__ == "__main__":
    main()
