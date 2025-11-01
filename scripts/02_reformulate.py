import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from config import settings


client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


async def reformulate_for_rag(text: str, semaphore: asyncio.Semaphore) -> str:
    """Переформулировка описания товара для семантического поиска"""
    prompt = f"""Переформулируй описание товара для семантического поиска.

ВАЖНО:
- Начни с указания категории товара (кроссовки, футболка, куртка, сумка и т.д.)
- Цвета описывай простыми словами (черный, белый, красный, синий, серый, зеленый и т.д.)
- Пиши ТОЛЬКО о том, что есть в описании. Если нет информации - не упоминай.

ВКЛЮЧИ если есть:
- Категория товара
- Бренд и модель
- Коллаборации
- Цвета (простыми словами)
- Материалы
- Конкретные детали (логотипы, принты, особые элементы)
- Стиль
- Сезонность (лето, зима и т.д.), если указано
- Особые функции (водонепроницаемость, утепление и т.д.), если указано

УБЕРИ:
- Маркетинг ("идеальный выбор", "погрузитесь")
- Общие слова ("стильный", "современный", "качественный")
- Домыслы о том, чего нет в тексте

Исходное описание:
{text}

Переформулированное:"""

    async with semaphore:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "Извлекай факты из описания товара. Начинай с категории товара. Цвета описывай простым языком. Пиши только о том, что явно указано в тексте."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=600
        )

    return response.choices[0].message.content.strip()


async def process_batch(texts: list[str], max_concurrent: int = 20) -> list[str]:
    """Обработка батча текстов с ограничением параллелизма"""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [reformulate_for_rag(text, semaphore) for text in texts]
    return await tqdm.gather(*tasks, desc="Переформулировка")


async def reformulate_dataframe_async(df: pd.DataFrame, column: str = 'doc_text') -> pd.DataFrame:
    """Переформулировка всех описаний в датафрейме"""
    texts = df[column].tolist()
    results = await process_batch(texts)
    df['doc_text_rag'] = results
    return df


def main():
    print(f"\n1. Загрузка данных из {settings.PROCESSED_CSV}...")
    df = pd.read_csv(settings.PROCESSED_CSV)
    print(f"   Загружено {len(df)} товаров")

    print("\n2. Переформулировка описаний...")
    df = asyncio.run(reformulate_dataframe_async(df))

    print(f"\n3. Сохранение в {settings.PROCESSED_RAG_CSV}...")
    df.to_csv(settings.PROCESSED_RAG_CSV, index=False)

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print(f"\nСохранено: {settings.PROCESSED_RAG_CSV}")
    print(f"Товаров: {len(df)}")


if __name__ == "__main__":
    main()
