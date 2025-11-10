import json
import re
from collections import defaultdict
from typing import Callable

from config import settings


def generate_rewrites(query: str, llm_client) -> list[str]:
    REWRITE_PROMPT = """Ты помогаешь улучшить поисковый запрос для интернет-магазина одежды и обуви.

ЗАДАЧА: Создай синонимичный запрос с использованием альтернативных слов.

ПРАВИЛА:
1. Убери команды и глаголы: "найди", "покажи", "посоветуй", "подбери", "хочу", "ищу"
2. Замени слова на синонимы:
   - цвета: белый → светлый, черный → темный, и т.д.
   - типы товаров: рубашка → сорочка, кроссовки → кеды/спортивная обувь
   - стили: классический → деловой, спортивный → casual
3. Сохрани все характеристики (цвет, стиль, бренд, материал)
4. Если в запросе бренд - оставь его без изменений

Примеры:
- "найди белую рубашку" → "светлая сорочка"
- "черные кроссовки найк" → "темные кеды nike"
- "покажи синие джинсы" → "голубые штаны деним"

Верни ТОЛЬКО JSON с одним полем:
{{"rewrite": "новый_запрос_с_синонимами"}}

Запрос: "{query}"
"""

    prompt_text = REWRITE_PROMPT.format(query=query)

    try:
        content = llm_client.chat(
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=150,
            temperature=0.3
        )

        # Ищем JSON в ответе
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(content)

        rewrite = data.get("rewrite", "").strip()
        print(f"Parsed rewrite: '{rewrite}'")

        if not rewrite or rewrite == query:
            print(f"Rewrite is empty or same as query, returning [query]")
            return [query]

    except Exception as e:
        print(f"Warning: Failed to parse rewrite response: {e}")
        print(f"Raw content was: {content if 'content' in locals() else 'NO CONTENT'}")
        return [query]


def fuse_results_rrf(
    rewrites: list[str],
    search_fn: Callable,
    per_query_k: int = 5,
    final_k: int = 20,
    c: int = 60
) -> list[dict]:
    query_results = [search_fn(q, k=per_query_k) for q in rewrites]

    scores = defaultdict(float)
    first_occurrence = {}
    contributors = defaultdict(list)

    for query_id, results in enumerate(query_results, 1):
        for rank, item in enumerate(results, 1):
            doc_id = item["pid"]
            scores[doc_id] += 1.0 / (c + rank)
            first_occurrence.setdefault(doc_id, item)
            contributors[doc_id].append({"rewrite_id": query_id, "rank": rank})

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:final_k]

    fused = []
    for doc_id, score in ranked:
        item = dict(first_occurrence[doc_id])
        item["rrf_score"] = float(score)
        item["contributors"] = contributors[doc_id]
        fused.append(item)

    return fused
