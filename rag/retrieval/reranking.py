import json
import re
from collections import defaultdict
from typing import Callable

from config import settings


def generate_rewrites(query: str, llm_client) -> list[str]:
    REWRITE_PROMPT = """Ты анализируешь поисковый запрос для интернет-магазина одежды и обуви.

ЗАДАЧА: Определи категорию товара и добавь её в начало запроса.

ПРАВИЛА:
1. Определи основную категорию товара из запроса:
   - "одежда" - если запрос про куртку, футболку, джинсы, толстовку, пальто, штаны и т.д.
   - "обувь" - если запрос про кроссовки, ботинки, сапоги, туфли, сандалии и т.д.
   - "аксессуары" - если запрос про сумку, рюкзак, кепку, шапку, очки и т.д.

2. Убери из запроса:
   - Глаголы ("посоветуй", "подбери", "хочу", "ищу")
   - Местоимения ("мне", "я")
   - Вводные слова

3. Оставь:
   - Тип товара (куртка, кроссовки и т.д.)
   - Все важные характеристики (цвет, стиль, бренд, материал)

4. Верни запрос в формате: "категория: упрощенный_запрос"

Верни ТОЛЬКО JSON:
{{"rewrite": "категория: упрощенный_запрос"}}

Запрос: "{query}"
"""

    prompt_text = REWRITE_PROMPT.format(query=query)

    try:
        content = llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "Извлекай факты из описания товара. Начинай с категории товара. Цвета описывай простым языком. Пиши только о том, что явно указано в тексте."
                },
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=400,
            temperature=0.1
        )

        # Ищем JSON в ответе
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(content)

        rewrite = data.get("rewrite", "").strip()
        if not rewrite:
            return [query]

        # Возвращаем [оригинальный запрос, rewrite с категорией]
        return [query, rewrite]
    except Exception as e:
        print(f"Warning: Failed to parse rewrite response: {e}")
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
