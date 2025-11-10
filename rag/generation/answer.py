import json
import re
import pandas as pd

from config import settings


RAG_PROMPT = """
Ты отвечаешь на основе переданных источников, без выдумок. Пиши на языке запроса. Верни ТОЛЬКО валидный JSON.

ВНИМАНИЕ:
- Каждая рекомендация должна соответствовать требованиям запроса ПО СМЫСЛУ, а не только по точным словам.
- Если в описании товара используется синоним или близкий термин, считай, что требование выполнено.
- Например:
  - "худи" == "толстовка" == "свитшот" == "hoodie"
  - "кроссовки" == "кеды" == "sneakers"
  - "джинсы" == "denim pants" и т.п.

Запрос: {user_query}

Задача:
1) Выбери до {max_sources} самых релевантных источников.
2) Напиши 1–4 предложений рекомендаций, ссылаясь метками [S#] в тексте.
3) Верни JSON:
{{
  "chosen_ids": ["S1","S3",...],
  "answer_md": "<короткий markdown с [S#]>"
}}

СТРОГО:
- chosen_ids = ровно те [S#], что есть в answer_md, в том же порядке.
- Не выдумывай свойства товаров, опирайся только на источники.

Источники (СТРОГО ОБЯЗАТЕЛЬНЫ):
{sources_block}
"""



def prepare_sources(df: pd.DataFrame, fused_results: list[dict], top_n: int = 20) -> list[dict]:
    sources = []
    for i, item in enumerate(fused_results[:top_n], 1):
        doc_id = int(item["pid"])
        row = df.iloc[doc_id]
        text = str(row.get("doc_text", row.get("doc_text_rag", "")))
        sources.append({
            "id": f"S{i}",
            "pid": doc_id,
            "title": str(row["title"]) if "title" in row.index else "",
            "url": str(row["url"]) if "url" in row.index else "",
            "snippet": text.replace("\n", " ").strip(),
            # Добавляем поля для отображения карточек товаров
            "image_url": str(row["image_url"]) if "image_url" in row.index and pd.notna(row["image_url"]) else "",
            "price": str(row["price"]) if "price" in row.index and pd.notna(row["price"]) else "",
            "old_price": str(row["old_price"]) if "old_price" in row.index and pd.notna(row["old_price"]) else "",
            "description": str(row["description"]) if "description" in row.index and pd.notna(row["description"]) else "",
            "brand": str(row["brand"]) if "brand" in row.index and pd.notna(row["brand"]) else ""
        })
    return sources


def format_sources(sources: list[dict]) -> str:
    """Форматирование источников для промпта"""
    blocks = []
    for s in sources:
        block = f"[{s['id']}] title: {s['title']} url: {s['url']} snippet: {s['snippet']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def extract_citations(text: str) -> list[str]:
    """Извлечение цитат из текста"""
    citations = []
    seen = set()
    for match in re.findall(r"\[S(\d+)\]", text or ""):
        source_id = f"S{match}"
        if source_id not in seen:
            seen.add(source_id)
            citations.append(source_id)
    return citations


def generate_answer(
    user_query: str,
    fused_results: list[dict],
    df: pd.DataFrame,
    llm_client,
    top_n_context: int = None,
    max_sources: int = None
) -> dict:
    top_n_context = top_n_context or settings.CONTEXT_DOCS
    max_sources = max_sources or settings.MAX_SOURCES

    sources = prepare_sources(df, fused_results, top_n_context)
    sources_by_id = {s["id"]: s for s in sources}

    prompt_text = RAG_PROMPT.format(
        user_query=user_query,
        max_sources=max_sources,
        sources_block=format_sources(sources)
    )

    full_content = llm_client.chat(
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=1000,
        temperature=0.1
    )

    print(f"\n=== LLM Response ===")
    print(f"Length: {len(full_content)}")
    print(f"Content: {full_content[:500]}")
    print(f"===================\n")

    # Парсим JSON из ответа
    try:
        json_match = re.search(r'\{.*\}', full_content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(full_content)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        print(f"Raw content: {full_content}")
        data = {"answer_md": full_content, "chosen_ids": []}

    answer = (data.get("answer_md") or "").strip()
    cited_ids = extract_citations(answer)
    chosen = [sources_by_id[sid] for sid in cited_ids if sid in sources_by_id]

    return {
        "answer_md": answer,
        "chosen": chosen
    }
