import pandas as pd

from config import settings


RAG_PROMPT = """
Выбери самые релевантные товары. Верни ТОЛЬКО список ID через запятую.

ВНИМАНИЕ:
- Синонимы считай эквивалентными: например: "худи"=="толстовка"=="свитшот", "джинсы"=="denim pants" и тд.
- Учитывай все параметры запроса: цвет, пол, бренд, категория

Запрос: {user_query}

Задача: Выбери до {max_sources} самых релевантных товаров. Верни их ID через запятую в порядке убывания релевантности.

Формат ответа (СТРОГО): S1, S3, S7

Источники:
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
        max_tokens=200,
        temperature=0.1
    )

    print(f"\n=== LLM Response ===")
    print(f"Length: {len(full_content)}")
    print(f"Content: {full_content}")
    print(f"===================\n")

    # Парсим список ID из ответа (формат: S1, S3, S7)
    cleaned = full_content.strip().replace('"', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '')
    chosen_ids = []
    for part in cleaned.split(','):
        part = part.strip()
        if part and (part.startswith('S') or part.startswith('s')):
            chosen_ids.append(part.upper())

    print(f"DEBUG: Parsed IDs: {chosen_ids}")

    chosen = [sources_by_id[sid] for sid in chosen_ids if sid in sources_by_id]

    print(f"DEBUG: Found {len(chosen)} matching sources")

    return {"chosen": chosen}
