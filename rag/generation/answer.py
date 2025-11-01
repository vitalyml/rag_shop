import json
import re
import pandas as pd

from config import settings


RAG_PROMPT = """
Ты отвечаешь на основе переданных источников, без выдумок. Пиши на языке запроса. Верни ТОЛЬКО валидный JSON.
Каждая из рекомендаций должна СТРОГО(!) соответствовать ВСЕМ требованиям в запросе.
Если источник частично релевантен, обясни, почему ты советуешь его.
Если ни один источник не подходит, верни пустой список chosen_ids и пустой answer_md.

Запрос: {user_query}

Задача:
1) Выбери до {max_sources} самых релевантных источников.
2) Напиши 1–3 предложений рекомендаций, ссылаясь метками [S#] в тексте.
3) Верни JSON:
{{
  "chosen_ids": ["S1","S3",...],
  "answer_md": "<короткий markdown с [S#]>"
}}
СТРОГО: chosen_ids = ровно те [S#], что есть в answer_md, в том же порядке.

Источники (СТРОГО ОБЯЗАТЕЛЬНЫ):
{sources_block}
"""


def prepare_sources(df: pd.DataFrame, fused_results: list[dict], top_n: int = 20) -> list[dict]:
    sources = []
    for i, item in enumerate(fused_results[:top_n], 1):
        doc_id = int(item["pid"])
        text = str(df.at[doc_id, "doc_text"])
        sources.append({
            "id": f"S{i}",
            "pid": doc_id,
            "title": str(df.at[doc_id, "title"]),
            "url": str(df.at[doc_id, "url"]),
            "snippet": text.replace("\n", " ").strip()
        })
    return sources


def format_sources(sources: list[dict]) -> str:
    """Форматирование источников для промпта"""
    blocks = []
    for s in sources:
        block = f"[{s['id']}]\ntitle: {s['title']}\nurl: {s['url']}\nsnippet: {s['snippet']}"
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
    max_sources: int = None,
    rewrites: list[str] = None
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
        max_tokens=600,
        temperature=0.1
    )

    # Парсим JSON из ответа
    try:
        json_match = re.search(r'\{.*\}', full_content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(full_content)
    except json.JSONDecodeError:
        data = {"answer_md": full_content, "chosen_ids": []}

    answer = (data.get("answer_md") or "").strip()
    cited_ids = extract_citations(answer)
    chosen = [sources_by_id[sid] for sid in cited_ids if sid in sources_by_id]

    return {
        "answer_md": answer,
        "chosen": chosen,
        "rewrites": rewrites or []
    }
