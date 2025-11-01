import textwrap


def render_answer(result: dict, width: int = 100, show_rewrites: bool = False) -> str:
    def wrap(text, indent=""):
        return textwrap.fill(
            text, width=width,
            break_long_words=False,
            break_on_hyphens=False,
            initial_indent=indent,
            subsequent_indent=indent
        )

    lines = [wrap(result.get("answer_md", "—")), "", "**Источники:**"]

    chosen = result.get("chosen", [])
    if not chosen:
        return "\n".join(lines + ["_Источники не выбраны_"])

    for source in chosen:
        lines.append(wrap(f"- [{source.get('id', 'S?')}] {source.get('title', '')}"))
        url = (source.get("url") or "").strip()
        if url:
            lines.append(wrap(url, indent="  "))

    if show_rewrites and result.get("rewrites"):
        lines.extend(["", "**Перефразы:**"])
        for i, rewrite in enumerate(result["rewrites"], 1):
            lines.append(wrap(f"{i}. {rewrite}"))

    return "\n".join(lines)
