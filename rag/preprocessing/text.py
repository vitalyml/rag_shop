import re
import string
import pandas as pd


def parse_price(s) -> int | None:
    """Парсинг цены из строки"""
    if pd.isna(s):
        return None
    s = str(s).replace('\xa0', ' ')
    s = re.sub(r'[^\d]', '', s)
    return int(s) if s else None


def build_text(row) -> str:
    """Склейка title + description для полнотекстового поиска"""
    parts = [str(row['title']).strip()]
    if row.get('description') and not pd.isna(row['description']):
        parts.append(str(row['description']).strip())
    return ' '.join(parts)


def simple_tokenize(text: str) -> list[str]:
    """Простая токенизация для BM25"""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()
