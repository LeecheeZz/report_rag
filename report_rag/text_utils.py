from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")
SPACE_PATTERN = re.compile(r"\s+")
PAGE_NUMBER_PATTERN = re.compile(r"^\s*[-—–]?\s*(?:第\s*)?\d+\s*(?:页)?\s*[-—–]?\s*$")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = text.replace("\uf06e", "•").replace("\uf0b7", "•")
    return SPACE_PATTERN.sub(" ", text).strip()


def normalized_repeated_text(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"[\W_]+", "", text)


def token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def lexical_tokens(text: str) -> list[str]:
    try:
        import jieba

        return [token.strip().lower() for token in jieba.cut(text) if token.strip()]
    except ImportError:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = max((first[2] - first[0]) * (first[3] - first[1]), 1.0)
    return intersection / first_area

