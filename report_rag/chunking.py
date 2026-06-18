from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Sequence

import numpy as np

from .schemas import Chunk, Element
from .text_utils import token_count


class ChunkBuilder:
    def __init__(
        self,
        chunk_tokens: int = 512,
        overlap_tokens: int = 100,
        min_chunk_tokens: int = 30,
    ) -> None:
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def build(self, elements: Sequence[Element]) -> list[Chunk]:
        by_document: dict[str, list[Element]] = defaultdict(list)
        for element in elements:
            by_document[element.doc_id].append(element)

        chunks: list[Chunk] = []
        for doc_elements in by_document.values():
            chunks.extend(self._build_document(doc_elements))
        return chunks

    def _build_document(self, elements: Sequence[Element]) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[Element] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            text = "\n".join(item.text for item in buffer).strip()
            current_tokens = token_count(text)
            if text and current_tokens >= self.min_chunk_tokens:
                chunks.append(self._make_chunk(buffer, text))
            elif text and chunks and chunks[-1].element_type != "table":
                previous = chunks[-1]
                previous.text = f"{previous.text}\n{text}"
                previous.pages = sorted({*previous.pages, *(item.page for item in buffer)})
                previous.bboxes.extend(
                    (item.page, *[round(value, 2) for value in item.bbox])
                    for item in buffer
                )
            overlap = self._tail_overlap(buffer)
            buffer = overlap
            buffer_tokens = sum(token_count(item.text) for item in buffer)

        for element in elements:
            if element.element_type == "table":
                flush()
                chunks.append(self._make_chunk([element], element.text))
                buffer = []
                buffer_tokens = 0
                continue

            element_tokens = token_count(element.text)
            starts_section = self._looks_like_title(element, elements)
            if buffer and (
                buffer_tokens + element_tokens > self.chunk_tokens
                or (starts_section and buffer_tokens >= self.min_chunk_tokens)
            ):
                flush()
            buffer.append(element)
            buffer_tokens += element_tokens
        flush()
        return chunks

    def _tail_overlap(self, elements: Sequence[Element]) -> list[Element]:
        if self.overlap_tokens <= 0:
            return []
        selected: list[Element] = []
        total = 0
        for element in reversed(elements):
            selected.append(element)
            total += token_count(element.text)
            if total >= self.overlap_tokens:
                break
        return list(reversed(selected))

    @staticmethod
    def _looks_like_title(element: Element, all_elements: Sequence[Element]) -> bool:
        sizes = [item.font_size for item in all_elements if item.font_size > 0]
        median_size = float(np.median(sizes)) if sizes else 0.0
        numbered = bool(
            re.match(
                r"^(?:第[一二三四五六七八九十百]+[章节]|"
                r"\d+(?:\.\d+){0,3}[、.\s]|"
                r"[一二三四五六七八九十]+、)",
                element.text,
            )
        )
        return (
            numbered
            or (
                median_size > 0
                and element.font_size >= median_size * 1.22
                and len(element.text) <= 100
            )
            or (
                median_size > 0
                and element.bold
                and element.font_size >= median_size * 1.08
                and 4 <= token_count(element.text) <= 45
            )
        )

    @staticmethod
    def _make_chunk(elements: Sequence[Element], text: str) -> Chunk:
        first = elements[0]
        chunk_id = hashlib.sha1(
            f"{first.doc_id}:{first.page}:{text}".encode("utf-8")
        ).hexdigest()[:20]
        element_types = {item.element_type for item in elements}
        return Chunk(
            chunk_id=chunk_id,
            doc_id=first.doc_id,
            source=first.source,
            pages=sorted({item.page for item in elements}),
            element_type=next(iter(element_types)) if len(element_types) == 1 else "mixed",
            text=text,
            bboxes=[
                (item.page, *[round(value, 2) for value in item.bbox])
                for item in elements
            ],
        )

