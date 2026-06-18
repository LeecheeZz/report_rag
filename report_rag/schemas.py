from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Element:
    doc_id: str
    source: str
    page: int
    element_type: str
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float = 0.0
    bold: bool = False
    column: int = 0
    page_width: float = 0.0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    pages: list[int]
    element_type: str
    text: str
    bboxes: list[tuple[int, float, float, float, float]]

