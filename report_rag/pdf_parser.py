from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .schemas import Element
from .text_utils import (
    PAGE_NUMBER_PATTERN,
    normalize_text,
    normalized_repeated_text,
    overlap_ratio,
)


class PdfLayoutParser:
    def __init__(
        self,
        header_ratio: float = 0.08,
        footer_ratio: float = 0.08,
        repeated_page_ratio: float = 0.35,
    ) -> None:
        self.header_ratio = header_ratio
        self.footer_ratio = footer_ratio
        self.repeated_page_ratio = repeated_page_ratio

    def parse_directory(self, pdf_dir: Path) -> list[Element]:
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

        elements: list[Element] = []
        for index, pdf_path in enumerate(pdf_files, 1):
            print(f"[parse {index}/{len(pdf_files)}] {pdf_path.name}", flush=True)
            elements.extend(self.parse_pdf(pdf_path))
        return elements

    def parse_pdf(self, pdf_path: Path) -> list[Element]:
        try:
            import fitz
            import pdfplumber
        except ImportError as exc:
            raise RuntimeError(
                "Missing PDF dependency. Install PyMuPDF and pdfplumber first."
            ) from exc

        binary = pdf_path.read_bytes()
        doc_id = hashlib.sha1(binary).hexdigest()[:16]
        document = fitz.open(stream=binary, filetype="pdf")
        page_elements: list[list[Element]] = []

        try:
            with pdfplumber.open(pdf_path) as plumber:
                page_total = min(len(document), len(plumber.pages))
                for page_index in range(page_total):
                    fitz_page = document[page_index]
                    plumber_page = plumber.pages[page_index]
                    tables, table_boxes = self._extract_tables(
                        fitz_page, plumber_page, doc_id, pdf_path.name, page_index
                    )
                    lines = self._extract_text_lines(
                        fitz_page,
                        doc_id,
                        pdf_path.name,
                        page_index,
                        table_boxes,
                    )
                    page_elements.append(lines + tables)
        finally:
            document.close()

        self._remove_repeated_headers_footers(page_elements)
        ordered: list[Element] = []
        for elements in page_elements:
            ordered.extend(self._assign_columns_and_sort(elements))
        return ordered

    def _extract_text_lines(
        self,
        page,
        doc_id: str,
        source: str,
        page_index: int,
        table_boxes: Sequence[tuple[float, float, float, float]],
    ) -> list[Element]:
        page_dict = page.get_text("dict", sort=False)
        lines: list[Element] = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [
                    span for span in line.get("spans", [])
                    if span.get("text", "").strip()
                ]
                if not spans:
                    continue
                text = normalize_text("".join(span.get("text", "") for span in spans))
                if not text:
                    continue
                bbox = (
                    min(float(span["bbox"][0]) for span in spans),
                    min(float(span["bbox"][1]) for span in spans),
                    max(float(span["bbox"][2]) for span in spans),
                    max(float(span["bbox"][3]) for span in spans),
                )
                if any(overlap_ratio(bbox, table_box) >= 0.45 for table_box in table_boxes):
                    continue
                sizes = [float(span.get("size", 0.0)) for span in spans]
                flags = [int(span.get("flags", 0)) for span in spans]
                fonts = [str(span.get("font", "")).lower() for span in spans]
                bold = any(flag & 16 for flag in flags) or any("bold" in font for font in fonts)
                lines.append(
                    Element(
                        doc_id=doc_id,
                        source=source,
                        page=page_index + 1,
                        element_type="text",
                        text=text,
                        bbox=bbox,
                        font_size=max(sizes, default=0.0),
                        bold=bold,
                        page_width=float(page.rect.width),
                    )
                )
        return lines

    def _extract_tables(
        self,
        fitz_page,
        plumber_page,
        doc_id: str,
        source: str,
        page_index: int,
    ) -> tuple[list[Element], list[tuple[float, float, float, float]]]:
        settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 5,
            "text_tolerance": 3,
        }
        try:
            found = plumber_page.find_tables(table_settings=settings)
        except Exception as exc:
            print(f"  warning: table detection failed on page {page_index + 1}: {exc}")
            found = []

        elements: list[Element] = []
        boxes: list[tuple[float, float, float, float]] = []
        for table in found:
            bbox = tuple(float(value) for value in table.bbox)
            rows = table.extract() or []
            rendered_rows = []
            for row in rows:
                cells = [normalize_text(cell or "") for cell in row]
                if any(cells):
                    rendered_rows.append(" | ".join(cells))
            text = "\n".join(rendered_rows).strip()
            if not text:
                continue
            boxes.append(bbox)
            elements.append(
                Element(
                    doc_id=doc_id,
                    source=source,
                    page=page_index + 1,
                    element_type="table",
                    text=text,
                    bbox=bbox,
                    page_width=float(fitz_page.rect.width),
                )
            )
        return elements, boxes

    def _remove_repeated_headers_footers(
        self, page_elements: list[list[Element]]
    ) -> None:
        page_count = len(page_elements)
        if page_count < 3:
            return

        occurrences: dict[str, set[int]] = defaultdict(set)
        for page_index, elements in enumerate(page_elements):
            page_height = max((element.bbox[3] for element in elements), default=1.0)
            for element in elements:
                if element.element_type != "text":
                    continue
                is_margin = (
                    element.bbox[1] <= page_height * self.header_ratio
                    or element.bbox[3] >= page_height * (1 - self.footer_ratio)
                )
                if is_margin:
                    key = normalized_repeated_text(element.text)
                    if key:
                        occurrences[key].add(page_index)

        minimum_pages = max(3, math.ceil(page_count * self.repeated_page_ratio))
        repeated = {
            text for text, pages in occurrences.items() if len(pages) >= minimum_pages
        }
        for page_index, elements in enumerate(page_elements):
            page_elements[page_index] = [
                element
                for element in elements
                if not (
                    element.element_type == "text"
                    and (
                        normalized_repeated_text(element.text) in repeated
                        or PAGE_NUMBER_PATTERN.fullmatch(element.text)
                    )
                )
            ]

    def _assign_columns_and_sort(self, elements: list[Element]) -> list[Element]:
        text_elements = [item for item in elements if item.element_type == "text"]
        if not text_elements:
            return sorted(elements, key=lambda item: (item.bbox[1], item.bbox[0]))

        page_width = max(
            (item.page_width for item in elements if item.page_width > 0),
            default=max(item.bbox[2] for item in text_elements),
        )
        full_width: list[Element] = []
        column_candidates: list[Element] = []

        for element in elements:
            width = element.bbox[2] - element.bbox[0]
            if element.element_type != "text" or width >= page_width * 0.78:
                element.column = -1
                full_width.append(element)
            else:
                column_candidates.append(element)

        left_edges = sorted(item.bbox[0] for item in column_candidates)
        split = None
        if len(left_edges) >= 6:
            gaps = [
                (left_edges[i + 1] - left_edges[i], i)
                for i in range(len(left_edges) - 1)
            ]
            largest_gap, gap_index = max(gaps, default=(0.0, 0))
            candidate_split = (left_edges[gap_index] + left_edges[gap_index + 1]) / 2
            left_count = sum(edge < candidate_split for edge in left_edges)
            right_count = len(left_edges) - left_count
            if (
                largest_gap >= page_width * 0.10
                and left_count >= 3
                and right_count >= 3
            ):
                split = candidate_split

        for element in column_candidates:
            element.column = 0 if split is None or element.bbox[0] < split else 1

        if split is None:
            return sorted(elements, key=lambda item: (item.bbox[1], item.bbox[0]))

        columns = {
            column: [item for item in column_candidates if item.column == column]
            for column in (0, 1)
        }
        column_chars = {
            column: sum(len(item.text) for item in items)
            for column, items in columns.items()
        }
        column_widths = {
            column: (
                max((item.bbox[2] for item in items), default=0)
                - min((item.bbox[0] for item in items), default=0)
            )
            for column, items in columns.items()
        }
        smaller = min(column_chars, key=column_chars.get)
        larger = 1 - smaller
        is_sidebar = (
            column_chars[larger] > 0
            and column_chars[smaller] < column_chars[larger] * 0.50
            and column_widths[smaller] < page_width * 0.36
        )
        if is_sidebar:
            return sorted(
                [*full_width, *columns[larger]],
                key=lambda item: (item.bbox[1], item.bbox[0]),
            )

        anchors = sorted(full_width, key=lambda item: (item.bbox[1], item.bbox[0]))
        result: list[Element] = []
        previous_bottom = -math.inf
        for anchor in anchors:
            band = [
                item
                for item in column_candidates
                if previous_bottom <= item.bbox[1] < anchor.bbox[1]
            ]
            result.extend(
                sorted(band, key=lambda item: (item.column, item.bbox[1], item.bbox[0]))
            )
            result.append(anchor)
            previous_bottom = max(previous_bottom, anchor.bbox[3])
        tail = [item for item in column_candidates if item.bbox[1] >= previous_bottom]
        result.extend(
            sorted(tail, key=lambda item: (item.column, item.bbox[1], item.bbox[0]))
        )
        return result

