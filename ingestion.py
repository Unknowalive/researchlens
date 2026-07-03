from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import re
import collections

try:
    import marker_pdf  # type: ignore
    _HAS_MARKER = True
except Exception:
    _HAS_MARKER = False

import pdfplumber


@dataclass
class PageBlock:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    page_no: int


class EmptyDocumentError(ValueError):
    """Raised when a PDF contains insufficient extractable body text."""


class PDFLayoutParser:
    """Layout-aware parser for academic PDFs.

    Uses `marker_pdf` when available and falls back to `pdfplumber`.
    Drops repeated headers and footers, handles two-column text flows,
    preserves math/citation blocks, and validates extracted content.
    """

    header_footer_threshold = 0.6

    def parse(self, path: str) -> List[str]:
        if _HAS_MARKER:
            pages = self._parse_with_marker(path)
        else:
            pages = self._parse_with_pdfplumber(path)

        pages = [self._normalize_text(page) for page in pages if page.strip()]
        self._validate_document(pages)
        return pages

    def _parse_with_marker(self, path: str) -> List[str]:
        doc = marker_pdf.load(path)
        blocks_by_page: Dict[int, List[PageBlock]] = collections.defaultdict(list)

        for page_no, page in enumerate(doc.pages, start=1):
            for block in getattr(page, "blocks", []):
                text = str(getattr(block, "text", "") or "").strip()
                if not text:
                    continue
                blocks_by_page[page_no].append(
                    PageBlock(
                        text=text,
                        x0=float(getattr(block, "x0", 0.0)),
                        top=float(getattr(block, "top", 0.0)),
                        x1=float(getattr(block, "x1", 0.0)),
                        bottom=float(getattr(block, "bottom", 0.0)),
                        page_no=page_no,
                    )
                )

        return self._render_pages(blocks_by_page)

    def _parse_with_pdfplumber(self, path: str) -> List[str]:
        blocks_by_page: Dict[int, List[PageBlock]] = collections.defaultdict(list)

        with pdfplumber.open(path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(use_text_flow=True)
                if not words:
                    text = str(page.extract_text() or "").strip()
                    if text:
                        blocks_by_page[page_no].append(
                            PageBlock(
                                text=text,
                                x0=0.0,
                                top=0.0,
                                x1=float(page.width or 0.0),
                                bottom=float(page.height or 0.0),
                                page_no=page_no,
                            )
                        )
                    continue

                lines: Dict[int, List[dict]] = collections.defaultdict(list)
                for word in words:
                    top_key = int(round(float(word.get("top", 0.0))))
                    lines[top_key].append(word)

                for top_key in sorted(lines):
                    line_words = sorted(lines[top_key], key=lambda w: float(w.get("x0", 0.0)))
                    text = " ".join(str(w.get("text", "")) for w in line_words).strip()
                    if not text or self._is_page_number(text):
                        continue
                    blocks_by_page[page_no].append(
                        PageBlock(
                            text=text,
                            x0=float(line_words[0].get("x0", 0.0)),
                            top=float(line_words[0].get("top", 0.0)),
                            x1=float(line_words[-1].get("x1", 0.0)),
                            bottom=float(line_words[0].get("bottom", page.height or 0.0)),
                            page_no=page_no,
                        )
                    )

        return self._render_pages(blocks_by_page)

    def _render_pages(self, blocks_by_page: Dict[int, List[PageBlock]]) -> List[str]:
        headers, footers = self._find_repeating_headers_footers(blocks_by_page)
        pages: Dict[int, str] = {}

        for page_no, blocks in blocks_by_page.items():
            if not blocks:
                pages[page_no] = ""
                continue

            columns = self._cluster_columns(blocks)
            ordered_blocks = columns[0] + columns[1] if columns[1] else columns[0]

            filtered: List[PageBlock] = []
            for block in ordered_blocks:
                if block.text in headers or block.text in footers or self._is_page_number(block.text):
                    continue
                filtered.append(block)

            paragraphs: List[str] = []
            buffer: List[str] = []
            for block in filtered:
                text = block.text
                if buffer and buffer[-1].endswith("-"):
                    buffer[-1] = buffer[-1][:-1] + text
                else:
                    buffer.append(text)

                if self._is_paragraph_break(text):
                    joined = " ".join(buffer).strip()
                    if not self._has_unbalanced_math(joined):
                        paragraphs.append(joined)
                        buffer = []

            if buffer:
                paragraphs.append(" ".join(buffer).strip())

            pages[page_no] = "\n\n".join(paragraphs)

        return [pages[page_no] for page_no in sorted(pages)]

    def _find_repeating_headers_footers(self, blocks_by_page: Dict[int, List[PageBlock]]) -> tuple[set[str], set[str]]:
        top_texts = collections.Counter()
        bottom_texts = collections.Counter()
        page_count = len(blocks_by_page)

        for blocks in blocks_by_page.values():
            if not blocks:
                continue
            page_height = max(block.bottom for block in blocks) or 1.0
            top_zone = [block.text for block in blocks if block.top <= page_height * 0.12]
            bottom_zone = [block.text for block in blocks if block.bottom >= page_height * 0.88]
            top_texts.update(top_zone)
            bottom_texts.update(bottom_zone)

        headers = {text for text, count in top_texts.items() if count / max(1, page_count) >= self.header_footer_threshold}
        footers = {text for text, count in bottom_texts.items() if count / max(1, page_count) >= self.header_footer_threshold}
        return headers, footers

    def _cluster_columns(self, blocks: List[PageBlock]) -> Dict[int, List[PageBlock]]:
        x0_values = [block.x0 for block in blocks]
        if len(x0_values) < 8:
            return {0: sorted(blocks, key=lambda b: b.top), 1: []}

        median_x0 = sorted(x0_values)[len(x0_values) // 2]
        left = [block for block in blocks if block.x0 <= median_x0]
        right = [block for block in blocks if block.x0 > median_x0]

        if not left or not right:
            return {0: sorted(blocks, key=lambda b: b.top), 1: []}

        return {0: sorted(left, key=lambda b: b.top), 1: sorted(right, key=lambda b: b.top)}

    def _validate_document(self, pages: List[str]) -> None:
        full_text = " ".join(page for page in pages if page).strip()
        if len(full_text) < 100 or not re.search(r"[A-Za-z0-9]", full_text):
            raise EmptyDocumentError(
                f"Parsed document contains insufficient extractable text ({len(full_text)} chars)."
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _is_page_number(text: str) -> bool:
        return bool(re.match(r"^\s*\d+\s*$", text))

    @staticmethod
    def _is_paragraph_break(text: str) -> bool:
        return bool(text.endswith(".") or text.endswith("?") or text.endswith("!") or text.endswith(":") or text.endswith(";"))

    @staticmethod
    def _has_unbalanced_math(text: str) -> bool:
        single = text.count("$") % 2 != 0
        display = text.count("$$") % 2 != 0
        bracket = text.count("\\[") != text.count("\\]")
        return single or display or bracket


__all__ = ["PDFLayoutParser", "PageBlock", "EmptyDocumentError"]
