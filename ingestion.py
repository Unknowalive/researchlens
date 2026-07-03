from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Dict
import re
import collections

try:
    # Prefer a layout-aware parser if installed (marker-pdf recommended)
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


class PDFLayoutParser:
    """Layout-aware PDF parser.

    - Prefers `marker_pdf` when available (install separately).
    - Falls back to `pdfplumber` while keeping layout heuristics.
    - Detects and removes repeated headers/footers.
    - Handles two-column pages by column clustering and left->right reading order.
    - Attempts to preserve math-like blocks and LaTeX fragments.
    """

    header_footer_threshold = 0.6

    def __init__(self) -> None:
        pass

    def parse(self, path: str) -> List[str]:
        if _HAS_MARKER:
            return self._parse_with_marker(path)
        return self._parse_with_pdfplumber(path)

    def _parse_with_marker(self, path: str) -> List[str]:
        # marker_pdf API details may vary; this is a best-effort adapter.
        doc = marker_pdf.load(path)
        pages_text: List[str] = []
        blocks_by_page: Dict[int, List[PageBlock]] = collections.defaultdict(list)
        for i, page in enumerate(doc.pages, start=1):
            for block in page.blocks:
                blocks_by_page[i].append(
                    PageBlock(text=block.text.strip(), x0=block.x0, top=block.top, x1=block.x1, bottom=block.bottom, page_no=i)
                )

        cleaned = self._process_pages(blocks_by_page)
        for p in sorted(cleaned.keys()):
            pages_text.append(cleaned[p])
        return pages_text

    def _parse_with_pdfplumber(self, path: str) -> List[str]:
        blocks_by_page: Dict[int, List[PageBlock]] = collections.defaultdict(list)
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                # extract words with bounding boxes
                words = page.extract_words(use_text_flow=True)
                if not words:
                    # fallback to simple text
                    text = page.extract_text() or ""
                    blocks_by_page[i].append(PageBlock(text=text.strip(), x0=0, top=0, x1=page.width, bottom=page.height, page_no=i))
                    continue

                # group words into lines by their top coordinate
                lines: Dict[int, List[dict]] = collections.defaultdict(list)
                for w in words:
                    top_rounded = int(round(float(w.get("top", 0))))
                    lines[top_rounded].append(w)

                for top_k in sorted(lines.keys()):
                    line_words = sorted(lines[top_k], key=lambda w: float(w.get("x0", 0)))
                    text = " ".join(w.get("text", "") for w in line_words).strip()
                    if not text:
                        continue
                    x0 = float(line_words[0].get("x0", 0))
                    x1 = float(line_words[-1].get("x1", 0))
                    top = float(line_words[0].get("top", 0))
                    bottom = float(line_words[0].get("bottom", page.height))
                    blocks_by_page[i].append(PageBlock(text=text, x0=x0, top=top, x1=x1, bottom=bottom, page_no=i))

        cleaned = self._process_pages(blocks_by_page)
        pages_text = [cleaned[p] for p in sorted(cleaned.keys())]
        return pages_text

    def _process_pages(self, blocks_by_page: Dict[int, List[PageBlock]]) -> Dict[int, str]:
        # Identify repeated header/footer candidates
        top_texts = collections.Counter()
        bottom_texts = collections.Counter()
        page_count = len(blocks_by_page)

        for p, blocks in blocks_by_page.items():
            if not blocks:
                continue
            # top 10% and bottom 10% heuristics
            page_height = max(b.bottom for b in blocks) if blocks else 1
            top_zone = [b.text for b in blocks if b.top <= page_height * 0.12]
            bottom_zone = [b.text for b in blocks if b.bottom >= page_height * 0.88]
            if top_zone:
                top_texts.update(top_zone)
            if bottom_zone:
                bottom_texts.update(bottom_zone)

        headers = {t for t, c in top_texts.items() if c / max(1, page_count) >= self.header_footer_threshold}
        footers = {t for t, c in bottom_texts.items() if c / max(1, page_count) >= self.header_footer_threshold}

        cleaned_pages: Dict[int, str] = {}
        for p, blocks in blocks_by_page.items():
            if not blocks:
                cleaned_pages[p] = ""
                continue

            # detect columns: examine x0 distribution; if bimodal -> two columns
            x0s = [b.x0 for b in blocks]
            # simple heuristic: if variance of x0 is large and there are gaps, assume two columns
            columns: Dict[int, List[PageBlock]] = {0: [], 1: []}
            if len(blocks) >= 8:
                # split by median x0
                median_x = sorted(x0s)[len(x0s) // 2]
                left = [b for b in blocks if b.x0 <= median_x]
                right = [b for b in blocks if b.x0 > median_x]
                if left and right:
                    columns[0] = sorted(left, key=lambda b: b.top)
                    columns[1] = sorted(right, key=lambda b: b.top)
                else:
                    columns[0] = sorted(blocks, key=lambda b: b.top)
            else:
                columns[0] = sorted(blocks, key=lambda b: b.top)

            # Build reading order: left column top->bottom then right column
            ordered_blocks: List[PageBlock] = []
            if columns[1]:
                ordered_blocks.extend(columns[0])
                ordered_blocks.extend(columns[1])
            else:
                ordered_blocks = columns[0]

            # Drop headers/footers
            filtered_blocks: List[PageBlock] = []
            for b in ordered_blocks:
                if b.text in headers or b.text in footers:
                    continue
                filtered_blocks.append(b)

            # Merge nearby lines into paragraphs; preserve math and citations intact
            paragraphs: List[str] = []
            cur_lines: List[str] = []
            for b in filtered_blocks:
                text = b.text
                if self._is_page_separator(text):
                    continue
                # if line ends with hyphen likely a broken word — join without space
                if cur_lines and cur_lines[-1].endswith("-"):
                    cur_lines[-1] = cur_lines[-1][:-1] + text
                else:
                    cur_lines.append(text)
                # Heuristic paragraph break: empty line or explicit end-of-paragraph punctuation
                if text.endswith(".") or text.endswith(":") or text.endswith(";") or text.endswith("?") or text.endswith("!"):
                    # If this sentence contains unbalanced math markers, keep collecting
                    joined = " ".join(cur_lines).strip()
                    if not self._has_unbalanced_math(joined):
                        paragraphs.append(joined)
                        cur_lines = []

            if cur_lines:
                paragraphs.append(" ".join(cur_lines).strip())

            cleaned_pages[p] = "\n\n".join(paragraphs)

        return cleaned_pages

    @staticmethod
    def _is_page_separator(text: str) -> bool:
        return bool(re.match(r"^\s*[0-9]+\s*$", text))

    @staticmethod
    def _has_unbalanced_math(text: str) -> bool:
        # detect unmatched $ or $$ or \[ \]
        single = text.count("$") % 2 != 0
        display = (text.count("$$") % 2 != 0)
        bracket = (text.count("\\[") != text.count("\\]"))
        return single or display or bracket


__all__ = ["PDFLayoutParser", "PageBlock"]
