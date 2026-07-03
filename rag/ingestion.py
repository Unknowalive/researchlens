from __future__ import annotations

from typing import List, Union
from pathlib import Path

from ingestion import PDFLayoutParser


class DocumentIngestor:
    """Compatibility wrapper exposing `extract_text(pdf_path)`.

    Internally uses `PDFLayoutParser` from the refactor.
    """

    def __init__(self) -> None:
        self._parser = PDFLayoutParser()

    def extract_text(self, pdf_path: Union[str, Path]) -> List[str]:
        return self._parser.parse(str(pdf_path))
