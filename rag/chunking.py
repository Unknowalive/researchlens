from __future__ import annotations

from typing import List, Any
from chunking import semantic_chunk


class SemanticChunker:
    """Compatibility wrapper exposing `chunk_document(text, source_id)`.

    Uses the `semantic_chunk` function from the refactor and returns a list-like of chunks.
    Each chunk will be a simple object with `.id`, `.text`, `.metadata` attributes.
    """

    def __init__(self) -> None:
        pass

    def chunk_document(self, text: str, source_id: str) -> List[Any]:
        return semantic_chunk(text, source=source_id)
