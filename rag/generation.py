from __future__ import annotations

from typing import List, Dict, Any

from generation import Generator, SourceChunk


class GenerativeQA:
    """Compatibility wrapper exposing `generate_answer(query, retrieved_chunks)`.

    Uses the `Generator` class under the hood. For local testing without a heavy model,
    this will fall back to a simple synthesis when the Generator cannot be loaded.
    """

    def __init__(self) -> None:
        try:
            self._gen = Generator(model_name="facebook/opt-1.3b")
        except Exception:
            self._gen = None

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        # Adapt retrieved_chunks (list of dicts with id,text,metadata) to SourceChunk
        source_chunks = [SourceChunk(id=c.get("id", ""), text=c.get("text", ""), metadata=c.get("metadata", {})) for c in retrieved_chunks]
        if self._gen:
            return self._gen.synthesize(query, source_chunks)
        # fallback simple concatenation
        joined = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(source_chunks))
        return f"[SYNTHESIS-FAKE] Based on provided context:\n\n{joined}\n\n(Real LLM not loaded in this environment)"
