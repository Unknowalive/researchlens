from __future__ import annotations

from typing import List, Tuple, Sequence, Any
import numpy as np

from retrieval import Retriever, IndexedChunk


class VectorRetriever:
    """Compatibility wrapper exposing `index_chunks(chunks)` and `search(query, top_k)`.

    - `index_chunks` accepts an iterable of (id, text, metadata) tuples or objects.
    - `search` accepts a query string and returns a list of (chunk_tuple, score).
    """

    def __init__(self) -> None:
        self._retriever = Retriever()

    def index_chunks(self, chunks: Sequence[Tuple[str, str, dict]]) -> None:
        # Convert to IndexedChunk and compute embeddings via a small embedding function.
        # For compatibility, we'll create IndexedChunk items without embeddings and expect Retriever to handle embedding generation via an external client.
        indexed: List[IndexedChunk] = []
        for cid, text, meta in chunks:
            indexed.append(IndexedChunk(id=str(cid), text=str(text), metadata=dict(meta or {})))

        dim = 768
        if not indexed:
            self._retriever.index([], np.zeros((0, dim), dtype=np.float32))
            return

        # The original Retriever expects precomputed embeddings; for compatibility we create a deterministic vector per chunk.
        vectors = np.vstack([
            np.random.RandomState(abs(hash(c.id)) % (2**32)).randn(dim).astype(np.float32)
            for c in indexed
        ])

        self._retriever.index(indexed, vectors)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Tuple[str, str, dict], float]]:
        # In a real system, you'd embed the query with the same model as used for chunks.
        # For this wrapper, create a random query vector for demonstration and call the retriever.
        if self._retriever._embeddings is None:
            return []
        d = self._retriever._embeddings.shape[1]
        q = np.random.RandomState(abs(hash(query)) % (2**32)).randn(d).astype(np.float32)
        results = self._retriever.search(q, top_k=top_k)
        # results is [[(IndexedChunk, score), ...]]
        out = []
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        for chunk_obj, score in results:
            out.append(((chunk_obj.id, chunk_obj.text, chunk_obj.metadata), float(score)))
        return out
