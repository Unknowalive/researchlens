from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Any
import numpy as np
import math
import re

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


@dataclass
class IndexedChunk:
    id: str
    text: str
    metadata: dict


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


_REFERENCE_PATTERN = re.compile(
    r"^\s*(?:\[\d+\]\s+[A-Z][a-z]+,|References|Bibliography|REFERENCES|BIBLIOGRAPHY)",
    re.MULTILINE,
)


def _is_reference_text(text: str) -> bool:
    if not text or _REFERENCE_PATTERN.search(text):
        return True
    if re.search(r"\[\d+(?:,\s*\d+)*\]\s+[A-Z][a-z]+,", text):
        return True
    return False


class BGEEmbeddingClient:
    """Minimal wrapper to call an external embedding API.

    By default this uses `openai` if available; pass `model_name` that maps to a BGE model.
    """

    def __init__(self, model_name: str = "text-embedding-gecko-001") -> None:
        if not _HAS_OPENAI:
            raise RuntimeError("openai package not installed; provide your own embedding client")
        self.model_name = model_name

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        # Batch call to OpenAI embeddings API
        resp = openai.Embedding.create(input=list(texts), model=self.model_name)
        vectors = [r["embedding"] for r in resp["data"]]
        return np.asarray(vectors, dtype=np.float32)


class Retriever:
    """In-memory, vectorized cosine-similarity retriever.

    - Index with `index(chunks, embeddings_matrix)` where `embeddings_matrix` shape is (n_chunks, d).
    - Query with `search(query_embeddings, top_k)` where query_embeddings can be a single vector or matrix.
    All similarity math is done with dense NumPy operations (no Python loops over chunks).
    """

    def __init__(self) -> None:
        self._embeddings: Optional[np.ndarray] = None  # normalized (n, d)
        self._chunks: List[IndexedChunk] = []

    def index(self, chunks: Sequence[IndexedChunk], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks length must match embeddings rows")

        filtered: List[IndexedChunk] = []
        filtered_embeddings: List[np.ndarray] = []
        for chunk, emb in zip(chunks, embeddings):
            if _is_reference_text(chunk.text):
                continue
            filtered.append(chunk)
            filtered_embeddings.append(emb)

        if not filtered:
            self._embeddings = None
            self._chunks = []
            return

        embedding_matrix = np.vstack(filtered_embeddings)
        self._embeddings = _normalize(embedding_matrix.astype(np.float32))
        self._chunks = filtered

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[Tuple[IndexedChunk, float]]]:
        """Vectorized search.

        Accepts `query_embeddings` shape (d,) or (m, d). Returns list (length m) of top_k (chunk, score).
        """
        if self._embeddings is None:
            raise RuntimeError("index embeddings first")
        emb = self._embeddings  # (n, d)

        q = np.asarray(query_embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # normalize queries
        q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)

        # similarities: (m, n) = q_norm @ emb.T
        sims = np.dot(q_norm, emb.T)

        results: List[List[Tuple[IndexedChunk, float]]] = []
        # For each query row, pick top_k by vectorized argpartition then sort the top candidates
        for row in sims:
            if top_k >= row.size:
                idx = np.argsort(-row)
            else:
                idx_part = np.argpartition(-row, top_k)[:top_k]
                idx = idx_part[np.argsort(-row[idx_part])]
            results.append([(self._chunks[i], float(row[i])) for i in idx])

        return results


__all__ = ["BGEEmbeddingClient", "Retriever", "IndexedChunk"]
