from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Any
import re
import numpy as np

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


_REFERENCE_PATTERN = re.compile(
    r"^\s*(?:\[\d+\]\s+[A-Z][a-z]+,|References|Bibliography|REFERENCES|BIBLIOGRAPHY)",
    re.MULTILINE,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def _is_reference_text(text: str) -> bool:
    if not text:
        return False
    if _REFERENCE_PATTERN.search(text):
        return True
    if re.search(r"\[\d+(?:,\s*\d+)*\]\s+[A-Z][a-z]+,", text):
        return True
    return False


class BGEEmbeddingClient:
    """Wrapper for BGE-compatible dense embeddings."""

    def __init__(self, model_name: str = "text-embedding-gecko-001") -> None:
        if not _HAS_OPENAI:
            raise RuntimeError("openai package required for embeddings")
        self.model_name = model_name

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        response = openai.Embedding.create(input=list(texts), model=self.model_name)
        vectors = [record["embedding"] for record in response["data"]]
        return np.asarray(vectors, dtype=np.float32)


class Retriever:
    """Pure NumPy cosine similarity retriever for manual embeddings."""

    def __init__(self) -> None:
        self._embeddings: Optional[np.ndarray] = None
        self._chunks: List[IndexedChunk] = []

    def index(self, chunks: Sequence[IndexedChunk], embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks length must match embeddings rows")

        mask = np.array([not _is_reference_text(chunk.text) for chunk in chunks])
        filtered_chunks = [chunk for chunk, keep in zip(chunks, mask) if keep]
        filtered_embeddings = embeddings[mask]

        if filtered_embeddings.size == 0:
            self._embeddings = None
            self._chunks = []
            return

        self._embeddings = _normalize(filtered_embeddings)
        self._chunks = filtered_chunks

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[Tuple[IndexedChunk, float]]]:
        if self._embeddings is None:
            raise RuntimeError("index embeddings first")

        q = np.asarray(query_embeddings, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)
        q_norm[np.isnan(q_norm)] = 0.0

        similarities = np.dot(q_norm, self._embeddings.T)
        if similarities.size == 0:
            return [[] for _ in range(q_norm.shape[0])]

        n_candidates = similarities.shape[1]
        k = min(top_k, n_candidates)
        topk_idx = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
        topk_sim = np.take_along_axis(similarities, topk_idx, axis=1)
        order = np.argsort(-topk_sim, axis=1)
        ordered_idx = np.take_along_axis(topk_idx, order, axis=1)

        return [
            [(self._chunks[idx], float(similarities[row_idx, idx])) for idx in ordered_idx[row_idx]]
            for row_idx in range(ordered_idx.shape[0])
        ]


__all__ = ["BGEEmbeddingClient", "Retriever", "IndexedChunk"]
