from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import re

import spacy


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]


def _init_nlp():
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])  # type: ignore
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    return nlp


NLP = _init_nlp()


_MATH_INLINE_RE = re.compile(r"(?s)\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$")
_CITATION_RE = re.compile(r"\[[0-9,\-\s]+\]|\([A-Z][a-zA-Z]+ et al\.,? \d{4}\)|\\cite\{.*?\}")


def semantic_chunk(text: str, source: str = "", max_sentences: int = 8) -> List[Chunk]:
    normalized_text = re.sub(r"\s+", " ", text or "").strip()
    paragraphs = [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", normalized_text) if p.strip()]

    chunks: List[Chunk] = []
    chunk_id = 0

    for page_idx, paragraph in enumerate(paragraphs):
        doc = NLP(paragraph)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if not sentences:
            continue

        current: List[str] = []
        for sentence in sentences:
            if _MATH_INLINE_RE.search(sentence) or _CITATION_RE.search(sentence):
                if current:
                    chunks.append(
                        Chunk(
                            id=f"chunk-{chunk_id}",
                            text=" ".join(current),
                            metadata={"source": source, "page": page_idx},
                        )
                    )
                    chunk_id += 1
                    current = []
                chunks.append(
                    Chunk(
                        id=f"chunk-{chunk_id}",
                        text=sentence,
                        metadata={"source": source, "page": page_idx},
                    )
                )
                chunk_id += 1
                continue

            current.append(sentence)
            if len(current) >= max_sentences:
                chunks.append(
                    Chunk(
                        id=f"chunk-{chunk_id}",
                        text=" ".join(current),
                        metadata={"source": source, "page": page_idx},
                    )
                )
                chunk_id += 1
                current = []

        if current:
            chunks.append(
                Chunk(
                    id=f"chunk-{chunk_id}",
                    text=" ".join(current),
                    metadata={"source": source, "page": page_idx},
                )
            )
            chunk_id += 1

    return chunks


__all__ = ["semantic_chunk", "Chunk"]
