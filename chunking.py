from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Iterable
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
    """Splits `text` into semantic chunks using sentence boundaries.

    - Splits on paragraph boundaries first (double newlines).
    - Uses SpaCy sentencizer to avoid breaking sentences.
    - Ensures math delimiters ($, $$, \[ \]) and citation tokens are not split across chunks.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[Chunk] = []
    chunk_id = 0

    for page_idx, para in enumerate(paragraphs):
        doc = NLP(para)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if not sentences:
            continue

        current: List[str] = []
        for sent in sentences:
            # Combine to avoid breaking math or citations
            if _MATH_INLINE_RE.search(sent) or _CITATION_RE.search(sent):
                # If current has content and adding this sentence would exceed max_sentences,
                # flush current first to keep math+citations with surrounding context.
                if current and len(current) >= max_sentences:
                    chunks.append(Chunk(id=f"chunk-{chunk_id}", text=" ".join(current), metadata={"source": source, "page": page_idx}))
                    chunk_id += 1
                    current = []
                current.append(sent)
                # flush immediately to avoid mixing citation with unrelated sentences
                chunks.append(Chunk(id=f"chunk-{chunk_id}", text=" ".join(current), metadata={"source": source, "page": page_idx}))
                chunk_id += 1
                current = []
                continue

            current.append(sent)
            # If we've reached size limit, flush
            if len(current) >= max_sentences:
                chunks.append(Chunk(id=f"chunk-{chunk_id}", text=" ".join(current), metadata={"source": source, "page": page_idx}))
                chunk_id += 1
                current = []

        if current:
            chunks.append(Chunk(id=f"chunk-{chunk_id}", text=" ".join(current), metadata={"source": source, "page": page_idx}))
            chunk_id += 1

    return chunks


__all__ = ["semantic_chunk", "Chunk"]
