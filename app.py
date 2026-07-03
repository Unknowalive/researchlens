from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from chunking import Chunk, semantic_chunk
from generation import Generator, SourceChunk
from ingestion import EmptyDocumentError, PDFLayoutParser
from retrieval import BGEEmbeddingClient, IndexedChunk, Retriever


DEFAULT_EMBED_MODEL = "text-embedding-gecko-001"
DEFAULT_LLM_MODEL = "facebook/opt-1.3b"
DEFAULT_TOP_K = 5


class ResearchLensEngine:
    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.parser = PDFLayoutParser()
        self.embedding_client = BGEEmbeddingClient(model_name=embed_model)
        self.retriever = Retriever()
        self.generator = Generator(model_name=llm_model, device=device)
        self.chunks: List[IndexedChunk] = []

    def ingest_pdf(self, pdf_path: str, source_id: Optional[str] = None, max_sentences: int = 8) -> None:
        pages = self.parser.parse(pdf_path)
        source_id = source_id or Path(pdf_path).name

        document_chunks: List[Chunk] = []
        for page_text in pages:
            document_chunks.extend(
                semantic_chunk(page_text, source=source_id, max_sentences=max_sentences)
            )

        if not document_chunks:
            raise ValueError("Document parsed successfully but no semantic chunks were created.")

        self.chunks = [IndexedChunk(id=chunk.id, text=chunk.text, metadata=chunk.metadata) for chunk in document_chunks]
        embeddings = self.embedding_client.embed_texts([chunk.text for chunk in self.chunks])
        self.retriever.index(self.chunks, embeddings)

    def answer_query(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, object]:
        if self.retriever._embeddings is None:
            raise RuntimeError("Document must be indexed before querying.")

        query_embedding = self.embedding_client.embed_texts([query])
        results = self.retriever.search(query_embedding, top_k=top_k)[0]

        source_chunks = [SourceChunk(id=chunk.id, text=chunk.text, metadata=chunk.metadata) for chunk, _ in results]
        answer = self.generator.synthesize(query, source_chunks)

        return {
            "answer": answer,
            "results": [
                {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata, "score": score}
                for chunk, score in results
            ],
        }

    def save_index(self, path: str) -> None:
        if self.retriever._embeddings is None:
            raise RuntimeError("No index available to save.")

        data = {
            "chunks": [
                {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
                for chunk in self.chunks
            ]
        }
        np.savez_compressed(path, embeddings=self.retriever._embeddings, chunks=json.dumps(data))

    def load_index(self, path: str) -> None:
        archive = np.load(path, allow_pickle=True)
        embeddings = archive["embeddings"]
        chunks_json = archive["chunks"].item() if hasattr(archive["chunks"], "item") else archive["chunks"]
        chunks_data = json.loads(str(chunks_json))
        self.chunks = [IndexedChunk(**chunk) for chunk in chunks_data["chunks"]]
        self.retriever.index(self.chunks, embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research Lens backend CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse and index a PDF document")
    ingest.add_argument("pdf_path", type=str, help="Path to the PDF file")
    ingest.add_argument("--index-path", type=str, help="Optional output path for the saved index (.npz file)")
    ingest.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    ingest.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, help="LLM model name")
    ingest.add_argument("--device", type=str, default=None, help="Device for the text generation model")

    query = subparsers.add_parser("query", help="Run a question against an indexed PDF")
    query.add_argument("pdf_path", type=str, help="Path to the PDF file")
    query.add_argument("query", type=str, help="Question to ask")
    query.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    query.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="Embedding model name")
    query.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, help="LLM model name")
    query.add_argument("--device", type=str, default=None, help="Device for the text generation model")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = ResearchLensEngine(embed_model=args.embed_model, llm_model=args.llm_model, device=getattr(args, "device", None))

    if args.command == "ingest":
        try:
            engine.ingest_pdf(args.pdf_path)
            print(f"Indexed {len(engine.chunks)} chunks from {args.pdf_path}")
            if args.index_path:
                engine.save_index(args.index_path)
                print(f"Saved index to {args.index_path}")
        except EmptyDocumentError as exc:
            raise SystemExit(f"Document ingestion failed: {exc}") from exc
        except Exception as exc:
            raise SystemExit(f"Ingestion error: {exc}") from exc
    elif args.command == "query":
        try:
            engine.ingest_pdf(args.pdf_path)
            result = engine.answer_query(args.query, top_k=args.top_k)
            print("ANSWER:\n", result["answer"])
            print("\nRETRIEVED CHUNKS:")
            for idx, item in enumerate(result["results"], start=1):
                print(f"[{idx}] id={item['id']} score={item['score']:.4f}")
                print(item["text"])
                print("METADATA:", item["metadata"])
                print("---")
        except EmptyDocumentError as exc:
            raise SystemExit(f"Document ingestion failed: {exc}") from exc
        except Exception as exc:
            raise SystemExit(f"Query error: {exc}") from exc


if __name__ == "__main__":
    main()
