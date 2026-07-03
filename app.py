from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Tuple

import streamlit as st

from rag.ingestion import DocumentIngestor
from rag.chunking import SemanticChunker
from rag.retrieval import VectorRetriever
from rag.generation import GenerativeQA


TOP_K_DEFAULT = 5


def _save_uploaded_file(uploaded_file: Any) -> str:
    tmpdir = tempfile.mkdtemp(prefix="researchlens_")
    file_path = os.path.join(tmpdir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def _chunk_to_tuple(chunk: Any) -> Tuple[str, str, Dict[str, Any]]:
    if chunk is None:
        return ("", "", {})
    if isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
        cid = str(chunk[0])
        text = str(chunk[1])
        meta = dict(chunk[2]) if len(chunk) >= 3 and chunk[2] is not None else {}
        return cid, text, meta
    if isinstance(chunk, dict):
        cid = str(chunk.get("id") or "")
        text = str(chunk.get("text") or "")
        meta = dict(chunk.get("metadata") or {})
        return cid, text, meta
    cid = getattr(chunk, "id", "") or getattr(chunk, "chunk_id", "")
    text = getattr(chunk, "text", "") or getattr(chunk, "content", "")
    meta = getattr(chunk, "metadata", None) or getattr(chunk, "meta", None) or {}
    return str(cid), str(text), dict(meta)


@st.cache_resource
def load_retriever() -> VectorRetriever:
    return VectorRetriever()


@st.cache_resource
def load_generator() -> GenerativeQA:
    return GenerativeQA()


def ensure_session_state() -> None:
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "source_file" not in st.session_state:
        st.session_state.source_file = ""


def process_document(pdf_path: str, ingestor: DocumentIngestor, chunker: SemanticChunker, retriever: VectorRetriever) -> None:
    raw = ingestor.extract_text(pdf_path)
    if isinstance(raw, (list, tuple)):
        full_text = "\n\n".join(p for p in raw if p)
    else:
        full_text = str(raw or "")

    if not full_text.strip():
        raise ValueError("No extractable text was found in the PDF. Ensure the file contains selectable text rather than scanned images.")

    chunks = chunker.chunk_document(full_text, source_id=os.path.basename(pdf_path))

    normalized: List[Tuple[str, str, Dict[str, Any]]] = []
    for c in chunks:
        normalized.append(_chunk_to_tuple(c))

    if not normalized:
        raise ValueError("No semantic chunks were generated from the document. Please confirm that the PDF contains extractable text.")

    retriever.index_chunks(normalized)

    st.session_state.chunks = normalized
    st.session_state.indexed = True
    st.session_state.source_file = os.path.basename(pdf_path)


def run_query(query: str, retriever: VectorRetriever, generator: GenerativeQA, top_k: int = TOP_K_DEFAULT):
    retrieved = retriever.search(query, top_k=top_k)

    normalized_results = []
    if isinstance(retrieved, list) and retrieved and isinstance(retrieved[0], list):
        retrieved_list = retrieved[0]
    else:
        retrieved_list = retrieved

    for item in retrieved_list:
        if isinstance(item, tuple) and len(item) == 2:
            chunk_obj, score = item
        elif isinstance(item, dict):
            chunk_obj = item.get("chunk") or item
            score = float(item.get("score", 0.0))
        else:
            chunk_obj = getattr(item, "chunk", item)
            score = float(getattr(item, "score", 0.0))
        cid, text, meta = _chunk_to_tuple(chunk_obj)
        normalized_results.append(((cid, text, meta), float(score)))

    source_chunks = [{"id": c[0], "text": c[1], "metadata": c[2]} for (c, _) in normalized_results]

    answer = generator.generate_answer(query, source_chunks)
    return answer, normalized_results


def main() -> None:
    st.set_page_config(page_title="Research Lens — Manual RAG", layout="wide")
    ensure_session_state()

    st.sidebar.title("Research Lens — Document")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    process_btn = st.sidebar.button("Process Document")

    retriever = load_retriever()
    generator = load_generator()

    ingestor = DocumentIngestor()
    chunker = SemanticChunker()

    if process_btn:
        if not uploaded_file:
            st.sidebar.error("Please upload a PDF before processing.")
        else:
            with st.sidebar:
                st.info("Saving uploaded file and processing. This may take a moment.")
            temp_path = _save_uploaded_file(uploaded_file)
            st.sidebar.write(f"Saved to: {temp_path}")

            progress = st.sidebar.progress(0)
            with st.sidebar.spinner("Ingesting PDF (layout-aware parsing)..."):
                try:
                    progress.progress(10)
                    process_document(temp_path, ingestor, chunker, retriever)
                    progress.progress(100)
                    st.sidebar.success(f"Document processed & indexed: {os.path.basename(temp_path)}")
                except Exception as e:
                    st.sidebar.error(f"Failed to process document: {e}")
                    st.session_state.indexed = False

    st.title("Research Lens — Manual RAG (Backend Demo)")

    if st.session_state.indexed:
        st.info(f"Indexed: {st.session_state.source_file} — {len(st.session_state.chunks)} chunks")

    query = st.text_input("Enter your question", value="", key="query_input")
    ask_btn = st.button("Ask")

    if ask_btn:
        if not query.strip():
            st.error("Please type a question.")
        elif not st.session_state.indexed:
            st.error("Please process a document first (use the sidebar).")
        else:
            with st.spinner("Retrieving relevant chunks..."):
                try:
                    answer, retrieved = run_query(query, retriever, generator, top_k=TOP_K_DEFAULT)
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    answer, retrieved = "Error during query", []

            st.header("Answer")
            st.success(answer)

            with st.expander("Retrieved chunks and similarity scores (audit)"):
                if not retrieved:
                    st.write("No retrieved chunks to display.")
                else:
                    for idx, ((cid, text, meta), score) in enumerate(retrieved, start=1):
                        st.markdown(f"**[{idx}] Chunk ID:** {cid} — **Score:** {score:.4f}")
                        st.write(text)
                        if meta:
                            st.markdown(f"**Metadata:** {meta}")
                        st.divider()

    st.markdown("---")
    st.caption("Backend modules: rag.ingestion.DocumentIngestor, rag.chunking.SemanticChunker, rag.retrieval.VectorRetriever, rag.generation.GenerativeQA")


if __name__ == "__main__":
    main()
