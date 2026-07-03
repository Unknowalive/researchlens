# Research Lens — Manual RAG Backend Engine

Research Lens is a backend-first Retrieval-Augmented Generation (RAG) engine focused on research PDFs. This repository implements a production-oriented, modular pipeline for:

- layout-aware PDF ingestion (two-column handling, math preservation)
- structural/semantic chunking (sentence/paragraph boundaries)
- manual dense embedding generation and fully vectorized retrieval
- generative question answering using an instruction-tuned causal LLM

---

## 🧩 Core Features (Backend)

- Layout-aware PDF parsing that removes headers/footers and handles two-column layouts while preserving mathematical formulas.
- Semantic chunking at sentence/paragraph boundaries (no fixed token chopping). Citations and math are kept intact.
- Manual embedding workflow (BGE-compatible) and a highly optimized, vectorized NumPy cosine-similarity retriever.
- Generative QA that synthesizes answers from provided context chunks using an instruction-tuned causal LLM; outputs include explicit source citations.

---

## 🏗️ System Architecture (Backend)

```
PDF -> Layout-aware ingestion -> Clean paragraphs -> Semantic chunking -> Embedding generation -> In-memory chunk embeddings

User question -> Query embedding -> Vectorized cosine search -> Top-K chunks -> Generative QA (synthesis using provided chunks) -> Answer + SOURCES
```

---

## 📂 Project Structure

```
researchlens/
├── app.py                  # (optional UI; backend-first work focuses on the modules below)
├── requirements.txt
├── README.md
├── ingestion.py            # layout-aware PDF parsing
├── chunking.py             # semantic chunking (SpaCy)
├── retrieval.py            # embeddings wrapper + vectorized retriever
├── generation.py           # transformers-based generative QA
└── uploaded_pdfs/          # example PDFs
```

---

## ⚙️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- SpaCy
- NumPy
- pdfplumber / marker-pdf / nougat (layout-aware parsing)
- BGE-compatible embedding endpoints (OpenAI-style or vendor-provided)

---

## 🧠 RAG Pipeline (Summary)

1. Document processing: layout-aware parsing, cleaning, semantic chunking.
2. Embedding generation: BGE-compatible dense vectors per chunk.
3. Retrieval: fully vectorized NumPy cosine similarity search for Top-K chunks.
4. Generative QA: instruction-tuned LLM synthesizes answers strictly from the provided chunks; SOURCES mapping appended.

---

## ▶️ Installation (backend)

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

---

## ▶️ Example (programmatic usage)

```python
from ingestion import PDFLayoutParser
from chunking import semantic_chunk
from retrieval import BGEEmbeddingClient, Retriever, IndexedChunk
from generation import Generator, SourceChunk

parser = PDFLayoutParser()
pages = parser.parse('uploaded_pdfs/example.pdf')
indexed_chunks = []
for i, page in enumerate(pages):
    for c in semantic_chunk(page, source=f'example.pdf:page:{i}'):
        indexed_chunks.append(IndexedChunk(id=c.id, text=c.text, metadata=c.metadata))

# Use a BGE client to produce embeddings, index them with Retriever, then call Generator.synthesize(question, source_chunks)
```

---

## � Recent Update: Bug Fixes and Improvements

During the latest backend update, the app was failing while processing uploaded PDFs. The root cause was that some PDF files produced no extractable text or no semantic chunks, which caused the indexing pipeline to crash.

Improvements made:

- Added robust PDF ingestion validation to detect files with no selectable text and report a clear error.
- Added a guard against empty chunk sets so the app fails gracefully if chunking produces no data.
- Improved the `VectorRetriever` compatibility wrapper to handle empty indexes without breaking.
- Strengthened Streamlit backend validation so users receive meaningful feedback instead of low-level exceptions.

These updates make the document processing flow more reliable and easier to debug for edge-case PDFs.

---

## 📌 Note

This repository now focuses on a manual, transparent RAG backend (no extractive QA). The generative QA stage is strictly instructed to synthesize answers only from provided context chunks and to include explicit SOURCES mapping.

**Attribution:** This work is an update based on the original repository at https://github.com/DeekshaChat/researchlens/ and includes modifications made locally for the Research Lens backend.
