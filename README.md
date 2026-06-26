# 📄 RAG Pipeline — PDF Question Answering

A **Retrieval-Augmented Generation (RAG)** pipeline that ingests PDF documents, chunks and embeds them into a ChromaDB vector store, and answers natural-language questions using Groq's LLM with source citations.

## Architecture

```
PDF Files ──► Document Loader ──► Text Splitter ──► Embeddings ──► ChromaDB
                (PyPDF)        (Recursive Char)   (MiniLM-L6)    (Vector Store)
                                                                       │
User Query ──► Query Embedding ──► Similarity Search ──► Context ──► Groq LLM ──► Answer + Citations
```

## Features

- **PDF Ingestion** — Recursively loads all PDFs from a directory using LangChain's `PyPDFLoader`
- **Smart Chunking** — Splits documents with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
- **Sentence Embeddings** — Generates 384-dim vectors via `all-MiniLM-L6-v2` (SentenceTransformers)
- **Persistent Vector Store** — Stores embeddings in ChromaDB with full metadata (source, page, author, etc.)
- **Semantic Retrieval** — Cosine-similarity search with configurable top-k and score thresholds
- **LLM Generation** — Answers queries using Groq (`llama-3.1-8b-instant`) with context from retrieved chunks
- **Advanced Pipeline** — Includes streaming output, source citations, answer summarization, and query history

## Project Structure

```
RAG/
├── data/
│   ├── pdf/                  # Place your PDF files here
│   └── text_files/           # Sample text files (auto-generated)
├── notebook/
│   ├── document.ipynb        # Data ingestion basics (TextLoader, DirectoryLoader)
│   └── pdf_loader.ipynb      # Full RAG pipeline (load → chunk → embed → retrieve → generate)
├── main.py                   # Entry point (placeholder)
├── pyproject.toml            # Project config & dependencies
├── requirements.txt          # Pip requirements
└── .env                      # API keys (not committed)
```

## Setup

### Prerequisites

- Python 3.14+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
# Clone the repo
git clone https://github.com/abhip161/RAG.git
cd RAG

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. **Add PDFs** — Drop your PDF files into the `data/pdf/` directory.

2. **Run the notebook** — Open `notebook/pdf_loader.ipynb` and run all cells sequentially:

   | Step | What it does |
   |------|-------------|
   | **Load** | Reads all PDFs from `data/pdf/` using `PyPDFLoader` |
   | **Chunk** | Splits pages into ~1000-char chunks with 200-char overlap |
   | **Embed** | Generates 384-dim embeddings with `all-MiniLM-L6-v2` |
   | **Store** | Persists embeddings + metadata in ChromaDB |
   | **Query** | Retrieves relevant chunks and generates an answer via Groq |

3. **Ask questions** — Use any of the three query interfaces:

   ```python
   # Simple RAG
   answer = rag_simple("What is Multi-Head Attention?", rag_retriever, llm)

   # Advanced RAG (with sources & confidence)
   result = rag_advanced("What is the Transformer?", rag_retriever, llm,
                         top_k=3, min_score=0.1, return_context=True)

   # Full pipeline (streaming, citations, summarization, history)
   pipeline = AdvancedRAGPipeline(rag_retriever, llm)
   result = pipeline.query("Explain positional encoding",
                           stream=True, summarize=True)
   ```

## Key Components

| Class / Function | Description |
|---|---|
| `process_all_pdfs()` | Recursively loads all PDFs from a directory |
| `split_documents()` | Chunks documents using `RecursiveCharacterTextSplitter` |
| `EmbeddingManager` | Wraps `SentenceTransformer` for embedding generation |
| `VectorStore` | Manages ChromaDB collection (add, persist, query) |
| `RAGRetriever` | Encodes queries and performs similarity search |
| `rag_simple()` | Minimal retrieve-then-generate pipeline |
| `rag_advanced()` | Adds source tracking, confidence scores, and context return |
| `AdvancedRAGPipeline` | Full-featured pipeline with streaming, citations, summarization, and history |

## Tech Stack

| Category | Technology |
|---|---|
| Document Loading | LangChain (`PyPDFLoader`, `TextLoader`, `DirectoryLoader`) |
| Text Splitting | LangChain (`RecursiveCharacterTextSplitter`) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (persistent) |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Package Management | uv / pip |

## License

This project is for educational and personal use.
