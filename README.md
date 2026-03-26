# Agentic RAG for Legal Documents

A retrieval-augmented generation system for querying Alaska and Hawaii insurance regulations. Uses an agentic pipeline with self-reflection, hybrid retrieval (dense + BM25 + cross-encoder reranking), and a local LLM to answer questions with citations grounded in the source documents.

## What it does

- Parses 75+ legal PDFs across 3 document types (Alaska Admin Code, Hawaii Admin Code, Hawaii Revised Statutes)
- Chunks them hierarchically respecting section boundaries (subsections -> paragraphs -> sentences)
- Builds both dense (ChromaDB + BGE embeddings) and sparse (BM25) indexes
- Retrieves using hybrid search with reciprocal rank fusion and cross-encoder reranking
- Answers questions using Qwen 3 8B locally via Ollama
- Self-reflects: if the initial answer is weak, it reformulates the query and retries
- Every claim in the answer is backed by a `[Source: section_id, file]` citation

## Architecture

```
Query
  |
  v
Query Analysis (intent classification, jurisdiction detection)
  |
  v
Hybrid Retrieval
  |--- Dense: ChromaDB (BGE-large-en-v1.5, cosine similarity)
  |--- Sparse: BM25 (keyword matching)
  |--- Fusion: Reciprocal Rank Fusion (k=60)
  |--- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2
  |
  v
Answer Generation (Qwen 3 8B via Ollama)
  |
  v
Self-Reflection Loop (up to 3 iterations)
  |--- Evaluate answer quality
  |--- Reformulate query if insufficient
  |--- Re-retrieve and re-generate
  |
  v
Cited Answer + Sources
```

## Project structure

```
.
├── app.py                  # Streamlit chat interface
├── ingest.py               # PDF parsing + index building pipeline
├── requirements.txt
├── .env.example
├── data/
│   ├── alaska/             # Alaska Admin Code PDFs (chapters 26, 31, 32)
│   └── hawaii/
│       ├── admin_code/     # Hawaii Administrative Rules (HAR chapters)
│       └── statutes/       # Hawaii Revised Statutes (HRS sections)
└── src/
    ├── config.py           # all tunable parameters in one place
    ├── models.py           # shared data models (DocumentChunk, etc.)
    ├── pdf_parser.py       # PDF extraction for all 3 document formats
    ├── chunker.py          # hierarchical recursive chunking
    ├── embedder.py         # BGE embeddings + ChromaDB storage
    ├── bm25_index.py       # BM25 sparse keyword index
    ├── retriever.py        # hybrid retrieval (dense + sparse + rerank)
    ├── agent.py            # agentic RAG orchestrator with self-reflection
    └── answerer.py         # LLM calls via Ollama (Qwen 3 8B)
```

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- Qwen 3 8B pulled: `ollama pull qwen3:8b`

### Install

```bash
git clone https://github.com/SharanTejaKarra/agentic-rag-legal.git
cd agentic-rag-legal

pip install -r requirements.txt

cp .env.example .env
# edit .env if you need to change Ollama URL or model
```

### Add your data

Place PDF files in the data directory:

```
data/
├── alaska/             # Alaska Administrative Code PDFs
└── hawaii/
    ├── admin_code/     # Hawaii Administrative Rules PDFs
    └── statutes/       # Hawaii Revised Statutes PDFs
```

### Build the indexes

```bash
python ingest.py          # first time
python ingest.py --force  # rebuild from scratch
```

This parses all PDFs, chunks them, builds the ChromaDB vector index and BM25 keyword index. Takes about 4 minutes on an M-series Mac (mainly the embedding step).

### Run the app

```bash
# make sure ollama is running
ollama serve

# in another terminal
streamlit run app.py
```

## How it works

### Chunking strategy

Legal documents have clear structural boundaries (sections, subsections, clauses). Instead of naive fixed-size chunking, the system:

1. Preserves section boundaries as natural chunk boundaries
2. For sections exceeding 512 tokens, splits hierarchically:
   - First on subsection markers: (a), (b), (1), (2)
   - Then on paragraph boundaries
   - Then on sentences
   - Hard word-boundary split as last resort
3. Prepends section ID + title as context header to every chunk
4. Maintains overlap between sub-chunks for continuity

### Hybrid retrieval

Neither dense nor sparse search alone is enough for legal text:
- Dense (semantic) search catches paraphrases but can miss exact legal terms
- Sparse (BM25) search catches exact terms but misses semantic similarity

The system runs both, fuses results with Reciprocal Rank Fusion, then reranks with a cross-encoder for final precision.

### Agentic self-reflection

After generating an answer, the system evaluates it:
- Is the answer grounded in the sources?
- Does it actually address the question?

If not, it reformulates the search query and tries again (up to 3 iterations). This handles cases where the initial retrieval missed relevant sections.

## Configuration

All tunable parameters live in `src/config.py`:

| Parameter | Default | What it controls |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model (1024-dim) |
| `LLM_MODEL` | `qwen3:8b` | Local LLM via Ollama |
| `MAX_CHUNK_TOKENS` | 512 | Max tokens per chunk |
| `TOP_K_DENSE` | 15 | Dense search candidates |
| `TOP_K_SPARSE` | 15 | BM25 search candidates |
| `RERANK_TOP_K` | 5 | Final results after reranking |
| `MAX_AGENT_ITERATIONS` | 3 | Self-reflection loop cap |

## Tech stack

- **LLM**: Qwen 3 8B (local, via Ollama)
- **Embeddings**: BAAI/bge-large-en-v1.5
- **Vector DB**: ChromaDB (persistent, cosine similarity)
- **Sparse search**: BM25 (rank_bm25)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **PDF parsing**: pdfplumber
- **UI**: Streamlit
