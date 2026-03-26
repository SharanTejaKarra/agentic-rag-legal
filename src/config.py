import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ALASKA_DIR = DATA_DIR / "alaska"
HAWAII_ADMIN_DIR = DATA_DIR / "hawaii" / "admin_code"
HAWAII_STATUTES_DIR = DATA_DIR / "hawaii" / "statutes"
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
BM25_INDEX_PATH = str(PROJECT_ROOT / "bm25_index.pkl")

# ── embedding model ────────────────────────────────────────────────────
# BGE-large-en-v1.5: strong on legal/regulatory text, 1024-dim.
# requires a query-time prefix for retrieval but NOT at indexing time.
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIMENSION = 1024
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── chromadb ───────────────────────────────────────────────────────────
COLLECTION_NAME = "legal_sections"

# ── BM25 sparse index ─────────────────────────────────────────────────
BM25_K1 = 1.5
BM25_B = 0.75

# ── retrieval tuning ──────────────────────────────────────────────────
TOP_K_DENSE = 15          # candidates from vector search
TOP_K_SPARSE = 15         # candidates from BM25
TOP_K_FUSED = 10          # after reciprocal rank fusion
RRF_K = 60                # RRF constant (standard default)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 5          # final results after reranking

# ── LLM ────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048
MAX_AGENT_ITERATIONS = 3  # self-reflection loop cap

# ── chunking ───────────────────────────────────────────────────────────
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
