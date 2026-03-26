"""
BM25 sparse keyword index — the other half of hybrid retrieval.
Builds and queries a rank_bm25 index persisted to disk with pickle.
"""

import logging
import pickle
import re

from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH
from src.models import DocumentChunk, RetrievalResult

log = logging.getLogger(__name__)

# common English stopwords — not exhaustive, just enough to cut noise
STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "not", "no", "nor",
    "is", "are", "was", "were", "am", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "i", "you", "me", "him", "her",
    "if", "so", "as", "than", "then", "when", "what", "which", "who",
    "will", "would", "shall", "should", "can", "could", "may", "might",
}

# cached in-memory copy of the loaded index
_cached_index: BM25Okapi | None = None
_cached_chunks: list[DocumentChunk] | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric chars, drop stopwords."""
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in tokens if t and t not in STOPWORDS]


def build_bm25_index(chunks: list[DocumentChunk]) -> None:
    """Tokenize all chunks, fit BM25, and save to disk."""
    global _cached_index, _cached_chunks

    if not chunks:
        log.warning("No chunks to index — skipping BM25 build")
        return

    log.info("Tokenizing %d chunks for BM25 ...", len(chunks))
    corpus = [_tokenize(c.text) for c in chunks]

    log.info("Fitting BM25Okapi index ...")
    index = BM25Okapi(corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "chunks": chunks}, f)

    # update the module cache while we're at it
    _cached_index = index
    _cached_chunks = chunks

    log.info("BM25 index saved to %s (%d docs)", BM25_INDEX_PATH, len(chunks))


def _load_index() -> tuple[BM25Okapi, list[DocumentChunk]]:
    """Load (and cache) the BM25 index from disk."""
    global _cached_index, _cached_chunks

    if _cached_index is not None and _cached_chunks is not None:
        return _cached_index, _cached_chunks

    log.info("Loading BM25 index from %s", BM25_INDEX_PATH)
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    _cached_index = data["index"]
    _cached_chunks = data["chunks"]
    return _cached_index, _cached_chunks


def query_sparse(
    query: str,
    top_k: int,
    filter_jurisdiction: str | None = None,
) -> list[RetrievalResult]:
    """Score all docs with BM25, optionally filter by jurisdiction, return top_k."""
    index, chunks = _load_index()

    tokenized_query = _tokenize(query)
    scores = index.get_scores(tokenized_query)

    # pair up (score, chunk) and optionally filter
    scored = []
    for score, chunk in zip(scores, chunks):
        if filter_jurisdiction and chunk.jurisdiction != filter_jurisdiction:
            continue
        scored.append((score, chunk))

    # sort descending by score and take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    return [
        RetrievalResult(chunk=chunk, score=score, source="sparse", rank=rank)
        for rank, (score, chunk) in enumerate(top)
    ]
