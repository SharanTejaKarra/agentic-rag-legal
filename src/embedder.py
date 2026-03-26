"""
Dense embedding + ChromaDB vector store.
Handles indexing and querying with BGE-large-en-v1.5.
"""

import logging

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    BGE_QUERY_PREFIX,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_DIMENSION,
    EMBED_MODEL,
)
from src.models import DocumentChunk, RetrievalResult

log = logging.getLogger(__name__)

# lazy-loaded singleton so we don't reload the model on every call
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_or_create_collection(force_reset: bool = False) -> chromadb.Collection:
    """Return the main legal-sections collection, creating it if needed."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if force_reset:
        # blow away and recreate — useful during re-indexing
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Deleted existing collection '%s'", COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_store(chunks: list[DocumentChunk], force_reset: bool = False) -> None:
    """Embed all chunks and upsert them into ChromaDB in batches."""
    if not chunks:
        log.warning("No chunks to embed — skipping")
        return

    model = _get_model()
    collection = get_or_create_collection(force_reset=force_reset)

    # BGE models don't need a prefix at indexing time, only at query time
    texts = [c.text for c in chunks]
    log.info("Encoding %d chunks with %s ...", len(texts), EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    batch_size = 32
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=[embeddings[i].tolist() for i in range(start, end)],
            documents=[c.text for c in batch],
            metadatas=[c.to_metadata_dict() for c in batch],
        )
        log.info("Upserted batch %d–%d / %d", start, end - 1, len(chunks))

    log.info(
        "Done — collection '%s' now has %d items",
        COLLECTION_NAME,
        collection.count(),
    )


def query_dense(
    query: str,
    top_k: int,
    filter_dict: dict | None = None,
) -> list[RetrievalResult]:
    """Run a dense vector search against the collection."""
    model = _get_model()
    collection = get_or_create_collection()

    # BGE needs the retrieval prefix at query time
    prefixed = BGE_QUERY_PREFIX + query
    embedding = model.encode(prefixed, normalize_embeddings=True).tolist()

    kwargs: dict = {"query_embeddings": [embedding], "n_results": top_k}
    if filter_dict:
        kwargs["where"] = filter_dict

    raw = collection.query(**kwargs)

    results: list[RetrievalResult] = []
    for rank, (doc, meta, dist) in enumerate(
        zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0])
    ):
        # chromadb cosine distance is 1 - similarity, flip it back
        score = 1.0 - dist
        chunk = DocumentChunk.from_metadata_dict(doc, meta)
        results.append(
            RetrievalResult(chunk=chunk, score=score, source="dense", rank=rank)
        )

    return results
