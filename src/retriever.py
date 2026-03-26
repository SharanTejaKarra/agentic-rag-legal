"""
Hybrid retriever: dense + sparse search, reciprocal rank fusion, cross-encoder reranking.
This is the main entry point for fetching relevant chunks given a user query.
"""

import logging
from collections import defaultdict

from sentence_transformers import CrossEncoder

from src.bm25_index import query_sparse
from src.config import (
    RERANK_MODEL,
    RERANK_TOP_K,
    RRF_K,
    TOP_K_DENSE,
    TOP_K_FUSED,
    TOP_K_SPARSE,
)
from src.embedder import query_dense
from src.models import RetrievalResult

log = logging.getLogger(__name__)

# lazy-loaded cross-encoder singleton
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        log.info("Loading reranker: %s", RERANK_MODEL)
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    k: int = RRF_K,
) -> list[RetrievalResult]:
    """
    Merge two ranked lists using RRF.
    score(doc) = sum over lists of 1/(k + rank) where rank is 0-based.
    """
    # accumulate RRF scores per chunk_id
    fused_scores: dict[str, float] = defaultdict(float)
    chunk_lookup: dict[str, RetrievalResult] = {}

    for ranked_list in (dense_results, sparse_results):
        for result in ranked_list:
            cid = result.chunk.chunk_id
            fused_scores[cid] += 1.0 / (k + result.rank)
            # keep whichever copy we saw first — they carry the same chunk data
            if cid not in chunk_lookup:
                chunk_lookup[cid] = result

    # sort by fused score descending
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        RetrievalResult(
            chunk=chunk_lookup[cid].chunk,
            score=score,
            source="fused",
            rank=rank,
        )
        for rank, (cid, score) in enumerate(ranked)
    ]


def rerank(
    query: str,
    results: list[RetrievalResult],
    top_k: int = RERANK_TOP_K,
) -> list[RetrievalResult]:
    """Re-score candidates with a cross-encoder and keep the top_k."""
    if not results:
        return []

    model = _get_reranker()
    pairs = [(query, r.chunk.text) for r in results]
    scores = model.predict(pairs)

    # attach scores and sort
    scored = sorted(
        zip(scores, results), key=lambda x: x[0], reverse=True
    )
    top = scored[:top_k]

    return [
        RetrievalResult(
            chunk=result.chunk,
            score=float(score),
            source="reranked",
            rank=rank,
        )
        for rank, (score, result) in enumerate(top)
    ]


def hybrid_retrieve(
    query: str,
    top_k: int = RERANK_TOP_K,
    jurisdiction: str | None = None,
) -> list[RetrievalResult]:
    """
    Full retrieval pipeline:
      dense search -> sparse search -> RRF fusion -> cross-encoder rerank
    """
    log.info("Hybrid retrieval for: '%s' (jurisdiction=%s)", query, jurisdiction)

    # build the jurisdiction filter for dense search if needed
    dense_filter = {"jurisdiction": jurisdiction} if jurisdiction else None

    dense_hits = query_dense(query, top_k=TOP_K_DENSE, filter_dict=dense_filter)
    log.info("Dense search returned %d candidates", len(dense_hits))

    sparse_hits = query_sparse(
        query, top_k=TOP_K_SPARSE, filter_jurisdiction=jurisdiction
    )
    log.info("Sparse search returned %d candidates", len(sparse_hits))

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)[:TOP_K_FUSED]
    log.info("RRF fusion kept %d candidates", len(fused))

    final = rerank(query, fused, top_k=top_k)
    log.info(
        "Reranking done — returning top %d results (scores %.3f – %.3f)",
        len(final),
        final[0].score if final else 0.0,
        final[-1].score if final else 0.0,
    )

    return final
