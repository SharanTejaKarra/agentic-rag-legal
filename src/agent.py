"""
Agentic RAG orchestrator for legal document Q&A.
Coordinates query analysis, retrieval, answer generation,
and a self-reflection loop that reformulates when needed.
"""

import logging
import re

from src.answerer import evaluate_answer_quality, generate_answer, reformulate_query
from src.config import MAX_AGENT_ITERATIONS
from src.models import AgentResponse, DocumentChunk, RetrievalResult
from src.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

# patterns that hint at a specific section reference
_SECTION_RE = re.compile(
    r"""
    (?:\d+\s*AAC\s*[\d.]+)      # Alaska admin code  (3 AAC 26.080)
    |(?:§\s*[\d:A-Za-z.\-]+)    # Hawaii statute     (§431:10A-102)
    |(?:HRS\s*§?\s*[\d:.\-]+)   # HRS references
    |(?:HAR\s*§?\s*[\d:.\-]+)   # HAR references
    |(?:(?:section|sec\.?)\s+[\d.:A-Za-z\-]+)   # generic "section X.Y"
    """,
    re.IGNORECASE | re.VERBOSE,
)

# jurisdiction keywords
_JURISDICTION_HINTS = {
    "alaska": "alaska",
    "aac": "alaska",
    "hawaii": "hawaii",
    "hrs": "hawaii",
    "har": "hawaii",
}


# ── query analysis (no LLM needed) ──────────────────────────────────────

def analyze_query(query: str) -> dict:
    """Classify intent, detect jurisdiction, and extract section references
    using simple heuristics -- fast and deterministic."""
    query_lower = query.lower()

    # pull out section references
    section_refs = [m.group().strip() for m in _SECTION_RE.finditer(query)]

    # figure out intent
    if any(kw in query_lower for kw in ("compare", "difference between", "differ from", "vs")):
        intent = "comparison"
    elif any(kw in query_lower for kw in ("what is", "what are", "define", "definition of", "meaning of")):
        intent = "definition"
    elif section_refs:
        intent = "section_lookup"
    else:
        intent = "topic_search"

    # jurisdiction detection from keywords or section format
    jurisdiction = None
    for token, jur in _JURISDICTION_HINTS.items():
        if token in query_lower:
            jurisdiction = jur
            break

    return {
        "intent": intent,
        "jurisdiction": jurisdiction,
        "section_refs": section_refs,
    }


# ── citation extraction ─────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[Source:\s*([^,\]]+),\s*([^\]]+)\]")


def _extract_citations(
    answer: str, chunks: list[DocumentChunk]
) -> list[dict]:
    """Pull [Source: section_id, file] markers out of the LLM's answer
    and enrich them with metadata from the retrieved chunks."""
    # build a lookup for quick metadata enrichment
    # store both exact and base section_id (strip subsection like "(a)")
    chunk_meta: dict[str, dict] = {}
    for c in chunks:
        meta = {"title": c.title, "jurisdiction": c.jurisdiction}
        chunk_meta[c.section_id] = meta
        # also store stripped version for fuzzy matching
        base = re.sub(r"\([^)]*\)\s*$", "", c.section_id).strip()
        chunk_meta[base] = meta

    def _find_meta(section_id: str) -> dict:
        """Try exact match, then strip subsection refs like (a), (b)."""
        if section_id in chunk_meta:
            return chunk_meta[section_id]
        base = re.sub(r"\([^)]*\)\s*$", "", section_id).strip()
        return chunk_meta.get(base, {})

    seen = set()
    citations = []
    for match in _CITATION_RE.finditer(answer):
        section_id = match.group(1).strip()
        source_file = match.group(2).strip()
        key = (section_id, source_file)
        if key not in seen:
            seen.add(key)
            meta = _find_meta(section_id)
            citations.append({
                "section_id": section_id,
                "source_file": source_file,
                "title": meta.get("title", ""),
                "jurisdiction": meta.get("jurisdiction", ""),
            })
    return citations


# ── main orchestrator ────────────────────────────────────────────────────

def _deduplicate_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Remove duplicate chunks (by chunk_id), keeping first occurrence."""
    seen: set[str] = set()
    unique: list[DocumentChunk] = []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique.append(c)
    return unique


def run(query: str, jurisdiction: str | None = None) -> AgentResponse:
    """Execute the full agentic RAG pipeline.

    Steps: analyze -> retrieve -> answer -> self-reflect (loop) -> return.
    """
    query = query.strip()
    if not query:
        return AgentResponse(answer="Please provide a question.")

    # -- 1. analyze --
    analysis = analyze_query(query)
    logger.info(
        "Query analysis | intent=%s  jurisdiction=%s  refs=%s",
        analysis["intent"], analysis["jurisdiction"], analysis["section_refs"],
    )

    # caller-supplied jurisdiction overrides heuristic detection
    if jurisdiction is None:
        jurisdiction = analysis["jurisdiction"]
    reformulations: list[str] = []

    # -- 2. retrieve --
    try:
        results: list[RetrievalResult] = hybrid_retrieve(
            query, jurisdiction=jurisdiction
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return AgentResponse(
            answer=f"Retrieval error: {exc}. Make sure the index has been built.",
        )

    if not results:
        logger.warning("No results for initial query")
        return AgentResponse(
            answer="No relevant documents were found for your query. "
                   "Try rephrasing or broadening your search.",
        )

    # -- 3. check relevance (very low scores = probably off-topic) --
    top_score = results[0].score
    current_query = query
    all_chunks = [r.chunk for r in results]

    if top_score < 0.05:
        logger.info("Top score %.4f is very low, attempting reformulation", top_score)
        new_query = reformulate_query(
            query, "The initial retrieval returned very low relevance scores."
        )
        if new_query != query:
            reformulations.append(new_query)
            current_query = new_query
            try:
                results = hybrid_retrieve(current_query, jurisdiction=jurisdiction)
                all_chunks = _deduplicate_chunks(
                    all_chunks + [r.chunk for r in results]
                )
            except Exception as exc:
                logger.warning("Re-retrieval after reformulation failed: %s", exc)

    # -- 4. generate answer --
    answer_chunks = all_chunks[: 5]  # feed top chunks to the LLM
    answer = generate_answer(current_query, answer_chunks, jurisdiction)

    # -- 5. self-reflection loop --
    iterations = 1
    for i in range(MAX_AGENT_ITERATIONS - 1):
        evaluation = evaluate_answer_quality(current_query, answer, answer_chunks)
        logger.info(
            "Self-reflection #%d | sufficient=%s  reason=%s",
            i + 1, evaluation["is_sufficient"], evaluation.get("reason", ""),
        )

        if evaluation["is_sufficient"]:
            break

        # not good enough -- try reformulating and re-retrieving
        suggested = evaluation.get("suggested_reformulation")
        if suggested:
            reformulations.append(suggested)
            current_query = suggested
        else:
            # fall back to LLM-based reformulation
            context = f"Previous answer was judged insufficient: {evaluation.get('reason', '')}"
            new_q = reformulate_query(query, context)
            reformulations.append(new_q)
            current_query = new_q

        try:
            new_results = hybrid_retrieve(current_query, jurisdiction=jurisdiction)
            new_chunks = [r.chunk for r in new_results]
            all_chunks = _deduplicate_chunks(all_chunks + new_chunks)
        except Exception as exc:
            logger.warning("Re-retrieval on iteration %d failed: %s", i + 1, exc)

        # regenerate with the expanded chunk pool
        answer_chunks = all_chunks[: 7]  # allow a few more chunks on retries
        answer = generate_answer(current_query, answer_chunks, jurisdiction)
        iterations += 1
    else:
        # ran out of iterations without a "sufficient" verdict
        logger.info("Hit max iterations (%d) without a sufficient answer", MAX_AGENT_ITERATIONS)

    # -- 6. extract citations --
    citations = _extract_citations(answer, all_chunks)

    # -- 7. build response --
    return AgentResponse(
        answer=answer,
        citations=citations,
        retrieved_chunks=results,
        query_reformulations=reformulations,
        iterations_used=iterations,
    )
