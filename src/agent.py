"""
Agentic RAG orchestrator for legal document Q&A.
Coordinates query analysis, retrieval, answer generation,
and a self-reflection loop that reformulates when needed.
"""

import logging
import re

from src.answerer import (
    decompose_query,
    evaluate_answer_quality,
    generate_answer,
    generate_multi_hop_answer,
    reformulate_query,
    summarize_hop,
)
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


def _run_single_hop(
    query: str, jurisdiction: str | None
) -> AgentResponse:
    """Standard single-hop pipeline: retrieve -> answer -> self-reflect loop."""
    reformulations: list[str] = []

    # -- retrieve --
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
        return AgentResponse(
            answer="No relevant documents were found for your query. "
                   "Try rephrasing or broadening your search.",
        )

    # low-score reformulation
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

    # generate answer
    answer_chunks = all_chunks[:5]
    answer = generate_answer(current_query, answer_chunks, jurisdiction)

    # self-reflection loop
    iterations = 1
    for i in range(MAX_AGENT_ITERATIONS - 1):
        evaluation = evaluate_answer_quality(current_query, answer, answer_chunks)
        logger.info(
            "Self-reflection #%d | sufficient=%s  reason=%s",
            i + 1, evaluation["is_sufficient"], evaluation.get("reason", ""),
        )

        if evaluation["is_sufficient"]:
            break

        suggested = evaluation.get("suggested_reformulation")
        if suggested:
            reformulations.append(suggested)
            current_query = suggested
        else:
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

        answer_chunks = all_chunks[:7]
        answer = generate_answer(current_query, answer_chunks, jurisdiction)
        iterations += 1
    else:
        logger.info("Hit max iterations (%d) without a sufficient answer", MAX_AGENT_ITERATIONS)

    citations = _extract_citations(answer, all_chunks)

    return AgentResponse(
        answer=answer,
        citations=citations,
        retrieved_chunks=results,
        query_reformulations=reformulations,
        iterations_used=iterations,
    )


def _run_multi_hop(
    query: str,
    sub_questions: list[str],
    jurisdiction: str | None,
) -> AgentResponse:
    """Multi-hop pipeline: retrieve for each sub-question sequentially,
    carrying forward context from earlier hops to inform later ones."""
    all_chunks: list[DocumentChunk] = []
    all_results: list[RetrievalResult] = []
    hop_summaries: list[str] = []
    prior_context = ""

    for step, sub_q in enumerate(sub_questions, 1):
        # if we have findings from earlier hops, prepend them so the
        # retrieval query is better grounded
        search_query = sub_q
        if prior_context:
            search_query = f"{sub_q} (context: {prior_context})"

        logger.info("Multi-hop step %d/%d: %s", step, len(sub_questions), sub_q)

        try:
            results = hybrid_retrieve(search_query, jurisdiction=jurisdiction)
        except Exception as exc:
            logger.warning("Retrieval failed on hop %d: %s", step, exc)
            hop_summaries.append(f"Retrieval failed: {exc}")
            continue

        hop_chunks = [r.chunk for r in results]
        all_results.extend(results)
        all_chunks = _deduplicate_chunks(all_chunks + hop_chunks)

        # produce an intermediate summary for this hop
        summary = summarize_hop(sub_q, hop_chunks[:5])
        hop_summaries.append(summary)
        logger.info("Hop %d summary (%.60s...)", step, summary)

        # carry this summary forward as context for the next hop
        prior_context = summary

    # final synthesis across all hops
    # cap evidence chunks sent to the LLM to avoid blowing the context
    evidence_chunks = all_chunks[:10]
    answer = generate_multi_hop_answer(
        query, sub_questions, hop_summaries, evidence_chunks, jurisdiction
    )

    citations = _extract_citations(answer, all_chunks)

    return AgentResponse(
        answer=answer,
        citations=citations,
        retrieved_chunks=all_results,
        query_reformulations=[],
        iterations_used=len(sub_questions),
        is_multi_hop=True,
        sub_questions=sub_questions,
        hop_summaries=hop_summaries,
    )


def run(query: str, jurisdiction: str | None = None) -> AgentResponse:
    """Execute the full agentic RAG pipeline.

    Decides between single-hop and multi-hop based on query complexity:
      - Single-hop: analyze -> retrieve -> answer -> self-reflect
      - Multi-hop:  analyze -> decompose -> sequential retrieve per sub-question
                    -> intermediate summaries -> final synthesis
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

    if jurisdiction is None:
        jurisdiction = analysis["jurisdiction"]

    # -- 2. check if multi-hop is needed --
    decomposition = decompose_query(query)
    if decomposition["needs_multi_hop"]:
        logger.info(
            "Multi-hop activated — %d sub-questions: %s",
            len(decomposition["sub_questions"]),
            decomposition["sub_questions"],
        )
        return _run_multi_hop(query, decomposition["sub_questions"], jurisdiction)

    # -- 3. single-hop path --
    logger.info("Single-hop path")
    return _run_single_hop(query, jurisdiction)
