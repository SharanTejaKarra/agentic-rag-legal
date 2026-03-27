"""
LLM interface for the legal Q&A pipeline.
All calls go through Ollama's REST API (Qwen 3 8B).
"""

import logging
import time

import requests

from src.config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE, MAX_MULTI_HOP_STEPS, OLLAMA_BASE_URL
from src.models import DocumentChunk

logger = logging.getLogger(__name__)

CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"

# retry config for ollama calls
MAX_RETRIES = 2
BACKOFF_BASE = 1.5  # seconds; doubles each retry


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Send a chat completion request to Ollama with retry + backoff.

    Appends /no_think to the user message so Qwen 3 skips its internal
    chain-of-thought and gives a direct answer.
    """
    user_prompt_final = f"{user_prompt}\n\n/no_think"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_final},
        ],
        "stream": False,
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
        },
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info("Ollama request (attempt %d/%d)", attempt + 1, MAX_RETRIES + 1)
            resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            logger.debug("LLM response length: %d chars", len(content))
            return content
        except requests.ConnectionError:
            logger.warning("Ollama connection failed (attempt %d)", attempt + 1)
        except requests.Timeout:
            logger.warning("Ollama request timed out (attempt %d)", attempt + 1)
        except (requests.HTTPError, KeyError) as exc:
            logger.warning("Ollama error on attempt %d: %s", attempt + 1, exc)

        if attempt < MAX_RETRIES:
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.info("Retrying in %.1fs ...", wait)
            time.sleep(wait)

    return "[Error] Could not reach the language model. Please check that Ollama is running."


# ── answer generation ────────────────────────────────────────────────────

SYSTEM_PROMPT_ANSWER = """\
You are a legal research assistant. Answer the user's question using ONLY the \
legal text excerpts provided below. Do not rely on outside knowledge.

Rules:
- After each factual claim, cite the source in this exact format: \
[Source: <section_id>, <source_file>]
- Quote the specific subsections or clauses that support your answer.
- If the provided excerpts do not contain enough information to answer, \
say: "The provided documents do not address this question."
- Do not speculate or hallucinate. Stick to what the text says."""


def generate_answer(
    query: str,
    chunks: list[DocumentChunk],
    jurisdiction: str | None = None,
) -> str:
    """Build a prompt from retrieved chunks and ask the LLM for a cited answer."""
    if not chunks:
        return "No relevant documents were retrieved for this query."

    # format the context block with numbered excerpts
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[{i}] Section: {chunk.section_id} | Title: {chunk.title} | "
            f"Jurisdiction: {chunk.jurisdiction} | File: {chunk.source_file}"
        )
        context_parts.append(f"{header}\n{chunk.text}")
    context_block = "\n\n---\n\n".join(context_parts)

    jurisdiction_note = ""
    if jurisdiction:
        jurisdiction_note = f"\nFocus on {jurisdiction.title()} law where relevant."

    user_prompt = (
        f"Legal excerpts:\n\n{context_block}\n\n---\n\n"
        f"Question: {query}{jurisdiction_note}"
    )

    logger.info("Generating answer for query: %.80s...", query)
    return _call_ollama(SYSTEM_PROMPT_ANSWER, user_prompt)


# ── answer quality evaluation ────────────────────────────────────────────

SYSTEM_PROMPT_EVAL = """\
You are an answer-quality evaluator for a legal Q&A system. Given a question, \
an answer, and the source excerpts, decide whether the answer adequately \
addresses the question.

Respond with EXACTLY this JSON (no markdown fences):
{"is_sufficient": true/false, "reason": "...", "suggested_reformulation": "..." or null}

- is_sufficient: true if the answer directly addresses the question using the sources.
- reason: brief explanation of your judgment.
- suggested_reformulation: if the answer is insufficient, suggest a better search \
query that might find the missing information. null if sufficient."""


def evaluate_answer_quality(
    query: str,
    answer: str,
    chunks: list[DocumentChunk],
) -> dict:
    """Ask the LLM whether the generated answer actually addresses the query."""
    chunk_summaries = "\n".join(
        f"- {c.section_id}: {c.title}" for c in chunks[:5]
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Answer given:\n{answer}\n\n"
        f"Sources used:\n{chunk_summaries}"
    )

    logger.info("Evaluating answer quality for: %.80s...", query)
    raw = _call_ollama(SYSTEM_PROMPT_EVAL, user_prompt)

    # try to parse the json response; fall back to "sufficient" so we don't
    # loop forever when the LLM returns malformed output
    import json

    try:
        result = json.loads(raw)
        # make sure required keys exist
        return {
            "is_sufficient": bool(result.get("is_sufficient", True)),
            "reason": str(result.get("reason", "")),
            "suggested_reformulation": result.get("suggested_reformulation"),
        }
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse evaluation response, assuming sufficient: %s", raw[:200])
        return {
            "is_sufficient": True,
            "reason": "Evaluation response was not valid JSON; defaulting to sufficient.",
            "suggested_reformulation": None,
        }


# ── query reformulation ─────────────────────────────────────────────────

SYSTEM_PROMPT_REFORMULATE = """\
You are a search-query optimizer for a legal document retrieval system. \
Given the original query and some context about why it didn't work well, \
produce a single improved search query. Return ONLY the new query, nothing else."""


def reformulate_query(original_query: str, context: str) -> str:
    """Ask the LLM to rewrite the query for better retrieval results."""
    user_prompt = (
        f"Original query: {original_query}\n\n"
        f"Context: {context}"
    )

    logger.info("Reformulating query: %.80s...", original_query)
    result = _call_ollama(SYSTEM_PROMPT_REFORMULATE, user_prompt)

    # strip any stray quotes the LLM might wrap around the query
    reformulated = result.strip().strip('"').strip("'")
    if not reformulated:
        return original_query
    return reformulated


# ── multi-hop: query decomposition ────────────────────────────────────

SYSTEM_PROMPT_DECOMPOSE = """\
You are a query planner for a legal document retrieval system.

Given a user question, decide whether it can be answered by a single search or \
needs to be broken into ordered sub-questions where each step builds on the previous.

Respond with EXACTLY this JSON (no markdown fences, no explanation):
{"needs_multi_hop": true/false, "sub_questions": ["q1", "q2", ...]}

Rules:
- If the question is straightforward (one topic, one jurisdiction, no cross-references), \
set needs_multi_hop to false and sub_questions to an empty list.
- If the question requires chaining information (e.g. find a rule, then find what \
penalizes its violation; or compare provisions across jurisdictions), \
set needs_multi_hop to true.
- Order sub_questions so that earlier answers inform later searches.
- Maximum """ + str(MAX_MULTI_HOP_STEPS) + """ sub-questions. Fewer is better."""


def decompose_query(query: str) -> dict:
    """Break a complex query into ordered sub-questions when needed.

    Returns {"needs_multi_hop": bool, "sub_questions": list[str]}.
    For simple queries, needs_multi_hop is False and the list is empty.
    """
    import json

    logger.info("Checking if query needs multi-hop decomposition")
    raw = _call_ollama(SYSTEM_PROMPT_DECOMPOSE, query)

    try:
        result = json.loads(raw)
        needs = bool(result.get("needs_multi_hop", False))
        subs = result.get("sub_questions", [])
        # sanity: cap at MAX_MULTI_HOP_STEPS, discard empty strings
        subs = [s.strip() for s in subs if s.strip()][:MAX_MULTI_HOP_STEPS]
        if needs and len(subs) < 2:
            # if the LLM says multi-hop but only gave 0-1 sub-questions, skip it
            needs = False
            subs = []
        return {"needs_multi_hop": needs, "sub_questions": subs}
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse decomposition response, treating as single-hop: %s", raw[:200])
        return {"needs_multi_hop": False, "sub_questions": []}


# ── multi-hop: intermediate summarization ─────────────────────────────

SYSTEM_PROMPT_HOP_SUMMARY = """\
You are a legal research assistant performing multi-step reasoning. \
Given a sub-question and the retrieved legal excerpts, write a brief factual \
summary that captures only the key information needed for the next step.

Rules:
- Stick strictly to what the excerpts say. No speculation.
- Cite sections inline like [Section X.Y].
- Keep it under 150 words — this is an intermediate step, not the final answer."""


def summarize_hop(sub_question: str, chunks: list[DocumentChunk]) -> str:
    """Produce a short intermediate summary for one hop of a multi-hop chain."""
    if not chunks:
        return "No relevant information found for this sub-question."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk.section_id} | {chunk.title} | {chunk.jurisdiction}"
        context_parts.append(f"{header}\n{chunk.text}")
    context_block = "\n\n---\n\n".join(context_parts)

    user_prompt = f"Sub-question: {sub_question}\n\nExcerpts:\n\n{context_block}"
    return _call_ollama(SYSTEM_PROMPT_HOP_SUMMARY, user_prompt)


# ── multi-hop: final synthesis ────────────────────────────────────────

SYSTEM_PROMPT_MULTI_HOP = """\
You are a legal research assistant. The user asked a complex question that was \
broken into steps. You have intermediate findings from each step plus the \
original legal excerpts.

Synthesize a final answer that chains the findings together logically. \
Cite every factual claim with [Source: <section_id>, <source_file>].

Rules:
- Only use information from the provided excerpts and intermediate findings.
- Show how the pieces connect — that's the whole point of multi-hop reasoning.
- If any step found no relevant information, acknowledge the gap.
- Do not speculate beyond what the sources say."""


def generate_multi_hop_answer(
    original_query: str,
    sub_questions: list[str],
    hop_summaries: list[str],
    all_chunks: list[DocumentChunk],
    jurisdiction: str | None = None,
) -> str:
    """Final synthesis across all hops for a multi-hop query."""
    # build the chain-of-findings block
    chain_parts = []
    for i, (sq, summary) in enumerate(zip(sub_questions, hop_summaries), 1):
        chain_parts.append(f"Step {i}: {sq}\nFindings: {summary}")
    chain_block = "\n\n".join(chain_parts)

    # build the evidence block from all accumulated chunks
    evidence_parts = []
    for i, chunk in enumerate(all_chunks, 1):
        header = (
            f"[{i}] Section: {chunk.section_id} | Title: {chunk.title} | "
            f"Jurisdiction: {chunk.jurisdiction} | File: {chunk.source_file}"
        )
        evidence_parts.append(f"{header}\n{chunk.text}")
    evidence_block = "\n\n---\n\n".join(evidence_parts)

    jurisdiction_note = ""
    if jurisdiction:
        jurisdiction_note = f"\nFocus on {jurisdiction.title()} law where relevant."

    user_prompt = (
        f"Chain of findings:\n\n{chain_block}\n\n"
        f"===\n\nFull legal excerpts:\n\n{evidence_block}\n\n---\n\n"
        f"Original question: {original_query}{jurisdiction_note}"
    )

    logger.info("Generating multi-hop synthesis for: %.80s...", original_query)
    return _call_ollama(SYSTEM_PROMPT_MULTI_HOP, user_prompt)
