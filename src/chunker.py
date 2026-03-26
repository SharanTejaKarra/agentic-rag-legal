"""
Converts raw parsed sections into DocumentChunk objects.
Long sections get split hierarchically: subsection boundaries first,
then paragraph boundaries, then sentence boundaries with overlap.
"""

import re
import logging

from src.models import DocumentChunk
from src.config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS

logger = logging.getLogger(__name__)

# ── token counting (whitespace-based) ─────────────────────────────────

def _count_tokens(text: str) -> int:
    """Simple whitespace tokenizer -- good enough for chunking decisions."""
    return len(text.split())


# ── splitting helpers ─────────────────────────────────────────────────

def _split_on_subsections(text: str) -> list[str]:
    """
    Split text on subsection markers like (a), (b), (1), (2), etc.
    These appear at the start of a line (possibly indented) in legal text.
    Only splits if it produces more than one chunk.
    """
    # Match lines starting with (a), (b), (1), (2), (A), (B), (i), (ii) etc.
    pattern = re.compile(r"(?=\n\s*\([a-zA-Z0-9]+\)\s)")
    parts = pattern.split(text)
    # Filter out empty parts
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [text]


def _split_on_paragraphs(text: str) -> list[str]:
    """Split on double newlines (paragraph boundaries)."""
    parts = re.split(r"\n\s*\n", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [text]


def _split_on_sentences(text: str) -> list[str]:
    """
    Split on sentence boundaries. Uses a simple heuristic: period followed
    by a space and an uppercase letter, or period at end of line.
    Legal text has lots of abbreviations so we keep it conservative.
    """
    # Split on ". " followed by uppercase, or ".\n"
    parts = re.split(r"(?<=\.)\s+(?=[A-Z(§\"])", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [text]


def _merge_small_parts_with_overlap(
    parts: list[str], max_tokens: int, overlap_tokens: int
) -> list[str]:
    """
    Greedily merge a list of text fragments into chunks that fit within
    max_tokens. When starting a new chunk, pull in overlap_tokens worth
    of trailing text from the previous chunk for context continuity.
    """
    if not parts:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for part in parts:
        part_tokens = _count_tokens(part)

        # If a single part exceeds max_tokens on its own, it gets its own chunk.
        # We'll handle this downstream if needed -- nothing we can do here.
        if current_tokens + part_tokens > max_tokens and current_parts:
            chunks.append("\n".join(current_parts))

            # Build overlap: take trailing text from the chunk we just closed
            overlap_text = _get_trailing_overlap(chunks[-1], overlap_tokens)
            if overlap_text:
                current_parts = [overlap_text, part]
                current_tokens = _count_tokens(overlap_text) + part_tokens
            else:
                current_parts = [part]
                current_tokens = part_tokens
        else:
            current_parts.append(part)
            current_tokens += part_tokens

    if current_parts:
        chunks.append("\n".join(current_parts))

    return chunks


def _get_trailing_overlap(text: str, overlap_tokens: int) -> str:
    """Grab the last `overlap_tokens` tokens from text as overlap context."""
    if overlap_tokens <= 0:
        return ""
    words = text.split()
    if len(words) <= overlap_tokens:
        return ""  # chunk is too small for meaningful overlap
    return " ".join(words[-overlap_tokens:])


# ── recursive splitting logic ─────────────────────────────────────────

def _split_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Recursively split text to fit within max_tokens per chunk.
    Strategy order: subsections -> paragraphs -> sentences.
    """
    if _count_tokens(text) <= max_tokens:
        return [text]

    # Try subsection splits first
    parts = _split_on_subsections(text)
    if len(parts) > 1:
        merged = _merge_small_parts_with_overlap(parts, max_tokens, overlap_tokens)
        # Recurse on any chunk that's still too big
        result = []
        for chunk in merged:
            if _count_tokens(chunk) > max_tokens:
                result.extend(_split_text_paragraphs(chunk, max_tokens, overlap_tokens))
            else:
                result.append(chunk)
        return result

    # Fall through to paragraph splitting
    return _split_text_paragraphs(text, max_tokens, overlap_tokens)


def _split_text_paragraphs(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split on paragraphs, falling through to sentences if needed."""
    parts = _split_on_paragraphs(text)
    if len(parts) > 1:
        merged = _merge_small_parts_with_overlap(parts, max_tokens, overlap_tokens)
        result = []
        for chunk in merged:
            if _count_tokens(chunk) > max_tokens:
                result.extend(_split_text_sentences(chunk, max_tokens, overlap_tokens))
            else:
                result.append(chunk)
        return result

    return _split_text_sentences(text, max_tokens, overlap_tokens)


def _split_text_sentences(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Last resort: split on sentence boundaries."""
    parts = _split_on_sentences(text)
    merged = _merge_small_parts_with_overlap(parts, max_tokens, overlap_tokens)

    # If a chunk is STILL too big after sentence splitting (e.g. a single
    # enormous run-on sentence), hard-split on word boundaries.
    result = []
    for chunk in merged:
        if _count_tokens(chunk) > max_tokens:
            result.extend(_hard_split(chunk, max_tokens, overlap_tokens))
        else:
            result.append(chunk)
    return result


def _hard_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Brute-force word-boundary split for text that can't be split any other way."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        # Advance with overlap
        start = end - overlap_tokens if end < len(words) else end
    return chunks


# ── chunk ID generation ───────────────────────────────────────────────

def _clean_section_id_for_chunk_id(section_id: str) -> str:
    """
    Turn a section ID like '§431:10A-102' or '3 AAC 26.010' into a safe
    string for use in chunk IDs.
    """
    cleaned = section_id
    cleaned = cleaned.replace("§", "").replace(":", "_")
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", cleaned)
    # Collapse repeated underscores
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _make_chunk_id(jurisdiction: str, doc_type: str, section_id: str, chunk_idx: int) -> str:
    clean_sid = _clean_section_id_for_chunk_id(section_id)
    return f"{jurisdiction}_{doc_type}_{clean_sid}_{chunk_idx}"


# ── main entry point ──────────────────────────────────────────────────

def chunk_sections(raw_sections: list[dict]) -> list[DocumentChunk]:
    """
    Convert raw parsed sections (from pdf_parser) into DocumentChunk objects.
    Sections that exceed MAX_CHUNK_TOKENS get split hierarchically.
    """
    all_chunks: list[DocumentChunk] = []
    seen_ids: dict[str, int] = {}  # track ID collisions from duplicate section numbers

    for section in raw_sections:
        section_id = section["section_id"]
        title = section["title"]
        text = section["text"]
        jurisdiction = section["jurisdiction"]
        doc_type = section["doc_type"]
        page_numbers = section.get("page_numbers", [])
        source_file = section.get("source_file", "")
        is_repealed = section.get("is_repealed", False)

        # Repealed sections get a single minimal chunk
        if is_repealed:
            chunk_id = _make_chunk_id(jurisdiction, doc_type, section_id, 0)
            if chunk_id in seen_ids:
                seen_ids[chunk_id] += 1
                chunk_id = f"{chunk_id}_d{seen_ids[chunk_id]}"
            else:
                seen_ids[chunk_id] = 0
            all_chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=section_id,
                jurisdiction=jurisdiction,
                doc_type=doc_type,
                section_id=section_id,
                title=title,
                text=f"Section {section_id} - {title}\n\nThis section has been repealed.",
                source_file=source_file,
                page_numbers=page_numbers,
                chunk_index=0,
                total_chunks=1,
                parent_section_id=section_id,
            ))
            continue

        # Empty sections (shouldn't happen, but be safe)
        if not text.strip():
            logger.warning("Empty text for section %s, skipping", section_id)
            continue

        # Reserve token budget for the header we prepend to every chunk
        header = f"Section {section_id} - {title}\n\n"
        header_tokens = _count_tokens(header)
        body_budget = max(MAX_CHUNK_TOKENS - header_tokens, 128)

        text_chunks = _split_text(text, body_budget, CHUNK_OVERLAP_TOKENS)
        total_chunks = len(text_chunks)

        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = _make_chunk_id(jurisdiction, doc_type, section_id, idx)
            # Ensure uniqueness if the same section ID appears multiple times
            # (e.g. continuation lines parsed as separate sections)
            if chunk_id in seen_ids:
                seen_ids[chunk_id] += 1
                chunk_id = f"{chunk_id}_d{seen_ids[chunk_id]}"
            else:
                seen_ids[chunk_id] = 0
            full_text = header + chunk_text

            all_chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=section_id,
                jurisdiction=jurisdiction,
                doc_type=doc_type,
                section_id=section_id,
                title=title,
                text=full_text,
                source_file=source_file,
                page_numbers=page_numbers,
                chunk_index=idx,
                total_chunks=total_chunks,
                parent_section_id=section_id,
            ))

    logger.info(
        "Chunked %d sections into %d chunks (max_tokens=%d, overlap=%d)",
        len(raw_sections), len(all_chunks), MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS,
    )
    return all_chunks


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Quick sanity check: parse everything and chunk it
    # add project root to path so src.* imports resolve
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from src.pdf_parser import parse_all

    raw = parse_all()
    chunks = chunk_sections(raw)

    print(f"\n{'='*60}")
    print(f"Total raw sections: {len(raw)}")
    print(f"Total chunks:       {len(chunks)}")
    print(f"{'='*60}\n")

    # Show a few sample chunks
    for c in chunks[:5]:
        tokens = _count_tokens(c.text)
        print(f"[{c.chunk_id}] tokens={tokens} chunk {c.chunk_index+1}/{c.total_chunks}")
        print(f"  {c.text[:100]}...")
        print()

    # Distribution stats
    token_counts = [_count_tokens(c.text) for c in chunks]
    print(f"Token stats: min={min(token_counts)}, max={max(token_counts)}, "
          f"mean={sum(token_counts)/len(token_counts):.0f}")
    over_limit = sum(1 for t in token_counts if t > MAX_CHUNK_TOKENS)
    if over_limit:
        print(f"WARNING: {over_limit} chunks exceed {MAX_CHUNK_TOKENS} tokens")
