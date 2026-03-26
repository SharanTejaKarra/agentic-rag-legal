"""
PDF text extraction and section boundary detection for legal documents.
Handles Alaska Admin Code, Hawaii Admin Code, and Hawaii Revised Statutes.
"""

import re
import logging
from pathlib import Path

import pdfplumber

from src.config import ALASKA_DIR, HAWAII_ADMIN_DIR, HAWAII_STATUTES_DIR, DATA_DIR

logger = logging.getLogger(__name__)

# ── noise patterns ────────────────────────────────────────────────────
# Timestamps like "7/29/25, 12:15 PM Alaska Admin Code" at the top of every page
_ALASKA_HEADER_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M\s+Alaska Admin Code\s*$",
    re.MULTILINE,
)
# URL footers at the bottom of Alaska pages
_ALASKA_FOOTER_RE = re.compile(
    r"^https?://www\.akleg\.gov/.*$", re.MULTILINE
)
# Page number footers like "36/36" at the very end of a page
_PAGE_NUM_FOOTER_RE = re.compile(r"\s*\d+/\d+\s*$")

# Hawaii statutes: timestamp header, nav links, URL footer
_HAWAII_STAT_HEADER_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M\s+\S+.*$",
    re.MULTILINE,
)
_HAWAII_STAT_NAV_RE = re.compile(
    r"(Previous|Next)\s+", re.MULTILINE
)
_HAWAII_STAT_VOL_RE = re.compile(
    r"Vol\d+_Ch\d+[-\w]*", re.MULTILINE
)
_HAWAII_STAT_URL_RE = re.compile(
    r"^https?://www\.capitol\.hawaii\.gov/.*$", re.MULTILINE
)
# "PART II." or similar part header lines that show up standalone
_HAWAII_STAT_PART_HEADER_RE = re.compile(
    r"^PART\s+[IVXLC]+\..*$", re.MULTILINE
)

# Hawaii admin code: bare page numbers at the end of a page, e.g. "1-1", "5-3"
_HAR_PAGE_NUM_RE = re.compile(r"^\d+-\d+\s*$", re.MULTILINE)

# Section header patterns
_ALASKA_SECTION_RE = re.compile(
    r"^(3\s+AAC\s+\d+\.\d+)\.\s+(.+?)$", re.MULTILINE
)
_HAWAII_ADMIN_SECTION_RE = re.compile(
    r"^§(16-\d+-\d+)\s+([A-Z].+?)$", re.MULTILINE
)
_HAWAII_STATUTE_SECTION_RE = re.compile(
    r"^\[?§(431[:\d\w.-]+)\]?\s+(.+?)$", re.MULTILINE
)

# ── text cleaning ─────────────────────────────────────────────────────

def _clean_alaska_page(text: str) -> str:
    """Strip Alaska-specific headers, footers, and URLs from a page."""
    text = _ALASKA_HEADER_RE.sub("", text)
    text = _ALASKA_FOOTER_RE.sub("", text)
    text = _PAGE_NUM_FOOTER_RE.sub("", text)
    return text.strip()


def _clean_hawaii_statute_page(text: str) -> str:
    """Strip nav links, URL footers, timestamps from Hawaii statute pages."""
    # Normalize unicode dashes (en-dash, non-breaking hyphen) to regular hyphens
    text = text.replace("\u2011", "-").replace("\u2013", "-")
    text = _HAWAII_STAT_HEADER_RE.sub("", text)
    text = _HAWAII_STAT_NAV_RE.sub("", text)
    text = _HAWAII_STAT_VOL_RE.sub("", text)
    text = _HAWAII_STAT_URL_RE.sub("", text)
    text = _PAGE_NUM_FOOTER_RE.sub("", text)
    return text.strip()


def _clean_hawaii_admin_page(text: str) -> str:
    """Strip page numbers and repeated title headers from Hawaii admin code pages."""
    text = _HAR_PAGE_NUM_RE.sub("", text)
    return text.strip()


# ── Alaska Admin Code parser ──────────────────────────────────────────

def _is_article_or_chapter_line(line: str) -> bool:
    """Check if a line is a standalone Article/Chapter header (not section content)."""
    stripped = line.strip()
    if re.match(r"^(Article|Chapter)\s+\d+", stripped, re.IGNORECASE):
        return True
    # All-caps multi-word lines that aren't section bodies -- these are article titles
    if stripped.isupper() and len(stripped.split()) >= 2 and not stripped.startswith("3 AAC"):
        return True
    return False


def parse_alaska_pdf(pdf_path: Path) -> list[dict]:
    """
    Parse an Alaska Administrative Code PDF into sections.
    Sections follow the pattern: 3 AAC XX.XXX. Title.
    Returns a list of dicts with section_id, title, text, page_numbers,
    is_repealed, source_file.
    """
    pdf_path = Path(pdf_path)
    sections: list[dict] = []

    # First pass: extract and clean all page text
    pages_text: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            pages_text.append(_clean_alaska_page(raw))

    # Combine all pages, keeping track of page boundaries so we can map
    # sections back to page numbers.
    page_char_offsets: list[tuple[int, int]] = []  # (start, end) char offset per page
    full_text = ""
    for i, pt in enumerate(pages_text):
        start = len(full_text)
        full_text += pt + "\n"
        page_char_offsets.append((start, len(full_text)))

    def _pages_for_span(start: int, end: int) -> list[int]:
        """Return 1-indexed page numbers that overlap [start, end)."""
        result = []
        for page_idx, (ps, pe) in enumerate(page_char_offsets):
            if ps < end and pe > start:
                result.append(page_idx + 1)
        return result

    # Find all section headers
    matches = list(_ALASKA_SECTION_RE.finditer(full_text))
    if not matches:
        logger.warning("No sections found in %s", pdf_path.name)
        return sections

    for i, m in enumerate(matches):
        section_id = m.group(1).strip()
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        body = full_text[body_start:body_end].strip()

        # Filter out article/chapter header lines that bleed into the body
        body_lines = []
        for line in body.split("\n"):
            if _is_article_or_chapter_line(line):
                continue
            body_lines.append(line)
        body = "\n".join(body_lines).strip()

        # Detect repealed sections -- body is just "Repealed" optionally with a date
        is_repealed = bool(re.match(r"^Repealed\.?(\s+\d+/\d+/\d+\.?)?$", body, re.IGNORECASE))

        page_numbers = _pages_for_span(m.start(), body_end)

        sections.append({
            "section_id": section_id,
            "title": title,
            "text": body,
            "page_numbers": page_numbers,
            "is_repealed": is_repealed,
            "source_file": pdf_path.name,
            "jurisdiction": "alaska",
            "doc_type": "admin_code",
        })

    logger.info("Parsed %d sections from %s", len(sections), pdf_path.name)
    return sections


# ── Hawaii Admin Code parser ──────────────────────────────────────────

def _strip_hawaii_admin_toc(full_text: str) -> str:
    """
    Remove the table-of-contents block that appears at the start of Hawaii
    admin code PDFs. The TOC entries look like:
      §16-5-1 Purpose
    while real section bodies look like:
      §16-5-1 Purpose. The purpose of this chapter is to...
    We find the first section header that's followed by actual body text
    (a period and then more content on the same or next line).
    """
    # Match section headers that are followed by a period + body text
    # (not just a bare title line in the TOC)
    body_section = re.search(
        r"§16-\d+-\d+\s+[^.\n]+\.\s+\S",
        full_text,
    )
    if body_section:
        # Back up to the start of the § symbol on that line
        line_start = full_text.rfind("\n", 0, body_section.start())
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1
        return full_text[line_start:]

    # Fallback: if we can't find a body-style section, look for Historical Note
    # as a TOC/body boundary
    hist = re.search(r"Historical Note:", full_text, re.IGNORECASE)
    if hist:
        # Find the next §16- after the historical note
        after_hist = re.search(r"§16-\d+-\d+\s+", full_text[hist.end():])
        if after_hist:
            return full_text[hist.end() + after_hist.start():]

    return full_text


def parse_hawaii_admin_pdf(pdf_path: Path) -> list[dict]:
    """
    Parse a Hawaii Administrative Rules (HAR) chapter PDF.
    Sections follow the pattern: §16-X-Y Title.
    Auth/Imp citations at the end of each section are kept as metadata context.
    """
    pdf_path = Path(pdf_path)
    sections: list[dict] = []

    pages_text: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            pages_text.append(_clean_hawaii_admin_page(raw))

    page_char_offsets: list[tuple[int, int]] = []
    full_text = ""
    for i, pt in enumerate(pages_text):
        start = len(full_text)
        full_text += pt + "\n"
        page_char_offsets.append((start, len(full_text)))

    def _pages_for_span(start: int, end: int) -> list[int]:
        result = []
        for page_idx, (ps, pe) in enumerate(page_char_offsets):
            if ps < end and pe > start:
                result.append(page_idx + 1)
        return result

    # Strip the table of contents / preamble
    full_text_body = _strip_hawaii_admin_toc(full_text)
    # Adjust offset: the body may start partway into the full text
    toc_offset = len(full_text) - len(full_text_body)

    matches = list(_HAWAII_ADMIN_SECTION_RE.finditer(full_text_body))
    if not matches:
        logger.warning("No sections found in %s", pdf_path.name)
        return sections

    for i, m in enumerate(matches):
        section_id = "§" + m.group(1).strip()
        title = m.group(2).strip().rstrip(".")
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text_body)

        body = full_text_body[body_start:body_end].strip()

        # Filter out subchapter headers (all-caps lines like "SUBCHAPTER 2")
        body_lines = []
        for line in body.split("\n"):
            stripped = line.strip()
            if re.match(r"^SUBCHAPTER\s+\d+", stripped, re.IGNORECASE):
                continue
            if stripped.isupper() and len(stripped.split()) >= 2 and not stripped.startswith("§"):
                continue
            body_lines.append(line)
        body = "\n".join(body_lines).strip()

        # Absolute char positions in the original full_text (for page mapping)
        abs_start = m.start() + toc_offset
        abs_end = body_end + toc_offset
        page_numbers = _pages_for_span(abs_start, abs_end)

        sections.append({
            "section_id": section_id,
            "title": title,
            "text": body,
            "page_numbers": page_numbers,
            "is_repealed": False,
            "source_file": pdf_path.name,
            "jurisdiction": "hawaii",
            "doc_type": "admin_code",
        })

    logger.info("Parsed %d sections from %s", len(sections), pdf_path.name)
    return sections


# ── Hawaii Revised Statutes parser ────────────────────────────────────

def parse_hawaii_statute_pdf(pdf_path: Path) -> list[dict]:
    """
    Parse a Hawaii Revised Statutes PDF. Usually one section per file.
    Section pattern: §431:XXX-XXX Title.
    """
    pdf_path = Path(pdf_path)
    sections: list[dict] = []

    pages_text: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            pages_text.append(_clean_hawaii_statute_page(raw))

    full_text = "\n".join(pages_text).strip()

    # Also strip standalone PART headers (e.g. "PART II. GROUP AND BLANKET...")
    full_text = _HAWAII_STAT_PART_HEADER_RE.sub("", full_text).strip()

    # Strip Attorney General Opinions blocks that precede the actual section
    # They show up like "Attorney General Opinions\nSection 431:10A-601 applied..."
    ag_match = re.search(r"Attorney General Opinions\s*\n.*?\n(?=§)", full_text, re.DOTALL)
    if ag_match:
        full_text = full_text[:ag_match.start()] + full_text[ag_match.end():]
        full_text = full_text.strip()

    # Find the section header
    matches = list(_HAWAII_STATUTE_SECTION_RE.finditer(full_text))
    if not matches:
        logger.warning("No section header found in %s", pdf_path.name)
        return sections

    for i, m in enumerate(matches):
        raw_id = m.group(1).strip()
        section_id = "§" + raw_id
        title = m.group(2).strip()

        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[body_start:body_end].strip()

        # The title often runs into the first line of body text. The header regex
        # captures until end of line, and the period-terminated title is on the
        # same line as the start of body. We need to separate them.
        # If the title doesn't end with a period, the rest of the first line is
        # still part of the title or a continuation.
        if not title.endswith("."):
            # Title continues -- grab text up to the first period that looks like
            # end of a title (followed by a space and body text or newline)
            combined = title + " " + body
            # Find first sentence-ending period followed by body content
            title_end = re.search(r"\.(?:\s|$)", combined)
            if title_end:
                title = combined[:title_end.end()].strip()
                body = combined[title_end.end():].strip()
            else:
                title = combined.strip()
                body = ""

        # Strip trailing legislative history: [L 1987, c 347, pt of §2; ...]
        body = re.sub(r"\[L\s+\d{4},.*$", "", body, flags=re.DOTALL).strip()

        # Also strip any trailing nav/url remnants that survived earlier cleaning
        body = re.sub(r"(Previous|Next|Vol\d+_Ch\d+[-\w]*)\s*$", "", body).strip()

        page_numbers = list(range(1, len(pages_text) + 1))

        sections.append({
            "section_id": section_id,
            "title": title,
            "text": body,
            "page_numbers": page_numbers,
            "is_repealed": False,
            "source_file": pdf_path.name,
            "jurisdiction": "hawaii",
            "doc_type": "statute",
        })

    logger.info("Parsed %d sections from %s", len(sections), pdf_path.name)
    return sections


# ── main entry point ──────────────────────────────────────────────────

def parse_all(data_dir: Path | None = None) -> list[dict]:
    """
    Discover and parse all PDFs under the data directory.
    Auto-detects the format based on which subdirectory the file lives in.
    Returns a flat list of all parsed sections.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    all_sections: list[dict] = []

    # Alaska
    alaska_dir = data_dir / "alaska"
    if alaska_dir.exists():
        for pdf_file in sorted(alaska_dir.glob("*.pdf")):
            try:
                all_sections.extend(parse_alaska_pdf(pdf_file))
            except Exception:
                logger.exception("Failed to parse Alaska PDF: %s", pdf_file.name)

    # Hawaii admin code
    hawaii_admin_dir = data_dir / "hawaii" / "admin_code"
    if hawaii_admin_dir.exists():
        for pdf_file in sorted(hawaii_admin_dir.glob("*.pdf")):
            try:
                all_sections.extend(parse_hawaii_admin_pdf(pdf_file))
            except Exception:
                logger.exception("Failed to parse Hawaii admin PDF: %s", pdf_file.name)

    # Hawaii statutes
    hawaii_stat_dir = data_dir / "hawaii" / "statutes"
    if hawaii_stat_dir.exists():
        for pdf_file in sorted(hawaii_stat_dir.glob("*.pdf")):
            try:
                all_sections.extend(parse_hawaii_statute_pdf(pdf_file))
            except Exception:
                logger.exception("Failed to parse Hawaii statute PDF: %s", pdf_file.name)

    logger.info(
        "Total sections parsed: %d (Alaska: %d, Hawaii admin: %d, Hawaii statute: %d)",
        len(all_sections),
        sum(1 for s in all_sections if s["jurisdiction"] == "alaska"),
        sum(1 for s in all_sections if s["doc_type"] == "admin_code" and s["jurisdiction"] == "hawaii"),
        sum(1 for s in all_sections if s["doc_type"] == "statute"),
    )
    return all_sections


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sections = parse_all()
    for s in sections[:5]:
        print(f"[{s['jurisdiction']}/{s['doc_type']}] {s['section_id']} — {s['title']}")
        print(f"  repealed={s['is_repealed']}, pages={s['page_numbers']}, text_len={len(s['text'])}")
        print(f"  text preview: {s['text'][:120]}...")
        print()
    print(f"--- Total: {len(sections)} sections ---")
    repealed = sum(1 for s in sections if s["is_repealed"])
    print(f"--- Repealed: {repealed} ---")
