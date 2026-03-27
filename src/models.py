"""
Shared data models used across the entire pipeline.
Every module that produces or consumes chunks talks through these classes.
"""

from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """Single chunk of a legal document, ready for embedding and retrieval."""

    chunk_id: str            # unique id: "{jurisdiction}_{doc_type}_{section_id}_{chunk_idx}"
    doc_id: str              # parent document/section identifier
    jurisdiction: str        # "alaska" | "hawaii"
    doc_type: str            # "admin_code" | "statute"
    section_id: str          # e.g. "3 AAC 26.080" or "§431:10A-102"
    title: str               # section title
    text: str                # the actual chunk text
    source_file: str         # pdf filename
    page_numbers: list[int] = field(default_factory=list)
    chunk_index: int = 0     # position within parent section (0 = first/only chunk)
    total_chunks: int = 1    # how many chunks the parent section was split into
    parent_section_id: str = ""  # for sub-chunks: the full section id they belong to
    metadata: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        """Flatten to a ChromaDB-safe metadata dict (no nested objects, no booleans)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "jurisdiction": self.jurisdiction,
            "doc_type": self.doc_type,
            "section_id": self.section_id,
            "title": self.title,
            "source_file": self.source_file,
            "page_numbers": ",".join(str(p) for p in self.page_numbers),
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "parent_section_id": self.parent_section_id,
        }

    @classmethod
    def from_metadata_dict(cls, text: str, meta: dict) -> "DocumentChunk":
        """Reconstruct from a ChromaDB metadata dict + document text."""
        pages = [int(p) for p in meta.get("page_numbers", "").split(",") if p]
        return cls(
            chunk_id=meta["chunk_id"],
            doc_id=meta["doc_id"],
            jurisdiction=meta["jurisdiction"],
            doc_type=meta["doc_type"],
            section_id=meta["section_id"],
            title=meta["title"],
            text=text,
            source_file=meta["source_file"],
            page_numbers=pages,
            chunk_index=meta.get("chunk_index", 0),
            total_chunks=meta.get("total_chunks", 1),
            parent_section_id=meta.get("parent_section_id", ""),
        )


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its relevance score and origin."""

    chunk: DocumentChunk
    score: float
    source: str   # "dense" | "sparse" | "fused" | "reranked"
    rank: int = 0


@dataclass
class AgentResponse:
    """Final output from the agentic RAG pipeline."""

    answer: str
    citations: list[dict] = field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    query_reformulations: list[str] = field(default_factory=list)
    iterations_used: int = 1
    is_multi_hop: bool = False
    sub_questions: list[str] = field(default_factory=list)
    hop_summaries: list[str] = field(default_factory=list)
