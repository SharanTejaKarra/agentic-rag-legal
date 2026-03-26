"""
Ingestion pipeline — processes PDFs, chunks them, and builds both indexes.
Run: python ingest.py [--force]
"""

import argparse
import logging
import sys
from collections import Counter

from src.config import DATA_DIR
from src.pdf_parser import parse_all
from src.chunker import chunk_sections
from src.embedder import embed_and_store
from src.bm25_index import build_bm25_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes from legal PDFs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe existing indexes and rebuild from scratch",
    )
    args = parser.parse_args()

    # step 1: parse PDFs into raw sections
    log.info("Parsing PDFs from %s ...", DATA_DIR)
    raw_sections = parse_all(DATA_DIR)

    if not raw_sections:
        log.error("No sections extracted — check that PDFs exist in %s", DATA_DIR)
        sys.exit(1)

    log.info("Extracted %d raw sections", len(raw_sections))

    # step 2: chunk sections into retrieval-sized pieces
    log.info("Chunking sections ...")
    chunks = chunk_sections(raw_sections)
    log.info("Produced %d chunks", len(chunks))

    # quick breakdown by jurisdiction
    jurisdiction_counts = Counter(c.jurisdiction for c in chunks)
    for jur, count in sorted(jurisdiction_counts.items()):
        log.info("  %-10s %d chunks", jur, count)

    doc_type_counts = Counter(c.doc_type for c in chunks)
    for dtype, count in sorted(doc_type_counts.items()):
        log.info("  %-10s %d chunks", dtype, count)

    # step 3: embed and push to ChromaDB
    log.info("Building dense index (ChromaDB) ...")
    embed_and_store(chunks, force_reset=args.force)

    # step 4: build BM25 sparse index
    log.info("Building sparse index (BM25) ...")
    build_bm25_index(chunks)

    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
