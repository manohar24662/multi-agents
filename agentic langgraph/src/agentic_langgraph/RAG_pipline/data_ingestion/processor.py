"""Future-facing entry point for loading, splitting, and indexing documents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_documents
from .loader import load_documents
from ..services.retrieval.qdrant_store import collection_name, index_chunks


@dataclass(frozen=True)
class IngestionReport:
    """A concise result returned after an explicit ingestion run."""

    documents_loaded: int
    chunks_indexed: int
    collection: str


def ingest_documents(
    path: str | Path,
    *,
    min_chunk_tokens: int = 400,
    max_chunk_tokens: int = 512,
) -> IngestionReport:
    """Load supported files and write their chunks to Qdrant.

    This function is never called by the agent workflow automatically. It runs only
    when you explicitly ingest a file or a folder in the future.
    """
    documents = load_documents(path)
    chunks = chunk_documents(
        documents,
        min_chunk_tokens=min_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
    )
    return IngestionReport(
        documents_loaded=len(documents),
        chunks_indexed=index_chunks(chunks),
        collection=collection_name(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local documents into Qdrant.")
    parser.add_argument("path", help="A .txt, .md, .pdf file or a folder containing them")
    parser.add_argument("--min-chunk-tokens", type=int, default=400)
    parser.add_argument("--max-chunk-tokens", type=int, default=512)
    args = parser.parse_args()
    report = ingest_documents(
        args.path,
        min_chunk_tokens=args.min_chunk_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
    )
    print(
        f"Indexed {report.chunks_indexed} chunks from {report.documents_loaded} "
        f"document sections into Qdrant collection '{report.collection}'."
    )


if __name__ == "__main__":
    main()
