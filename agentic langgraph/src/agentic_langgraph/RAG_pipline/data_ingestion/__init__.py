"""Load, split, and index private documents."""

__all__ = ["IngestionReport", "ingest_documents"]


def __getattr__(name: str):
    """Avoid importing the CLI module before ``python -m ...processor`` runs it."""
    if name == "IngestionReport":
        from .processor import IngestionReport

        return IngestionReport
    if name == "ingest_documents":
        from .processor import ingest_documents

        return ingest_documents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
