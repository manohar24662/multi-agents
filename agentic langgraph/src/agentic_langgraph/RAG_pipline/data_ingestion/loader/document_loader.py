"""Load local text, Markdown, and PDF files for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx"}


@dataclass(frozen=True)
class SourceDocument:
    """Text extracted from one source file or one page of a PDF."""

    text: str
    source: str
    metadata: dict[str, str | int]

def _load_docx(path: Path) -> list[SourceDocument]:
    try:
        from docx import Document
    except ImportError as error:
        raise RuntimeError(
            "DOCX ingestion needs python-docx. Run `uv add python-docx`."
        ) from error

    document = Document(path)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    text = "\n".join(paragraphs).strip()

    if not text:
        return []

    return [
        SourceDocument(
            text=text,
            source=str(path),
            metadata={
                "source": str(path),
                "file_type": "docx",
            },
        )
    ]

def _load_pdf(path: Path) -> list[SourceDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF ingestion needs pypdf. Run `uv sync` to install it.") from error

    documents: list[SourceDocument] = []
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                SourceDocument(
                    text=text,
                    source=str(path),
                    metadata={"source": str(path), "page": page_number, "file_type": "pdf"},
                )
            )
    return documents


def _load_file(path: Path) -> list[SourceDocument]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)

    if suffix == ".docx":
        return _load_docx(path)

    if suffix not in SUPPORTED_SUFFIXES:
        return []

    text = path.read_text(encoding="utf-8", errors="replace").strip()

    if not text:
        return []

    return [
        SourceDocument(
            text=text,
            source=str(path),
            metadata={
                "source": str(path),
                "file_type": suffix.lstrip("."),
            },
        )
    ]


def load_documents(path: str | Path) -> list[SourceDocument]:
    """Load all supported files from a file or directory, without uploading data."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Document path does not exist: {source_path}")

    if source_path.is_file():
        documents = _load_file(source_path)
    else:
        documents = []
        for file_path in sorted(source_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
                documents.extend(_load_file(file_path))

    if not documents:
        raise ValueError(
            "No readable .txt, .md, .markdown, or text-based .pdf documents were found."
        )
    return documents
