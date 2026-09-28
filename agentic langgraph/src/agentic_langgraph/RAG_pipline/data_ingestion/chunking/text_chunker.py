
"""Markdown-aware semantic chunking for document retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ...services.retrieval.embeddings import embed_texts, token_counts
from ..loader.document_loader import SourceDocument

DEFAULT_MIN_CHUNK_TOKENS = 400
DEFAULT_MAX_CHUNK_TOKENS = 512

_ATX = re.compile(r"^[ \t]{0,3}(?P<marks>#{1,6})[ \t]+(?P<text>.*?)[ \t]*$")
_SETEXT = re.compile(r"^[ \t]{0,3}(?P<marks>=+|-+)[ \t]*$")
_SENTENCE = re.compile(r"(?<=[.!?])(?:[\"'\)\]]+)?\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")
_FRONT_MATTER = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_YAML_TITLE = re.compile(r"^title:\s*[\"']?(?P<title>.+?)[\"']?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TextChunk:
    """One retrievable, source-preserving section of a document."""

    text: str
    source: str
    chunk_index: int
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class _Heading:
    start: int
    content_start: int
    level: int
    text: str


@dataclass(frozen=True)
class _Section:
    text: str
    heading: str


@dataclass(frozen=True)
class _Unit:
    text: str
    tokens: int


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def _find_headings(text: str) -> list[_Heading]:
    """Find ATX and Setext headings while ignoring fenced-code examples."""
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    headings: list[_Heading] = []
    in_fence = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith((chr(96) * 3, "~~~")):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue

        atx = _ATX.match(line)
        if atx:
            title = re.sub(r"\s+#+\s*$", "", atx.group("text")).strip()
            if title:
                headings.append(
                    _Heading(
                        offsets[index],
                        offsets[index] + len(line),
                        len(atx.group("marks")),
                        title,
                    )
                )
            index += 1
            continue

        if index + 1 < len(lines) and line.strip():
            setext = _SETEXT.match(lines[index + 1])
            if setext:
                title = line.strip()
                headings.append(
                    _Heading(
                        offsets[index],
                        offsets[index + 1] + len(lines[index + 1]),
                        1 if setext.group("marks").startswith("=") else 2,
                        title,
                    )
                )
                index += 2
                continue
        index += 1
    return headings


def _front_matter(text: str) -> tuple[str | None, int]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return None, 0
    title = _YAML_TITLE.search(match.group("body"))
    return (title.group("title").strip() if title else None), match.end()


def _default_title(document: SourceDocument) -> str:
    stem = Path(document.source).stem.replace("_", " ").replace("-", " ").strip()
    return stem.title() if stem else "Untitled document"


def _markdown_sections(document: SourceDocument) -> tuple[str, list[_Section]]:
    """Split Markdown into direct heading sections with heading paths."""
    source = document.text
    front_matter_title, content_start = _front_matter(source)
    headings = _find_headings(source)
    title = next((item.text for item in headings if item.level == 1), None)
    title = title or front_matter_title or _default_title(document)

    if not headings:
        body = _clean(source[content_start:])
        return title, [_Section(body, title)] if body else []

    sections: list[_Section] = []
    preface = _clean(source[content_start : headings[0].start])
    if preface:
        sections.append(_Section(preface, "Introduction"))

    path: list[str] = []
    for index, heading in enumerate(headings):
        while len(path) >= heading.level:
            path.pop()
        path.append(heading.text)
        end = headings[index + 1].start if index + 1 < len(headings) else len(source)
        body = _clean(source[heading.content_start:end])
        if body:
            sections.append(_Section(body, " > ".join(path)))
    return title, sections


def _document_sections(document: SourceDocument) -> tuple[str, list[_Section]]:
    file_type = str(document.metadata.get("file_type", "")).lower()
    if file_type in {"md", "markdown"}:
        return _markdown_sections(document)

    text = _clean(document.text)
    title = _default_title(document)
    heading = f"Page {document.metadata['page']}" if "page" in document.metadata else title
    return title, [_Section(text, heading)] if text else []


def _sentences(text: str) -> list[str]:
    """Recursively split via paragraphs, then sentence boundaries."""
    output: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text.strip()):
        paragraph = _clean(paragraph)
        if paragraph:
            output.extend(item for item in map(_clean, _SENTENCE.split(paragraph)) if item)
    return output


def _split_words(text: str, *, maximum: int) -> list[str]:
    """Final fallback for an individual sentence above the token maximum."""
    chunks: list[str] = []
    current: list[str] = []
    for word in text.split():
        if not current and token_counts([word])[0] > maximum:
            chunks.extend(_split_long_word(word, maximum=maximum))
            continue
        candidate = " ".join([*current, word])
        if current and token_counts([candidate])[0] > maximum:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    return [*chunks, " ".join(current)] if current else chunks


def _split_long_word(word: str, *, maximum: int) -> list[str]:
    """Split a token-heavy uninterrupted string, such as minified code or a URL."""
    pieces: list[str] = []
    start = 0
    while start < len(word):
        low, high, best = start + 1, len(word), start
        while low <= high:
            middle = (low + high) // 2
            if token_counts([word[start:middle]])[0] <= maximum:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        # A single character cannot normally exceed a positive token maximum,
        # but advance defensively to guarantee progress with unusual tokenizers.
        best = max(best, start + 1)
        pieces.append(word[start:best])
        start = best
    return pieces


def _reduce_unit(text: str, *, maximum: int) -> list[str]:
    """Recursively reduce an oversize sentence through clauses, then words."""
    if token_counts([text])[0] <= maximum:
        return [text]
    clauses = [item for item in map(_clean, _CLAUSE.split(text)) if item]
    if len(clauses) > 1:
        return [piece for clause in clauses for piece in _reduce_unit(clause, maximum=maximum)]
    return _split_words(text, maximum=maximum)


def _units(text: str, *, maximum: int) -> list[_Unit]:
    fragments = [
        part
        for sentence in (_sentences(text) or [_clean(text)])
        for part in _reduce_unit(sentence, maximum=maximum)
        if part
    ]
    counts = token_counts(fragments)
    return [_Unit(text=part, tokens=count) for part, count in zip(fragments, counts, strict=True)]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine safely for normalized production and test vectors."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 1.0


def _strict_partition_tails(
    units: Sequence[_Unit],
    *,
    minimum: int,
    maximum: int,
) -> list[bool]:
    """Mark positions from which the remaining units fit entirely in range."""
    possible = [False] * (len(units) + 1)
    possible[-1] = True
    for start in range(len(units) - 1, -1, -1):
        running = 0
        for end in range(start + 1, len(units) + 1):
            running += units[end - 1].tokens
            if running > maximum:
                break
            if running >= minimum and possible[end]:
                possible[start] = True
                break
    return possible


def _boundary(
    units: Sequence[_Unit],
    vectors: Sequence[Sequence[float]],
    *,
    start: int,
    minimum: int,
    maximum: int,
    strict_tails: Sequence[bool] | None,
) -> int:
    """Choose the least-similar valid sentence boundary."""
    running = 0
    target = (minimum + maximum) / 2
    candidates: list[tuple[float, int]] = []
    for end in range(start + 1, len(units) + 1):
        running += units[end - 1].tokens
        if running > maximum:
            break
        if running < minimum:
            continue
        if end == len(units):
            return end
        if strict_tails is not None and not strict_tails[end]:
            continue

        similarity = _cosine(vectors[end - 1], vectors[end])
        size_penalty = abs(running - target) / maximum
        tail_tokens = sum(unit.tokens for unit in units[end:])
        tail_penalty = 0.15 if 0 < tail_tokens < minimum else 0.0
        candidates.append((similarity + 0.05 * size_penalty + tail_penalty, end))

    if candidates:
        return min(candidates, key=lambda candidate: candidate[0])[1]

    # No all-in-range partition is possible; keep the largest valid full units.
    valid = [
        end
        for end in range(start + 1, len(units) + 1)
        if sum(unit.tokens for unit in units[start:end]) <= maximum
    ]
    return valid[-1]


def _semantic_chunks(text: str, *, minimum: int, maximum: int) -> list[str]:
    units = _units(text, maximum=maximum)
    if not units:
        return []
    if sum(unit.tokens for unit in units) <= maximum:
        return [_clean(text)]

    vectors = embed_texts([unit.text for unit in units])
    if len(vectors) != len(units):
        raise RuntimeError("The embedding model did not return one vector per text unit.")

    chunks: list[str] = []
    strict_tails = _strict_partition_tails(
        units,
        minimum=minimum,
        maximum=maximum,
    )
    allowed_tails = strict_tails if strict_tails[0] else None
    start = 0
    while start < len(units):
        end = _boundary(
            units,
            vectors,
            start=start,
            minimum=minimum,
            maximum=maximum,
            strict_tails=allowed_tails,
        )
        chunks.append(" ".join(unit.text for unit in units[start:end]))
        start = end
    return chunks


def chunk_documents(
    documents: list[SourceDocument],
    *,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
) -> list[TextChunk]:
    """Split documents by headings and semantic 400--512-token subchunks.

    A shorter final chunk is kept only where a heading section is short or no
    all-in-range partition is possible. index_chunks embeds the completed chunks
    immediately before upserting them into Qdrant.
    """
    if min_chunk_tokens < 1:
        raise ValueError("min_chunk_tokens must be positive.")
    if max_chunk_tokens < min_chunk_tokens:
        raise ValueError("max_chunk_tokens must be at least min_chunk_tokens.")

    chunks: list[TextChunk] = []
    for document in documents:
        title, sections = _document_sections(document)
        page = document.metadata.get("page", 1)
        document_chunk_index = 0
        for section in sections:
            for text in _semantic_chunks(
                section.text,
                minimum=min_chunk_tokens,
                maximum=max_chunk_tokens,
            ):
                index = document_chunk_index
                document_chunk_index += 1
                chunks.append(
                    TextChunk(
                        text=text,
                        source=document.source,
                        chunk_index=index,
                        metadata={
                            **document.metadata,
                            "source": document.source,
                            "title": title,
                            "heading": section.heading,
                            "page": page,
                            "chunk_index": index,
                        },
                    )
                )
    return chunks
