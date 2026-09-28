"""Lazy local sentence-transformer embeddings used by the Qdrant store."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence


@lru_cache
def get_embedding_model():
    """Load the embedding model only when data is indexed or queried."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        # This default accepts the 512-token upper chunk limit. all-MiniLM-L6-v2
        # silently truncates at 256 tokens, making it unsuitable for this pipeline.
        os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        device=os.getenv("RAG_EMBEDDING_DEVICE", "cpu"),
    )


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed texts with normalized vectors suitable for cosine similarity."""
    if not texts:
        return []
    vectors = get_embedding_model().encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def token_counts(texts: Sequence[str]) -> list[int]:
    """Count embedding-model tokens without special tokens for chunk limits."""
    if not texts:
        return []
    tokenizer = getattr(get_embedding_model(), "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The embedding model does not expose a tokenizer.")
    encoded = tokenizer(list(texts), add_special_tokens=False, truncation=False)
    return [len(token_ids) for token_ids in encoded["input_ids"]]


def embedding_dimension() -> int:
    """Return the vector size of the selected model."""
    dimension = get_embedding_model().get_sentence_embedding_dimension()
    if not dimension:
        raise RuntimeError("The embedding model did not report a vector dimension.")
    return int(dimension)
