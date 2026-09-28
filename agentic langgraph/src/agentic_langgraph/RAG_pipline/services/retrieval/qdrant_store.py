"""Safe Qdrant Cloud storage and retrieval for private RAG documents."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from .embeddings import embed_texts, embedding_dimension

if TYPE_CHECKING:
    from agentic_langgraph.RAG_pipline.data_ingestion.chunking import TextChunk


load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk returned by vector similarity search."""

    text: str
    source: str
    score: float
    metadata: dict[str, str | int]


def collection_name() -> str:
    """Read the collection name without creating it."""
    return os.getenv("QDRANT_COLLECTION", "agentic_langgraph_documents")


def is_qdrant_configured() -> bool:
    """Whether the two required Qdrant Cloud settings are present."""
    return bool(os.getenv("QDRANT_CLUSTER_ENDPOINT") and os.getenv("QDRANT_API_KEY"))


def is_rag_enabled() -> bool:
    """Keep retrieval off until documents have been deliberately ingested."""
    return os.getenv("RAG_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Connect to the configured Qdrant Cloud cluster when it is actually needed."""
    endpoint = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    api_key = os.getenv("QDRANT_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError(
            "Qdrant is not configured. Add QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY to .env."
        )
    return QdrantClient(url=endpoint, api_key=api_key, timeout=20)


def _collection_exists(client: QdrantClient) -> bool:
    return client.collection_exists(collection_name())


def ensure_collection(vector_size: int) -> None:
    """Create the collection once, using cosine similarity for normalized embeddings."""
    client = get_qdrant_client()
    if not _collection_exists(client):
        client.create_collection(
            collection_name=collection_name(),
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def index_chunks(chunks: list[TextChunk]) -> int:
    """Embed and upsert chunks. Calling this is the only operation that writes to Qdrant."""
    if not chunks:
        return 0

    vectors = embed_texts([chunk.text for chunk in chunks])
    ensure_collection(embedding_dimension())
    points = [
        models.PointStruct(
            id=str(uuid5(NAMESPACE_URL, f"{chunk.source}:{chunk.metadata}:{chunk.chunk_index}")),
            vector=vector,
            payload={"text": chunk.text, "source": chunk.source, **chunk.metadata},
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    get_qdrant_client().upsert(
        collection_name=collection_name(),
        points=points,
        wait=True,
    )
    return len(points)


def retrieve_chunks(query: str, *, limit: int = 4) -> list[RetrievedChunk]:
    """Return relevant private-document chunks; empty is normal before ingestion."""
    if not query.strip() or not is_rag_enabled() or not is_qdrant_configured():
        return []

    try:
        client = get_qdrant_client()
        if not _collection_exists(client):
            return []
        query_vector = embed_texts([query])[0]
        response = client.query_points(
            collection_name=collection_name(),
            query=query_vector,
            limit=max(1, min(limit, 10)),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as error:
        # Retrieval must never stop the normal agent workflow if RAG is not ready.
        logger.warning("RAG retrieval was skipped: %s", type(error).__name__)
        return []

    chunks: list[RetrievedChunk] = []
    for point in response.points:
        payload = point.payload or {}
        text = str(payload.get("text", "")).strip()
        if text:
            metadata = {
                str(key): value
                for key, value in payload.items()
                if key not in {"text", "source"} and isinstance(value, (str, int))
            }
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(payload.get("source", "Unknown source")),
                    score=float(point.score),
                    metadata=metadata,
                )
            )
    return chunks


def format_retrieved_context(query: str, *, limit: int = 4) -> str:
    """Format retrieved chunks as clearly labelled, source-preserving LLM context."""
    chunks = retrieve_chunks(query, limit=limit)
    if not chunks:
        return ""
    sections = []
    for chunk in chunks:
        page = f", page {chunk.metadata['page']}" if "page" in chunk.metadata else ""
        sections.append(f"Source: {chunk.source}{page}\n{chunk.text}")
    return "\n\n---\n\n".join(sections)
