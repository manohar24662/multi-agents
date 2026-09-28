
"""Unit tests for Markdown-aware chunking without loading an embedding model."""

from unittest import TestCase
from unittest.mock import patch

from agentic_langgraph.RAG_pipline.data_ingestion.chunking import chunk_documents
from agentic_langgraph.RAG_pipline.data_ingestion.loader import SourceDocument


def _word_token_counts(texts: list[str]) -> list[int]:
    return [len(text.split()) for text in texts]


def _test_vectors(texts: list[str]) -> list[list[float]]:
    return [[float(index % 2), float((index + 1) % 2)] for index, _ in enumerate(texts)]


class MarkdownChunkingTests(TestCase):
    @patch(
        "agentic_langgraph.RAG_pipline.data_ingestion.chunking.text_chunker.embed_texts",
        side_effect=_test_vectors,
    )
    @patch(
        "agentic_langgraph.RAG_pipline.data_ingestion.chunking.text_chunker.token_counts",
        side_effect=_word_token_counts,
    )
    def test_markdown_heading_metadata_and_semantic_chunking(
        self,
        token_counts_mock,
        embeddings_mock,
    ):
        sentence = " ".join(["context"] * 98) + "."
        document = SourceDocument(
            text=(
                "# Retrieval Guide\n\n"
                f"{sentence} {sentence} {sentence} {sentence} {sentence} {sentence}\n\n"
                "## Vector Search\n\n"
                f"{sentence} {sentence} {sentence} {sentence} {sentence} {sentence}"
            ),
            source="docs/retrieval-guide.md",
            metadata={"file_type": "md"},
        )

        chunks = chunk_documents([document])

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(embeddings_mock.called)
        self.assertTrue(token_counts_mock.called)
        self.assertEqual({chunk.metadata["title"] for chunk in chunks}, {"Retrieval Guide"})
        self.assertEqual({chunk.metadata["source"] for chunk in chunks}, {document.source})
        self.assertEqual({chunk.metadata["page"] for chunk in chunks}, {1})
        self.assertIn(
            "Retrieval Guide > Vector Search",
            {chunk.metadata["heading"] for chunk in chunks},
        )
        self.assertTrue(all(_word_token_counts([chunk.text])[0] <= 512 for chunk in chunks))
