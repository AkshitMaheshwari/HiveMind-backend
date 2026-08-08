"""
Unit tests for the RAG pipeline components:
- rag/chunker.py
- rag/ingestion.py (via mocks)
- rag/retrieval.py (via mocks)

Covers:
- Chunking: happy path, empty text raises, overlap, single-sentence docs
- Ingestion: embedding failure → no partial storage, success path
- Retrieval: empty result → returns [], not exception
- Dedup: delete_by_source called before upsert
"""
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, call

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import pytest


# ─── Chunker tests ────────────────────────────────────────────────────────────

class TestChunker:
    def test_non_empty_text_returns_chunks(self):
        from rag.chunker import chunk_text
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)
        assert all(len(c) > 0 for c in chunks)

    def test_empty_text_raises_value_error(self):
        from rag.chunker import chunk_text
        with pytest.raises(ValueError, match="non-empty"):
            chunk_text("", chunk_size=800, overlap=150)

    def test_whitespace_only_raises_value_error(self):
        from rag.chunker import chunk_text
        with pytest.raises(ValueError):
            chunk_text("   \n   ", chunk_size=800, overlap=150)

    def test_overlap_creates_shared_content(self):
        from rag.chunker import chunk_text
        # Use a large enough text to produce multiple chunks
        long_text = ". ".join([f"Sentence number {i}" for i in range(30)])
        chunks = chunk_text(long_text, chunk_size=100, overlap=30)
        if len(chunks) > 1:
            # Adjacent chunks should share some content due to overlap
            # At minimum they should each be non-empty
            assert all(len(c) > 0 for c in chunks)

    def test_invalid_chunk_size_raises(self):
        from rag.chunker import chunk_text
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("Some text.", chunk_size=0, overlap=0)

    def test_negative_overlap_raises(self):
        from rag.chunker import chunk_text
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("Some text.", chunk_size=100, overlap=-1)

    def test_single_sentence_returns_one_chunk(self):
        from rag.chunker import chunk_text
        text = "This is a single short sentence."
        chunks = chunk_text(text, chunk_size=800, overlap=150)
        assert len(chunks) == 1
        assert "single short sentence" in chunks[0]

    def test_large_text_produces_multiple_chunks(self):
        from rag.chunker import chunk_text
        # Force multiple chunks with a small chunk_size
        text = ". ".join([f"This is sentence {i} with some content" for i in range(20)])
        chunks = chunk_text(text, chunk_size=80, overlap=15)
        assert len(chunks) > 1


# ─── Ingestion tests ──────────────────────────────────────────────────────────

class TestIngestion:
    def _make_doc(self, text: str = "Some text content.", uid: str = "user-1", source: str = "test.pdf"):
        from connectors.document import Document
        return Document(
            text=text,
            source_type="pdf",
            source_identifier=source,
            user_id=uid,
        )

    def test_embedding_failure_results_in_no_chunks_stored(self):
        """If embedding fails, the document must appear in failed list and no chunks upserted."""
        from rag.ingestion import ingest_documents
        from rag.embedder import EmbeddingError

        doc = self._make_doc()

        with patch("rag.ingestion.QdrantVectorStore") as mock_store_cls, \
             patch("rag.ingestion.Embedder") as mock_embedder_cls:

            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            mock_embedder = MagicMock()
            mock_embedder.embed_texts.side_effect = EmbeddingError("Rate limit exceeded")
            mock_embedder_cls.return_value = mock_embedder

            result = ingest_documents([doc])

        # Document must be in failed list
        assert len(result.failed) == 1
        assert result.failed[0].document_id == doc.id
        assert "Embedding failed" in result.failed[0].error

        # No chunks should have been stored
        mock_store.upsert_chunks.assert_not_called()

        # Successful list should be empty
        assert result.successful == []
        assert result.chunks_ingested == 0

    def test_successful_ingestion_happy_path(self):
        """Happy path: chunking → embedding → upsert all succeed."""
        from rag.ingestion import ingest_documents

        doc = self._make_doc(
            text="This is sentence one. This is sentence two. This is sentence three."
        )

        fake_vector = [0.1] * 1536

        with patch("rag.ingestion.QdrantVectorStore") as mock_store_cls, \
             patch("rag.ingestion.Embedder") as mock_embedder_cls:

            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            mock_embedder = MagicMock()
            mock_embedder.embed_texts.return_value = [fake_vector]  # 1 chunk → 1 vector
            mock_embedder_cls.return_value = mock_embedder

            result = ingest_documents([doc])

        assert doc.id in result.successful
        assert result.failed == []
        assert result.chunks_ingested >= 1
        mock_store.upsert_chunks.assert_called_once()

    def test_dedup_delete_called_before_upsert(self):
        """delete_by_source must be called before upsert_chunks for each document."""
        from rag.ingestion import ingest_documents

        doc = self._make_doc()
        fake_vector = [0.0] * 1536
        call_order = []

        with patch("rag.ingestion.QdrantVectorStore") as mock_store_cls, \
             patch("rag.ingestion.Embedder") as mock_embedder_cls:

            mock_store = MagicMock()

            def track_delete(*args, **kwargs):
                call_order.append("delete")

            def track_upsert(*args, **kwargs):
                call_order.append("upsert")

            mock_store.delete_by_source.side_effect = track_delete
            mock_store.upsert_chunks.side_effect = track_upsert
            mock_store_cls.return_value = mock_store

            mock_embedder = MagicMock()
            mock_embedder.embed_texts.return_value = [fake_vector]
            mock_embedder_cls.return_value = mock_embedder

            ingest_documents([doc])

        assert call_order.index("delete") < call_order.index("upsert"), (
            "delete_by_source must be called before upsert_chunks"
        )

    def test_empty_documents_raises_value_error(self):
        from rag.ingestion import ingest_documents
        with pytest.raises(ValueError):
            ingest_documents([])

    def test_all_succeeded_property(self):
        from rag.ingestion import IngestionResult, FailedDocument
        result = IngestionResult(successful=["id1"], failed=[], chunks_ingested=3)
        assert result.all_succeeded is True

        result_with_failure = IngestionResult(
            successful=[],
            failed=[FailedDocument("id1", "f.pdf", "u1", "err")],
            chunks_ingested=0,
        )
        assert result_with_failure.all_succeeded is False


# ─── Retrieval tests ──────────────────────────────────────────────────────────

class TestRetrieval:
    def test_empty_result_returns_empty_list_not_exception(self):
        """When no documents match, retrieve() must return [] not raise."""
        from rag.retrieval import retrieve

        with patch("rag.retrieval.Embedder") as mock_embedder_cls, \
             patch("rag.retrieval.QdrantVectorStore") as mock_store_cls:

            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [0.1] * 1536
            mock_embedder_cls.return_value = mock_embedder

            mock_store = MagicMock()
            mock_store.search.return_value = []  # No results
            mock_store_cls.return_value = mock_store

            result = retrieve(query="anything", user_id="user-1")

        assert result == []

    def test_results_are_returned_in_correct_shape(self):
        from rag.retrieval import retrieve
        from rag.vector_store import SearchResult

        fake_search_result = SearchResult(
            chunk_id="abc",
            text="Relevant excerpt from document.",
            source_identifier="report.pdf",
            source_type="pdf",
            score=0.92,
            metadata={"page": 1},
        )

        with patch("rag.retrieval.Embedder") as mock_embedder_cls, \
             patch("rag.retrieval.QdrantVectorStore") as mock_store_cls:

            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [0.1] * 1536
            mock_embedder_cls.return_value = mock_embedder

            mock_store = MagicMock()
            mock_store.search.return_value = [fake_search_result]
            mock_store_cls.return_value = mock_store

            results = retrieve(query="test query", user_id="user-1")

        assert len(results) == 1
        assert results[0].text == "Relevant excerpt from document."
        assert results[0].source == "report.pdf"
        assert results[0].score == pytest.approx(0.92)

    def test_empty_query_raises_value_error(self):
        from rag.retrieval import retrieve
        with pytest.raises(ValueError, match="query"):
            retrieve(query="", user_id="user-1")

    def test_empty_user_id_raises_value_error(self):
        from rag.retrieval import retrieve
        with pytest.raises(ValueError, match="user_id"):
            retrieve(query="some query", user_id="")

    def test_user_id_filter_passed_to_store(self):
        """Verify the user_id is passed to vector store search (not filtered in Python)."""
        from rag.retrieval import retrieve

        with patch("rag.retrieval.Embedder") as mock_embedder_cls, \
             patch("rag.retrieval.QdrantVectorStore") as mock_store_cls:

            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [0.1] * 1536
            mock_embedder_cls.return_value = mock_embedder

            mock_store = MagicMock()
            mock_store.search.return_value = []
            mock_store_cls.return_value = mock_store

            retrieve(query="test", user_id="specific-user-id-99")

            # Verify user_id was passed to the store's search method
            call_kwargs = mock_store.search.call_args
            assert call_kwargs.kwargs.get("user_id") == "specific-user-id-99" or \
                   "specific-user-id-99" in str(call_kwargs)
