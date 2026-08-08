"""
Unit tests for the LangChain RAG pipeline components:
- rag/chunker.py
- rag/ingestion.py
- rag/retrieval.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import pytest
from connectors.document import Document
from langchain_core.documents import Document as LangchainDocument


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
        with pytest.raises(ValueError, match="empty"):
            chunk_text("", chunk_size=800, overlap=150)

    def test_invalid_chunk_size_raises(self):
        from rag.chunker import chunk_text
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("Some text.", chunk_size=0, overlap=0)


# ─── Ingestion tests ──────────────────────────────────────────────────────────

class TestIngestion:
    def _make_doc(self, text: str = "Some text content.", uid: str = "user-1", source: str = "test.pdf"):
        return Document(
            text=text,
            source_type="pdf",
            source_identifier=source,
            user_id=uid,
        )

    def test_successful_ingestion_happy_path(self):
        from rag.ingestion import ingest_documents
        doc = self._make_doc()

        with patch("rag.ingestion.get_vector_store") as mock_store_factory, \
             patch("rag.ingestion.delete_user_document") as mock_delete:
             
            mock_store = MagicMock()
            mock_store_factory.return_value = mock_store

            result = ingest_documents([doc])

        assert doc.id in result.successful
        assert result.failed == []
        mock_delete.assert_called_once_with(user_id=doc.user_id, source_identifier=doc.source_identifier)
        mock_store.add_documents.assert_called_once()

    def test_ingestion_handles_dedup_failure_gracefully(self):
        from rag.ingestion import ingest_documents
        doc = self._make_doc()

        with patch("rag.ingestion.get_vector_store") as mock_store_factory, \
             patch("rag.ingestion.delete_user_document") as mock_delete:
             
            mock_store = MagicMock()
            mock_store_factory.return_value = mock_store
            
            mock_delete.side_effect = Exception("DB down")

            result = ingest_documents([doc])

        assert doc.id in result.successful
        mock_store.add_documents.assert_called_once()


    def test_ingestion_embedding_failure_records_failed_doc(self):
        from rag.ingestion import ingest_documents
        doc = self._make_doc()

        with patch("rag.ingestion.get_vector_store") as mock_store_factory, \
             patch("rag.ingestion.delete_user_document"):
             
            mock_store = MagicMock()
            mock_store.add_documents.side_effect = Exception("OpenAI down")
            mock_store_factory.return_value = mock_store

            result = ingest_documents([doc])

        assert len(result.failed) == 1
        assert result.failed[0].document_id == doc.id
        assert "OpenAI down" in result.failed[0].error
        assert result.successful == []


# ─── Retrieval tests ──────────────────────────────────────────────────────────

class TestRetrieval:
    def test_retrieve_context_happy_path(self):
        from rag.retrieval import retrieve_context
        
        with patch("rag.retrieval.get_vector_store") as mock_store_factory:
            mock_store = MagicMock()
            
            mock_store.similarity_search.return_value = [
                LangchainDocument(page_content="Context 1", metadata={"source_identifier": "doc1.pdf"})
            ]
            mock_store_factory.return_value = mock_store
            
            result = retrieve_context("query", "user1")
            
        assert "Context 1" in result
        assert "doc1.pdf" in result
        
        call_kwargs = mock_store.similarity_search.call_args.kwargs
        assert "filter" in call_kwargs

    def test_retrieve_context_empty_query(self):
        from rag.retrieval import retrieve_context
        assert retrieve_context("", "user1") == ""

    def test_retrieve_context_empty_results(self):
        from rag.retrieval import retrieve_context
        with patch("rag.retrieval.get_vector_store") as mock_store_factory:
            mock_store = MagicMock()
            mock_store.similarity_search.return_value = []
            mock_store_factory.return_value = mock_store
            
            assert retrieve_context("query", "user1") == ""
