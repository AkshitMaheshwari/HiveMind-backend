"""
RAG document search tool — wraps the retrieval pipeline so agents can call it
like any other registered tool.

This module is responsible only for the tool wrapper layer. The actual
retrieval logic lives in ``rag/retrieval.py``.
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_NO_DOCS_MESSAGE = (
    "No relevant documents found for this query. "
    "The user has not uploaded any documents that match this topic. "
    "Please answer from your general knowledge or ask the user to upload relevant files."
)


def rag_document_search(query: str, user_id: str, top_k: int = 5) -> Tuple[str, float]:
    """
    Search the user's uploaded documents using vector similarity retrieval.

    This tool retrieves semantically relevant chunks from documents the
    authenticated user has previously uploaded and ingested. It enforces
    strict user-level data isolation — results are scoped exclusively to
    the requesting user's documents.

    Parameters:
        query: The natural-language question or search query.
        user_id: The authenticated user's ID. Used to filter results at the
                 database level — never bypassed.
        top_k: Maximum number of document chunks to retrieve (default 5).

    Returns:
        A formatted string of the most relevant document excerpts with source
        attribution, or a clear message indicating no documents were found.
        Never raises — all errors are logged and a safe fallback is returned.
        Returns a tuple: (formatted_string, highest_confidence_score).
    """
    if not query or not query.strip():
        logger.warning("rag_document_search called with empty query: user_id=%s", user_id)
        return "RAG search error: query must not be empty.", 0.0

    if not user_id or not user_id.strip():
        logger.error("rag_document_search called without user_id")
        return "RAG search error: user_id is required for document retrieval.", 0.0

    try:
        from rag.retrieval import retrieve  # lazy import to avoid circular deps at startup

        chunks = retrieve(query=query, user_id=user_id, top_k=top_k)

        if not chunks:
            logger.info(
                "rag_document_search: no results found: query=%r user_id=%s", query, user_id
            )
            return _NO_DOCS_MESSAGE, 0.0

        max_score = 0.0
        lines = ["**Relevant document excerpts:**\n"]
        for i, chunk in enumerate(chunks, 1):
            score = chunk.score if chunk.score is not None else 0.0
            if score > max_score:
                max_score = score
            score_pct = int(score * 100)
            lines.append(
                f"[{i}] Source: {chunk.source} (relevance: {score_pct}%)\n"
                f"{chunk.text.strip()}\n"
            )

        logger.info(
            "rag_document_search: returned %d chunks: query=%r user_id=%s",
            len(chunks),
            query,
            user_id,
        )
        return "\n".join(lines), max_score

    except ImportError as exc:
        logger.error(
            "rag_document_search: RAG module not available (missing dependencies?): %s", exc
        )
        return (
            "Document search is currently unavailable. "
            "Please ensure the RAG pipeline is configured correctly."
        ), 0.0
    except Exception as exc:
        logger.error(
            "rag_document_search failed: query=%r user_id=%s error=%s",
            query,
            user_id,
            exc,
            exc_info=True,
        )
        return (
            "An error occurred while searching your documents. "
            "Please try again or contact support if the issue persists."
        ), 0.0
