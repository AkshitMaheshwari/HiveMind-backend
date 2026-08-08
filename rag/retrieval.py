"""
RAG retrieval — embed a query and fetch the most relevant user-scoped chunks.

The ``user_id`` filter is enforced at the Qdrant server side. This function
never fetches documents from other users and then filters them in Python.

Returns an empty list if no matching documents exist — never raises on
an empty result set. The caller (agent tool) is responsible for producing
a clear "no documents found" message to the user.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

from rag.config import TOP_K_RESULTS
from rag.embedder import Embedder, EmbeddingError
from rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """
    A single retrieved document chunk.

    Attributes:
        text: The chunk text content.
        source: The original source identifier (filename or URL).
        source_type: Data source type (pdf, excel, csv, etc.).
        score: Cosine similarity score (0.0–1.0).
        metadata: Additional metadata stored with the chunk.
    """

    text: str
    source: str
    source_type: str
    score: float
    metadata: dict


def retrieve(
    query: str,
    user_id: str,
    top_k: int = 0,
) -> List[RetrievedChunk]:
    """
    Retrieve the most semantically relevant document chunks for a query.

    Data isolation is enforced at the Qdrant query level: the ``user_id``
    is passed as a server-side filter, not applied in Python after fetching.

    If the user has no uploaded documents, or no documents match the query,
    an empty list is returned — not an exception.

    Parameters:
        query: The natural-language search query.
        user_id: The authenticated user. Only this user's chunks are returned.
        top_k: Maximum chunks to return. Defaults to ``TOP_K_RESULTS`` from config.

    Returns:
        A list of :class:`RetrievedChunk` objects ordered by descending
        similarity score. Empty list if no results.

    Raises:
        ValueError: If ``query`` or ``user_id`` is empty.
        Exception: Re-raises unexpected errors from embedding or search.
    """
    if not query or not query.strip():
        raise ValueError("retrieve: query must not be empty.")
    if not user_id or not user_id.strip():
        raise ValueError("retrieve: user_id must not be empty.")


    k = top_k if top_k > 0 else TOP_K_RESULTS

    # Embed the query
    embedder = Embedder()
    try:
        query_vector = embedder.embed_query(query)
    except EmbeddingError as exc:
        logger.error(
            "retrieve: failed to embed query: query=%r user_id=%s error=%s",
            query,
            user_id,
            exc,
        )
        raise  # Propagate — caller should know embedding is unavailable

    # Search with server-side user_id filter
    store = QdrantVectorStore()
    try:
        raw_results = store.search(
            query_vector=query_vector,
            user_id=user_id,
            top_k=k,
        )
    except Exception as exc:
        logger.error(
            "retrieve: vector search failed: query=%r user_id=%s error=%s",
            query,
            user_id,
            exc,
            exc_info=True,
        )
        raise

    chunks = [
        RetrievedChunk(
            text=r.text,
            source=r.source_identifier,
            source_type=r.source_type,
            score=r.score,
            metadata=r.metadata,
        )
        for r in raw_results
    ]

    logger.info(
        "retrieve: found %d chunks: query=%r user_id=%s top_k=%d",
        len(chunks),
        query,
        user_id,
        k,
    )
    return chunks
