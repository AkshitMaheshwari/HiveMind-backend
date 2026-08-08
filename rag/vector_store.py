"""
RAG vector store — Qdrant client wrapper.

Handles:
- Collection creation (idempotent — safe to call on every startup)
- Bulk chunk upsert with user_id in the payload
- Deletion of all chunks for a given (user_id, source_identifier) pair
  (used for the overwrite-on-re-upload deduplication policy)
- Filtered search that enforces user_id at the Qdrant query level

The user_id filter is applied using Qdrant's server-side ``Filter`` API,
NOT by fetching all results and filtering in Python afterward. This is a
hard data isolation requirement.

Configuration (from rag.config):
    QDRANT_URL:          Server URL. Empty → in-memory client.
    QDRANT_API_KEY:      Cloud API key.
    QDRANT_COLLECTION:   Collection name.
    EMBEDDING_DIMENSIONS: Vector size.
"""
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkPayload:
    """
    A single text chunk ready for insertion into the vector store.

    Attributes:
        chunk_id: Unique identifier for this specific chunk.
        document_id: ID of the parent Document.
        text: The chunk text (already sliced by the chunker).
        vector: The embedding vector for ``text``.
        user_id: The owning user — stored in Qdrant payload for filtering.
        source_identifier: Original filename/URL of the parent document.
        source_type: Data source type (pdf, excel, csv, etc.).
        chunk_index: Position of this chunk within the document (0-indexed).
        metadata: Additional key-value pairs from the parent Document.
    """

    chunk_id: str
    document_id: str
    text: str
    vector: List[float]
    user_id: str
    source_identifier: str
    source_type: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """
    A single result from a vector similarity search.

    Attributes:
        chunk_id: Qdrant point ID for this chunk.
        text: The retrieved chunk text.
        source_identifier: Original source of the document.
        source_type: Data source type.
        score: Cosine similarity score (0.0–1.0, higher is more relevant).
        metadata: Additional metadata stored with this chunk.
    """

    chunk_id: str
    text: str
    source_identifier: str
    source_type: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QdrantVectorStore:
    """
    Qdrant-backed vector store for document chunks.

    Initialises a Qdrant client automatically:
    - If ``QDRANT_URL`` is set → connects to remote Qdrant Cloud / self-hosted.
    - Otherwise → uses in-memory Qdrant client (for local development).

    Parameters:
        collection_name: Override the collection name from config.
    """

    def __init__(self, collection_name: str = "") -> None:
        from rag.config import (
            QDRANT_URL,
            QDRANT_API_KEY,
            QDRANT_COLLECTION,
            EMBEDDING_DIMENSIONS,
        )

        self._collection = collection_name or QDRANT_COLLECTION
        self._dimensions = EMBEDDING_DIMENSIONS

        try:
            from qdrant_client import QdrantClient  # type: ignore[import]

            if QDRANT_URL:
                self._client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY or None,
                )
                logger.info("QdrantVectorStore: connected to %s", QDRANT_URL)
            else:
                self._client = QdrantClient(":memory:")
                logger.warning(
                    "QdrantVectorStore: QDRANT_URL not set — using in-memory Qdrant. "
                    "Data will not persist across restarts."
                )
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is not installed. Run: pip install qdrant-client"
            ) from exc

    def ensure_collection(self) -> None:
        """
        Create the Qdrant collection if it doesn't already exist.

        This method is idempotent — it is safe to call on every startup.
        If the collection already exists with correct configuration, it is
        left unchanged. If it exists with wrong vector size, a warning is logged.

        Raises:
            RuntimeError: If the Qdrant client is unavailable.
        """
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

        try:
            existing = [c.name for c in self._client.get_collections().collections]
            if self._collection not in existing:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "QdrantVectorStore.ensure_collection: created collection '%s' "
                    "(dimensions=%d, distance=COSINE).",
                    self._collection,
                    self._dimensions,
                )
            else:
                logger.info(
                    "QdrantVectorStore.ensure_collection: collection '%s' already exists.",
                    self._collection,
                )

            # Ensure the user_id payload index exists for filtering (idempotent)
            from qdrant_client.models import PayloadSchemaType # type: ignore[import]
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("QdrantVectorStore.ensure_collection: verified user_id index.")
        except Exception as exc:
            logger.error(
                "QdrantVectorStore.ensure_collection failed: collection=%s error=%s",
                self._collection,
                exc,
                exc_info=True,
            )
            raise

    def upsert_chunks(self, chunks: List[ChunkPayload]) -> None:
        """
        Bulk-upsert document chunks into the vector store.

        Parameters:
            chunks: A list of :class:`ChunkPayload` objects to store.

        Raises:
            ValueError: If ``chunks`` is empty.
            Exception: If the Qdrant upsert fails (propagated to caller).
        """
        if not chunks:
            raise ValueError("upsert_chunks: chunks list must not be empty.")

        from qdrant_client.models import PointStruct  # type: ignore[import]

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=chunk.vector,
                payload={
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "user_id": chunk.user_id,
                    "source_identifier": chunk.source_identifier,
                    "source_type": chunk.source_type,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
        ]

        try:
            self._client.upsert(
                collection_name=self._collection,
                points=points,
                wait=True,
            )
            logger.info(
                "QdrantVectorStore.upsert_chunks: upserted %d chunks "
                "(source=%r user_id=%s)",
                len(chunks),
                chunks[0].source_identifier,
                chunks[0].user_id,
            )
        except Exception as exc:
            logger.error(
                "QdrantVectorStore.upsert_chunks failed: count=%d "
                "source=%r user_id=%s error=%s",
                len(chunks),
                chunks[0].source_identifier if chunks else "N/A",
                chunks[0].user_id if chunks else "N/A",
                exc,
                exc_info=True,
            )
            raise

    def delete_by_source(self, user_id: str, source_identifier: str) -> int:
        """
        Delete all chunks belonging to a specific (user_id, source_identifier) pair.

        Used to implement the overwrite-on-re-upload deduplication policy.
        If the source has never been ingested, this is a no-op (returns 0).

        Parameters:
            user_id: The owning user.
            source_identifier: The original filename or URL.

        Returns:
            The number of points deleted (0 if none existed).

        Raises:
            Exception: If the Qdrant delete operation fails.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore[import]

        delete_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(
                    key="source_identifier",
                    match=MatchValue(value=source_identifier),
                ),
            ]
        )

        try:
            result = self._client.delete(
                collection_name=self._collection,
                points_selector=delete_filter,
                wait=True,
            )
            logger.info(
                "QdrantVectorStore.delete_by_source: deleted chunks for "
                "user_id=%s source=%r",
                user_id,
                source_identifier,
            )
            return getattr(result, "deleted_count", 0)
        except Exception as exc:
            logger.error(
                "QdrantVectorStore.delete_by_source failed: "
                "user_id=%s source=%r error=%s",
                user_id,
                source_identifier,
                exc,
                exc_info=True,
            )
            raise

    def search(
        self,
        query_vector: List[float],
        user_id: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Search for the most similar chunks, filtered to ``user_id``.

        The ``user_id`` filter is applied by Qdrant at the server side using
        a ``Filter`` condition. This is NOT application-level post-filtering.
        A user can only ever retrieve their own documents.

        Parameters:
            query_vector: The embedding vector of the search query.
            user_id: The user whose documents to search.
            top_k: Maximum number of results to return (default 5).

        Returns:
            A list of :class:`SearchResult` objects, ordered by descending
            similarity score. Returns an empty list if no results match.

        Raises:
            ValueError: If ``user_id`` is empty or ``query_vector`` is empty.
            Exception: If the Qdrant search fails.
        """
        if not user_id or not user_id.strip():
            raise ValueError("search: user_id must not be empty.")
        if not query_vector:
            raise ValueError("search: query_vector must not be empty.")

        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore[import]

        user_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]
        )

        try:
            raw_results = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                query_filter=user_filter,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.error(
                "QdrantVectorStore.search failed: user_id=%s top_k=%d error=%s",
                user_id,
                top_k,
                exc,
                exc_info=True,
            )
            raise

        results = []
        for hit in raw_results:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(hit.id),
                    text=payload.get("text", ""),
                    source_identifier=payload.get("source_identifier", ""),
                    source_type=payload.get("source_type", ""),
                    score=float(hit.score),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ("text", "user_id", "source_identifier", "source_type")
                    },
                )
            )

        logger.debug(
            "QdrantVectorStore.search: user_id=%s top_k=%d results=%d",
            user_id,
            top_k,
            len(results),
        )
        return results
