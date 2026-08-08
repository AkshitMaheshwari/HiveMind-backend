"""
RAG vector store — provides LangChain QdrantVectorStore.
"""
import logging
from langchain_qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


def get_vector_store() -> QdrantVectorStore:
    """
    Returns a configured LangChain QdrantVectorStore.
    """
    from rag.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
    from rag.embedder import get_embedder

    embedder = get_embedder()

    if QDRANT_URL:
        return QdrantVectorStore.from_existing_collection(
            embedding=embedder,
            collection_name=QDRANT_COLLECTION,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
        )
    else:
        logger.warning(
            "LangChain Qdrant: QDRANT_URL not set — using in-memory mode. "
            "Data will not persist across restarts."
        )
        return QdrantVectorStore.from_existing_collection(
            embedding=embedder,
            collection_name=QDRANT_COLLECTION,
            location=":memory:",
        )


def ensure_collection() -> None:
    """
    Ensure the Qdrant collection and indices exist.
    """
    from rag.config import (
        QDRANT_URL,
        QDRANT_API_KEY,
        QDRANT_COLLECTION,
        EMBEDDING_DIMENSIONS,
    )

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

        if QDRANT_URL:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        else:
            client = QdrantClient(":memory:")

        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE
                ),
            )
            logger.info("ensure_collection: created collection %s", QDRANT_COLLECTION)

        # In langchain_qdrant, metadata is usually stored under the 'metadata' payload key.
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="metadata.user_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="metadata.source_identifier",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("ensure_collection: verified indices")
    except ImportError:
        logger.error("qdrant-client not installed")
    except Exception as exc:
        logger.error("ensure_collection failed: %s", exc)
        raise


def delete_user_document(user_id: str, source_identifier: str) -> None:
    """
    Utility to delete a specific document from Qdrant by user_id and source_identifier.
    """
    from rag.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if QDRANT_URL:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        else:
            client = QdrantClient(":memory:")

        f = Filter(
            must=[
                FieldCondition(
                    key="metadata.user_id", match=MatchValue(value=user_id)
                ),
                FieldCondition(
                    key="metadata.source_identifier",
                    match=MatchValue(value=source_identifier),
                ),
            ]
        )

        client.delete(collection_name=QDRANT_COLLECTION, points_selector=f, wait=True)
        logger.info(
            "delete_user_document: deleted chunks for user=%s source=%r",
            user_id,
            source_identifier,
        )
    except Exception as exc:
        logger.error("delete_user_document failed: %s", exc)
        raise
