"""
RAG retrieval — searches the vector store using LangChain.
"""
import logging
from typing import List
from rag.vector_store import get_vector_store
from rag.config import TOP_K_RESULTS
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

def retrieve_context(query: str, user_id: str, top_k: int = 0) -> str:
    """
    Search the vector store for chunks matching the query, restricted to user_id.

    Parameters:
        query: The search query string.
        user_id: The ID of the user requesting the search.
        top_k: Maximum number of chunks to retrieve (defaults to config).

    Returns:
        A single formatted string containing all retrieved chunks, separated
        by newlines. Returns an empty string if no chunks match.
    """
    if not query or not query.strip():
        logger.warning("retrieve_context: empty query provided")
        return ""
    if not user_id or not user_id.strip():
        raise ValueError("retrieve_context: user_id must not be empty.")

    k = top_k or TOP_K_RESULTS
    store = get_vector_store()
    
    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.user_id", match=MatchValue(value=user_id)
            )
        ]
    )

    try:
        docs = store.similarity_search(query, k=k, filter=qdrant_filter)
    except Exception as exc:
        logger.error("retrieve_context: search failed: %s", exc)
        return ""

    if not docs:
        logger.debug("retrieve_context: no matching chunks found for user_id=%s", user_id)
        return ""

    logger.debug("retrieve_context: found %d chunks for user_id=%s", len(docs), user_id)

    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source_identifier", "Unknown")
        formatted_chunks.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n".join(formatted_chunks)
