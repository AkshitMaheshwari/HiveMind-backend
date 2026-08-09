"""
Documents API routes — list and delete user's indexed documents.

GET  /api/documents            → list all documents the user has indexed in Qdrant
DELETE /api/documents/{name}  → remove all vectors for a document from Qdrant
"""
import logging
from typing import Any, Dict, List
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import require_authenticated_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents")
async def list_user_documents(
    user: Dict[str, Any] = Depends(require_authenticated_user),
) -> List[Dict[str, Any]]:
    """
    Returns all unique documents the authenticated user has indexed in Qdrant.
    Each item has: source_identifier, chunk_count, source_type, metadata.
    """
    user_id: str = user.get("id", "")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    from rag.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    try:
        if QDRANT_URL:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        else:
            # In-memory Qdrant — no data persists
            return []

        # Scroll through all points for this user and aggregate by source_identifier
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )

        docs: Dict[str, Dict] = {}
        offset = None

        while True:
            results, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=user_filter,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                meta = point.payload.get("metadata", {})
                src = meta.get("source_identifier", "unknown")
                if src not in docs:
                    docs[src] = {
                        "source_identifier": src,
                        "source_type": meta.get("source_type", "unknown"),
                        "chunk_count": 0,
                        "created_at": meta.get("created_at", ""),
                    }
                docs[src]["chunk_count"] += 1

            if offset is None:
                break

        # Sort by source name
        return sorted(docs.values(), key=lambda x: x["source_identifier"].lower())

    except Exception as exc:
        logger.error("list_user_documents: failed for user_id=%s error=%s", user_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch document list.",
        ) from exc


@router.delete("/documents/{source_identifier:path}")
async def delete_user_document_endpoint(
    source_identifier: str,
    user: Dict[str, Any] = Depends(require_authenticated_user),
) -> Dict[str, Any]:
    """
    Removes all Qdrant vectors for the specified document belonging to the
    authenticated user.  URL-decoded automatically by FastAPI.
    """
    user_id: str = user.get("id", "")
    source_identifier = unquote(source_identifier)

    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    try:
        from rag.vector_store import delete_user_document
        delete_user_document(user_id=user_id, source_identifier=source_identifier)
        logger.info("delete_user_document_endpoint: deleted user_id=%s source=%r", user_id, source_identifier)
        return {"success": True, "deleted": source_identifier}
    except Exception as exc:
        logger.error(
            "delete_user_document_endpoint: failed user_id=%s source=%r error=%s",
            user_id, source_identifier, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {exc}",
        ) from exc
