"""
Upload API route — POST /api/upload

Accepts multipart file uploads from authenticated users, runs them through
the FileConnector to produce Document objects, then ingests them into the
RAG vector store.

The route never touches file bytes directly beyond reading them. All
file-format logic lives in the connector layer; all vector storage logic
lives in the RAG ingestion layer.
"""
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from api.auth import require_authenticated_user
from connectors.exceptions import ConnectorError
from connectors.file_connector import FileConnector

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Response schema ──────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response returned after a successful file upload and ingestion."""
    document_ids: List[str]
    chunks_ingested: int
    source: str
    message: str


# ─── Upload endpoint ──────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_file(
    file: UploadFile = File(..., description="File to upload. Supported: PDF, XLSX, XLS, CSV"),
    user: Dict[str, Any] = Depends(require_authenticated_user),
) -> UploadResponse:
    """
    Upload a file, extract its text content, and ingest it into the RAG vector store.

    The file goes through:
    1. ``FileConnector`` → validates, parses, and normalises into ``Document`` objects
    2. ``ingest_documents`` → chunks, embeds, and stores in Qdrant (scoped to user)

    Parameters:
        file: The uploaded file (multipart/form-data).
        user: The authenticated user (injected by dependency).

    Returns:
        An :class:`UploadResponse` with document IDs, chunk count, and the source filename.

    Raises:
        HTTPException 401: If the user is not authenticated.
        HTTPException 422: If the file is corrupted, empty, unsupported, or too large.
        HTTPException 500: For unexpected server errors.
    """
    user_id: str = user.get("id", "")
    filename: str = file.filename or "unknown_file"

    logger.info(
        "upload_file: user_id=%s filename=%r content_type=%s",
        user_id,
        filename,
        file.content_type,
    )

    # ── Read file bytes ───────────────────────────────────────────────────────
    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error(
            "upload_file: failed to read upload stream: user_id=%s filename=%r error=%s",
            user_id,
            filename,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read the uploaded file. Please try again.",
        ) from exc

    # ── Connector: validate + parse ───────────────────────────────────────────
    connector = FileConnector()
    try:
        documents = connector.ingest(
            file_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
        )
    except ValueError as exc:
        # Programming error — bad call site (e.g. missing user_id)
        logger.error(
            "upload_file: connector validation error: user_id=%s filename=%r error=%s",
            user_id,
            filename,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ConnectorError as exc:
        # Expected user-facing error: corrupted file, unsupported type, empty, too large
        logger.warning(
            "upload_file: connector rejected file: user_id=%s filename=%r reason=%s",
            user_id,
            filename,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(
            "upload_file: unexpected connector error: user_id=%s filename=%r error=%s",
            user_id,
            filename,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the file.",
        ) from exc

    # ── RAG ingestion ─────────────────────────────────────────────────────────
    try:
        from rag.ingestion import ingest_documents  # lazy import — RAG is optional
        result = ingest_documents(documents)
    except ImportError as exc:
        logger.warning(
            "upload_file: RAG ingestion module not available (documents parsed but not indexed): %s",
            exc,
        )
        # Still return success for the parsing step — inform user about indexing
        doc_ids = [doc.id for doc in documents]
        return UploadResponse(
            document_ids=doc_ids,
            chunks_ingested=0,
            source=filename,
            message=(
                "File parsed successfully, but RAG indexing is currently unavailable. "
                "Configure QDRANT_URL and EMBEDDING_API_KEY to enable document search."
            ),
        )
    except Exception as exc:
        logger.error(
            "upload_file: RAG ingestion failed: user_id=%s filename=%r error=%s",
            user_id,
            filename,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The file was parsed successfully but could not be indexed for search. "
                "Please try again or contact support."
            ),
        ) from exc

    # ── Report failures ───────────────────────────────────────────────────────
    if result.failed:
        failed_sources = [f.source_identifier for f in result.failed]
        logger.error(
            "upload_file: ingestion partially failed: user_id=%s filename=%r failed=%s",
            user_id,
            filename,
            failed_sources,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Ingestion failed for: {', '.join(failed_sources)}. "
                "Please try uploading again."
            ),
        )

    doc_ids = [doc.id for doc in documents]
    logger.info(
        "upload_file: success: user_id=%s filename=%r docs=%d chunks=%d",
        user_id,
        filename,
        len(doc_ids),
        result.chunks_ingested,
    )

    return UploadResponse(
        document_ids=doc_ids,
        chunks_ingested=result.chunks_ingested,
        source=filename,
        message=f"Successfully ingested '{filename}' — {result.chunks_ingested} chunks indexed.",
    )
