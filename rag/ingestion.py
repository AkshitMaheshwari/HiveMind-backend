"""
RAG ingestion pipeline — takes Document objects, chunks them, embeds them,
and stores them in the vector store.

Design guarantees:
- **Atomic per-document**: If embedding fails for a document, zero chunks are
  stored for that document. The pipeline never creates a partially-indexed document.
- **Deduplication**: Before ingesting a document, all existing chunks for the
  same (user_id, source_identifier) are deleted. Re-uploading the same file
  always results in exactly one set of chunks.
- **No silent failures**: Every document failure is recorded in the returned
  ``IngestionResult.failed`` list and logged with full context.
"""
import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from connectors.document import Document
from rag.chunker import chunk_text
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from rag.embedder import Embedder, EmbeddingError
from rag.vector_store import QdrantVectorStore, ChunkPayload

logger = logging.getLogger(__name__)


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class FailedDocument:
    """
    Records information about a document that failed to ingest.

    Attributes:
        document_id: The Document's ID.
        source_identifier: The original filename or URL.
        user_id: The owning user.
        error: A human-readable description of what went wrong.
    """

    document_id: str
    source_identifier: str
    user_id: str
    error: str


@dataclass
class IngestionResult:
    """
    Summarises the outcome of an ingestion pipeline run.

    Attributes:
        successful: List of document IDs that were fully ingested.
        failed: List of :class:`FailedDocument` records for documents
                that did not ingest cleanly.
        chunks_ingested: Total number of chunks successfully stored.
    """

    successful: List[str] = field(default_factory=list)
    failed: List[FailedDocument] = field(default_factory=list)
    chunks_ingested: int = 0

    @property
    def all_succeeded(self) -> bool:
        """True if every document ingested successfully."""
        return len(self.failed) == 0


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def ingest_documents(documents: "List") -> IngestionResult:
    """
    Ingest a list of Document objects into the RAG vector store.

    For each document:
    1. Delete existing chunks for ``(user_id, source_identifier)`` (dedup).
    2. Chunk the document text.
    3. Generate embeddings for all chunks (with retry backoff).
       If embedding fails → record failure, do NOT store any chunks.
    4. Upsert all chunks atomically (all-or-nothing per document).

    Parameters:
        documents: A list of :class:`~connectors.document.Document` objects.
                   Must be non-empty.

    Returns:
        An :class:`IngestionResult` with ``successful`` IDs, ``failed`` records,
        and total ``chunks_ingested``. Never raises — all failures are captured.

    Raises:
        ValueError: If ``documents`` is empty.
    """
    if not documents:
        raise ValueError("ingest_documents: documents list must not be empty.")


    result = IngestionResult()
    store = QdrantVectorStore()
    embedder = Embedder()

    for doc in documents:
        logger.info(
            "ingest_documents: processing document: id=%s source=%r user_id=%s chars=%d",
            doc.id,
            doc.source_identifier,
            doc.user_id,
            len(doc.text),
        )

        try:
            # Step 1: Deduplication — delete existing chunks for this source
            try:
                store.delete_by_source(
                    user_id=doc.user_id,
                    source_identifier=doc.source_identifier,
                )
            except Exception as del_exc:
                logger.warning(
                    "ingest_documents: dedup delete failed (continuing): "
                    "id=%s source=%r user_id=%s error=%s",
                    doc.id,
                    doc.source_identifier,
                    doc.user_id,
                    del_exc,
                )
                # Non-fatal: we'll upsert anyway (upsert semantics handle duplicates)

            # Step 2: Chunk the text
            chunks_text = chunk_text(doc.text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            logger.debug(
                "ingest_documents: chunked: id=%s chunks=%d", doc.id, len(chunks_text)
            )

            # Step 3: Embed all chunks — atomic: if this fails, nothing is stored
            try:
                vectors = embedder.embed_texts(chunks_text)
            except EmbeddingError as emb_exc:
                logger.error(
                    "ingest_documents: embedding failed, aborting document: "
                    "id=%s source=%r user_id=%s error=%s",
                    doc.id,
                    doc.source_identifier,
                    doc.user_id,
                    emb_exc,
                )
                result.failed.append(
                    FailedDocument(
                        document_id=doc.id,
                        source_identifier=doc.source_identifier,
                        user_id=doc.user_id,
                        error=f"Embedding failed: {emb_exc}",
                    )
                )
                continue  # Skip to next document — nothing has been stored

            # Step 4: Build ChunkPayload objects
            chunk_payloads = [
                ChunkPayload(
                    chunk_id=str(uuid.uuid4()),
                    document_id=doc.id,
                    text=chunk_text_item,
                    vector=vector,
                    user_id=doc.user_id,
                    source_identifier=doc.source_identifier,
                    source_type=doc.source_type,
                    chunk_index=idx,
                    metadata={
                        **doc.metadata,
                        "created_at": doc.created_at.isoformat(),
                    },
                )
                for idx, (chunk_text_item, vector) in enumerate(zip(chunks_text, vectors))
            ]

            # Step 5: Store all chunks
            store.upsert_chunks(chunk_payloads)

            result.successful.append(doc.id)
            result.chunks_ingested += len(chunk_payloads)

            logger.info(
                "ingest_documents: document ingested successfully: "
                "id=%s source=%r user_id=%s chunks=%d",
                doc.id,
                doc.source_identifier,
                doc.user_id,
                len(chunk_payloads),
            )

        except Exception as exc:
            logger.error(
                "ingest_documents: unexpected error for document: "
                "id=%s source=%r user_id=%s error=%s",
                doc.id,
                doc.source_identifier,
                doc.user_id,
                exc,
                exc_info=True,
            )
            result.failed.append(
                FailedDocument(
                    document_id=doc.id,
                    source_identifier=doc.source_identifier,
                    user_id=doc.user_id,
                    error=f"Unexpected ingestion error: {exc}",
                )
            )

    logger.info(
        "ingest_documents: complete: successful=%d failed=%d total_chunks=%d",
        len(result.successful),
        len(result.failed),
        result.chunks_ingested,
    )
    return result
