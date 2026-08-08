"""
RAG ingestion pipeline using LangChain.
"""
import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document as LangchainDocument
from connectors.document import Document as AppDocument
from rag.chunker import chunk_text
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from rag.vector_store import get_vector_store, delete_user_document

logger = logging.getLogger(__name__)

@dataclass
class FailedDocument:
    document_id: str
    source_identifier: str
    user_id: str
    error: str

@dataclass
class IngestionResult:
    successful: List[str] = field(default_factory=list)
    failed: List[FailedDocument] = field(default_factory=list)
    chunks_ingested: int = 0

    @property
    def all_succeeded(self) -> bool:
        return len(self.failed) == 0


def ingest_documents(documents: List[AppDocument]) -> IngestionResult:
    if not documents:
        raise ValueError("ingest_documents: documents list must not be empty.")

    result = IngestionResult()
    store = get_vector_store()

    for doc in documents:
        logger.info("ingest_documents: processing %r", doc.source_identifier)

        try:
            # 1. Deduplication
            try:
                delete_user_document(user_id=doc.user_id, source_identifier=doc.source_identifier)
            except Exception as del_exc:
                logger.warning("dedup delete failed: %s", del_exc)

            # 2. Chunking
            chunks_text = chunk_text(doc.text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            
            # 3. Create LangChain Documents
            lc_docs = []
            for idx, text in enumerate(chunks_text):
                lc_docs.append(
                    LangchainDocument(
                        page_content=text,
                        metadata={
                            **doc.metadata,
                            "user_id": doc.user_id,
                            "source_identifier": doc.source_identifier,
                            "source_type": doc.source_type,
                            "chunk_index": idx,
                            "document_id": doc.id,
                        }
                    )
                )

            # 4. Add to Vector Store (handles embedding natively)
            store.add_documents(lc_docs)

            result.successful.append(doc.id)
            result.chunks_ingested += len(lc_docs)
            logger.info("ingested %d chunks for %r", len(lc_docs), doc.source_identifier)

        except Exception as exc:
            logger.error("ingestion failed for %r: %s", doc.source_identifier, exc, exc_info=True)
            result.failed.append(
                FailedDocument(
                    document_id=doc.id,
                    source_identifier=doc.source_identifier,
                    user_id=doc.user_id,
                    error=str(exc)
                )
            )

    return result
