"""
Document — the normalised data structure that every connector produces.

All data sources (PDF, Excel, CSV, future: web pages, GitHub repos, etc.)
must normalise their raw content into this structure before handing it
to the RAG ingestion pipeline. The pipeline knows nothing about file formats
— it only ever sees ``Document`` objects.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class Document:
    """
    A normalised document produced by any data source connector.

    Every field is mandatory except ``metadata`` which defaults to an empty
    dict. Connectors are responsible for populating all required fields.

    Attributes:
        id: Globally unique document identifier (UUID4 string). Generated
            automatically if not provided.
        text: The full extracted text content. Must be non-empty — connectors
              must validate this before constructing a Document.
        source_type: Short identifier for the data source type, e.g.
                     ``"pdf"``, ``"excel"``, ``"csv"``, ``"web"``.
        source_identifier: The original filename, URL, or path that
                           uniquely identifies the source within its type.
                           Used as the key for deduplication in the vector store.
        user_id: The authenticated user who owns this document. Used to
                 enforce data isolation at query time.
        metadata: Arbitrary key-value pairs provided by the connector.
                  Examples: ``{"page_count": 12, "sheet_name": "Q1 Data"}``.
        created_at: UTC timestamp of when this Document object was created.
    """

    text: str
    source_type: str
    source_identifier: str
    user_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate required fields after construction."""
        errors = []
        if not self.text or not self.text.strip():
            errors.append("'text' must be non-empty.")
        if not self.source_type or not self.source_type.strip():
            errors.append("'source_type' must be non-empty.")
        if not self.source_identifier or not self.source_identifier.strip():
            errors.append("'source_identifier' must be non-empty.")
        if not self.user_id or not self.user_id.strip():
            errors.append("'user_id' must be non-empty.")
        if errors:
            raise ValueError("Invalid Document: " + "; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the Document to a plain dictionary.

        Returns:
            A dict representation with all fields, using ISO-format for the
            ``created_at`` timestamp.
        """
        return {
            "id": self.id,
            "text": self.text,
            "source_type": self.source_type,
            "source_identifier": self.source_identifier,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
