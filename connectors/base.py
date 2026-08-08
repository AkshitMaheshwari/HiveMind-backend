"""
BaseConnector — abstract interface that every data source connector must implement.

Adding a new data source (e.g. Google Drive, GitHub, a web scraper) means:
1. Subclass ``BaseConnector``.
2. Implement ``ingest(**kwargs) -> List[Document]``.
3. Wire the new connector into the relevant API route.

The rest of the system (RAG pipeline, agents) does not change.
"""
from abc import ABC, abstractmethod
from typing import Any, List

from connectors.document import Document


class BaseConnector(ABC):
    """
    Abstract base class for all data source connectors.

    A connector is responsible for:
    - Accepting source-specific inputs (file bytes, a URL, OAuth credentials, etc.)
    - Parsing / fetching the raw content
    - Normalising the extracted text into one or more :class:`~connectors.document.Document` objects
    - Raising :class:`~connectors.exceptions.ConnectorError` for any expected
      failure (bad file, unsupported type, empty content, size limit exceeded)
    - NOT performing any chunking, embedding, or storage — those belong in the RAG pipeline

    Implementations must guarantee that every returned ``Document`` has a
    non-empty ``text`` field and a correctly set ``user_id``. The pipeline
    trusts this invariant.
    """

    @abstractmethod
    def ingest(self, **kwargs: Any) -> List[Document]:
        """
        Ingest data from a source and return normalised ``Document`` objects.

        Parameters:
            **kwargs: Source-specific keyword arguments defined by the
                      concrete subclass (e.g. ``file``, ``url``, ``user_id``).

        Returns:
            A non-empty list of :class:`~connectors.document.Document` objects.

        Raises:
            ConnectorError: For any expected, user-facing failure such as
                a corrupted file, unsupported format, or empty content.
            ValueError: For programming errors such as missing required kwargs.
        """
        ...

    def validate_user_id(self, user_id: str) -> None:
        """
        Validate that ``user_id`` is present and non-empty.

        Parameters:
            user_id: The user identifier to validate.

        Raises:
            ValueError: If ``user_id`` is missing or blank.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required and must not be empty.")
