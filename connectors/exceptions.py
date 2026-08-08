"""
ConnectorError — the single exception type raised by all connector implementations.

Downstream code (API routes, tests) catches ``ConnectorError`` to distinguish
expected, user-facing problems (bad file, unsupported type, empty content)
from unexpected system errors.
"""


class ConnectorError(Exception):
    """
    Raised by any :class:`~connectors.base.BaseConnector` implementation when
    a source-specific ingestion problem occurs.

    Examples of situations that raise this:
    - The file is corrupted or cannot be parsed.
    - The file has no extractable text content.
    - The file exceeds the configured size limit.
    - The file type is not supported.

    Attributes:
        message: A human-readable description suitable for surfacing to the user.
        source: The original source identifier (filename, URL, etc.) if available.
    """

    def __init__(self, message: str, source: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.source = source

    def __str__(self) -> str:
        if self.source:
            return f"[{self.source}] {self.message}"
        return self.message
