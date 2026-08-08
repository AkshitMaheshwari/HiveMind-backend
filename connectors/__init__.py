"""
connectors — data source abstraction layer.

All data source implementations live here. The rest of the system (RAG
pipeline, API routes) only ever sees ``Document`` objects — never raw files
or source-specific data structures.
"""
from connectors.document import Document
from connectors.base import BaseConnector
from connectors.exceptions import ConnectorError

__all__ = ["Document", "BaseConnector", "ConnectorError"]
