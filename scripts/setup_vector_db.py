"""
Vector DB setup script — creates the Qdrant collection for document storage.

Run this once before starting the server for the first time, or any time
you need to re-create the collection (e.g. after changing EMBEDDING_DIMENSIONS).

This script is idempotent: if the collection already exists, it will not
be modified or deleted.

Usage:
    python scripts/setup_vector_db.py

Environment variables (set in .env or shell):
    QDRANT_URL           — Qdrant server URL (leave empty for in-memory/local)
    QDRANT_API_KEY       — API key for Qdrant Cloud (optional for self-hosted)
    QDRANT_COLLECTION    — Collection name (default: "documents")
    EMBEDDING_DIMENSIONS — Vector size (default: 1536)
"""
import sys
from pathlib import Path

# Ensure backend root is on sys.path when running this script directly
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import logging
import os

from dotenv import load_dotenv

load_dotenv(_BACKEND_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("setup_vector_db")


def main() -> int:
    """
    Create the Qdrant vector collection.

    Returns:
        0 on success, 1 on failure.
    """
    from rag.config import (
        QDRANT_URL,
        QDRANT_API_KEY,
        QDRANT_COLLECTION,
        EMBEDDING_DIMENSIONS,
    )

    logger.info("=" * 60)
    logger.info("Vector DB Setup")
    logger.info("=" * 60)
    logger.info("QDRANT_URL:          %s", QDRANT_URL or "(in-memory — no URL set)")
    logger.info("QDRANT_COLLECTION:   %s", QDRANT_COLLECTION)
    logger.info("EMBEDDING_DIMENSIONS: %d", EMBEDDING_DIMENSIONS)
    logger.info("=" * 60)

    try:
        from rag.vector_store import ensure_collection
        ensure_collection()
        logger.info("✅  Collection '%s' is ready.", QDRANT_COLLECTION)
        return 0
    except ImportError as exc:
        logger.error("❌  Missing dependency: %s", exc)
        logger.error("    Run: pip install qdrant-client")
        return 1
    except Exception as exc:
        logger.error("❌  Failed to create collection: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
