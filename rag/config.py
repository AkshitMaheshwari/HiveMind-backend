"""
RAG pipeline configuration — all values sourced from environment variables.

No hardcoded values. Every setting has a documented default so the system
works out-of-the-box for local development without any configuration.

Environment variables:
    QDRANT_URL:           Qdrant server URL (e.g. https://xyz.qdrant.io).
                          If unset, an in-memory Qdrant client is used.
    QDRANT_API_KEY:       API key for Qdrant Cloud. Not required for local.
    QDRANT_COLLECTION:    Collection name (default: "documents").
    EMBEDDING_PROVIDER:   "openai" or "groq" (default: "openai").
    EMBEDDING_MODEL:      Model name (default: "text-embedding-3-small").
    EMBEDDING_DIMENSIONS: Vector dimensions matching the model (default: 1536
                          for text-embedding-3-small).
    CHUNK_SIZE:           Target chunk size in characters (default: 800).
    CHUNK_OVERLAP:        Character overlap between adjacent chunks (default: 150).
    TOP_K_RESULTS:        Default number of results for retrieval (default: 5).
    MAX_EMBED_RETRIES:    Max retry attempts for embedding API failures (default: 3).
"""
import os


def _int(key: str, default: int) -> int:
    """Read an integer env var with a default."""
    raw = os.getenv(key, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
"""Qdrant server URL. Empty string → use in-memory client for local dev."""

QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
"""API key for Qdrant Cloud. Leave empty for local / in-memory mode."""

QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")
"""Qdrant collection name. Defaults to 'documents'."""

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
"""Embedding provider: 'openai'. Extensible for future providers."""

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
"""Embedding model name. Defaults to OpenAI text-embedding-3-small."""

EMBEDDING_DIMENSIONS: int = _int("EMBEDDING_DIMENSIONS", 1536)
"""Vector dimensions — must match the embedding model. 1536 for text-embedding-3-small."""

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = _int("CHUNK_SIZE", 800)
"""Target chunk size in characters. Smaller = more precise retrieval, more chunks."""

CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 150)
"""Overlap between adjacent chunks in characters. Prevents context loss at boundaries."""

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = _int("TOP_K_RESULTS", 5)
"""Default number of top results to return per retrieval query."""

# ── Retry ─────────────────────────────────────────────────────────────────────
MAX_EMBED_RETRIES: int = _int("MAX_EMBED_RETRIES", 3)
"""Maximum retry attempts for embedding API failures before raising."""
