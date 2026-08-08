"""
RAG embedder — generates vector embeddings for text chunks.

Uses OpenAI's embedding API with tenacity retry/backoff for resilience against
rate limits and transient timeouts. If all retries are exhausted, raises
``EmbeddingError`` — the caller (ingestion pipeline) must then abort the
entire document, not silently store a partial result.

Configuration (from rag.config):
    EMBEDDING_MODEL:      OpenAI model name (default: text-embedding-3-small)
    EMBEDDING_DIMENSIONS: Vector dimension count (default: 1536)
    MAX_EMBED_RETRIES:    Retry attempts before giving up (default: 3)
"""
import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """
    Raised when the embedding API fails after all retries are exhausted.

    Attributes:
        message: Human-readable description of the failure.
        cause: The underlying exception that triggered the final failure.
    """

    def __init__(self, message: str, cause: Exception = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


class Embedder:
    """
    Generates text embeddings using OpenAI's embedding API.

    Retries on rate limit and timeout errors using exponential backoff.
    Raises :class:`EmbeddingError` if all retries are exhausted.

    Parameters:
        model: OpenAI embedding model name (overrides env var if provided).
        dimensions: Vector dimension count (overrides env var if provided).
        max_retries: Max retry attempts (overrides env var if provided).
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
    """

    def __init__(
        self,
        model: str = "",
        dimensions: int = 0,
        max_retries: int = 0,
        api_key: str = "",
        base_url: str = "",
    ) -> None:
        from rag.config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, MAX_EMBED_RETRIES

        self._model = model or EMBEDDING_MODEL
        self._dimensions = dimensions or EMBEDDING_DIMENSIONS
        self._max_retries = max_retries or MAX_EMBED_RETRIES
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL", "")

        if not self._api_key:
            logger.warning(
                "Embedder: OPENAI_API_KEY is not set. "
                "Embedding calls will fail unless an API key is provided at call time."
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings.

        Retries on ``openai.RateLimitError``, ``openai.APITimeoutError``, and
        ``httpx.TimeoutException`` using exponential backoff via tenacity.

        Parameters:
            texts: A non-empty list of text strings to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same order.

        Raises:
            ValueError: If ``texts`` is empty.
            EmbeddingError: If the API fails after all retry attempts.
        """
        if not texts:
            raise ValueError("embed_texts: texts list must not be empty.")

        # Filter out empty strings to avoid API errors
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("embed_texts: all texts are empty or whitespace.")

        logger.debug(
            "Embedder.embed_texts: embedding %d texts with model=%s",
            len(valid_texts),
            self._model,
        )

        return self._embed_with_retry(valid_texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Parameters:
            query: The query text to embed.

        Returns:
            A single embedding vector.

        Raises:
            ValueError: If ``query`` is empty.
            EmbeddingError: If the API fails after all retry attempts.
        """
        if not query or not query.strip():
            raise ValueError("embed_query: query must not be empty.")
        results = self.embed_texts([query])
        return results[0]

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Internal: call the OpenAI embedding API with tenacity retry.

        Parameters:
            texts: Validated, non-empty list of text strings.

        Returns:
            A list of float vectors.

        Raises:
            EmbeddingError: After all retries are exhausted.
        """
        try:
            import openai
            from tenacity import (
                retry,
                stop_after_attempt,
                wait_exponential,
                retry_if_exception_type,
                before_sleep_log,
            )
        except ImportError as exc:
            raise EmbeddingError(
                "Required packages not installed: openai, tenacity. "
                "Run: pip install openai tenacity",
                cause=exc,
            )

        retryable_errors = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        )

        # Build the retry-decorated function inline so we can use instance variables
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(retryable_errors),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        def _call() -> List[List[float]]:
            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
                
            client = openai.OpenAI(**client_kwargs)
            response = client.embeddings.create(
                input=texts,
                model=self._model,
            )
            return [item.embedding for item in response.data]

        try:
            return _call()
        except retryable_errors as exc:
            logger.error(
                "Embedder: all %d retries exhausted for embedding %d texts: %s",
                self._max_retries,
                len(texts),
                exc,
            )
            raise EmbeddingError(
                f"Embedding API failed after {self._max_retries} attempts: {exc}",
                cause=exc,
            )
        except openai.OpenAIError as exc:
            logger.error(
                "Embedder: non-retryable OpenAI error while embedding %d texts: %s",
                len(texts),
                exc,
                exc_info=True,
            )
            raise EmbeddingError(
                f"Embedding API error (non-retryable): {exc}",
                cause=exc,
            )
        except Exception as exc:
            logger.error(
                "Embedder: unexpected error while embedding %d texts: %s",
                len(texts),
                exc,
                exc_info=True,
            )
            raise EmbeddingError(
                f"Unexpected embedding error: {exc}",
                cause=exc,
            )
