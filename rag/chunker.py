"""
RAG chunker — splits document text into overlapping chunks suitable for embedding.

Chunking strategy:
1. Split text into sentences (using regex — no NLTK dependency required).
2. Accumulate sentences into chunks up to ``CHUNK_SIZE`` characters.
3. Apply ``CHUNK_OVERLAP`` by re-including the tail of the previous chunk
   at the start of each new chunk.

This approach avoids cutting sentences mid-way, which preserves meaning at
chunk boundaries and improves retrieval quality.
"""
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Sentence boundary: period/exclamation/question followed by whitespace and a capital letter,
# or a newline that separates paragraphs.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?:\n\s*\n)")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks for embedding and vector storage.

    The chunker works at the sentence boundary level to avoid splitting
    meaningful phrases. Overlap ensures that queries spanning chunk boundaries
    still find relevant content.

    Parameters:
        text: The full document text to chunk. Must be non-empty.
        chunk_size: Target size for each chunk in characters (default 800).
                    Chunks may be slightly larger if a single sentence exceeds this.
        overlap: Number of characters from the end of the previous chunk to
                 prepend to the next chunk (default 150).

    Returns:
        A list of non-empty chunk strings. Will contain at least one chunk
        if the input is non-empty.

    Raises:
        ValueError: If ``text`` is empty or whitespace-only.
        ValueError: If ``chunk_size`` <= 0 or ``overlap`` < 0.
    """
    if not text or not text.strip():
        raise ValueError("chunk_text: text must be non-empty.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_text: chunk_size must be > 0, got {chunk_size}.")
    if overlap < 0:
        raise ValueError(f"chunk_text: overlap must be >= 0, got {overlap}.")
    if overlap >= chunk_size:
        logger.warning(
            "chunk_text: overlap (%d) >= chunk_size (%d), clamping overlap to chunk_size // 4",
            overlap,
            chunk_size,
        )
        overlap = chunk_size // 4

    # Split into sentences
    raw_sentences = _SENTENCE_BOUNDARY.split(text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        # Fallback: no sentence boundaries found — treat whole text as one chunk
        return [text.strip()[:chunk_size]]

    chunks: List[str] = []
    current_chunk = ""
    overlap_buffer = ""  # tail of the previous chunk to prepend

    for sentence in sentences:
        candidate = (overlap_buffer + " " + sentence).strip() if overlap_buffer else sentence

        if not current_chunk:
            current_chunk = candidate
        elif len(current_chunk) + 1 + len(sentence) <= chunk_size:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            # Flush current chunk
            if current_chunk:
                chunks.append(current_chunk)
                # Build overlap: take the last `overlap` chars from the flushed chunk
                overlap_buffer = current_chunk[-overlap:] if overlap > 0 else ""

            # Start new chunk, prepending overlap
            current_chunk = (overlap_buffer + " " + sentence).strip() if overlap_buffer else sentence

    # Flush the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    logger.debug(
        "chunk_text: produced %d chunks from %d chars (chunk_size=%d overlap=%d)",
        len(chunks),
        len(text),
        chunk_size,
        overlap,
    )
    return chunks
