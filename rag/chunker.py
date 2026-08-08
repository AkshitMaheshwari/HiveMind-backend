"""
RAG chunker — splits documents using LangChain's RecursiveCharacterTextSplitter.
"""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split a large text string into smaller chunks with overlap using LangChain.

    Parameters:
        text: The input text to chunk. Must be non-empty.
        chunk_size: Maximum chunk length in characters.
        overlap: Character overlap between consecutive chunks.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If text is empty or whitespace.
    """
    if not text or not text.strip():
        raise ValueError("chunk_text: text must not be empty.")
    if chunk_size <= 0:
        raise ValueError("chunk_text: chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_text: overlap must be >= 0 and < chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_text(text)
    
    if not chunks:
        chunks = [text.strip()]
        
    return chunks
