"""
RAG chunker — splits documents using LangChain's SemanticChunker.
"""
from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from rag.embedder import get_embedder


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split a large text string into semantically grouped chunks using LangChain.

    Parameters:
        text: The input text to chunk. Must be non-empty.
        chunk_size: Optional target chunk size in characters (for fallback / compatibility).
        overlap: Optional character overlap between chunks.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If text is empty or whitespace, or if chunk_size <= 0.
    """
    if not text or not text.strip():
        raise ValueError("chunk_text: text must not be empty.")
    if chunk_size <= 0:
        raise ValueError("chunk_text: chunk_size must be greater than 0.")


    try:
        # Get the embedding model to use for semantic chunking
        embedder = get_embedder()
        splitter = SemanticChunker(embedder)
        docs = splitter.create_documents([text])
        chunks = [doc.page_content for doc in docs if doc.page_content.strip()]
        if chunks:
            return chunks
    except Exception:
        pass

    # Robust fallback: Character / recursive splitting using chunk_size & overlap
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    return chunks if chunks else [text.strip()]

