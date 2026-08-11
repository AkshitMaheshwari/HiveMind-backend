"""
RAG chunker — splits documents using LangChain's SemanticChunker.
"""
from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from rag.embedder import get_embedder


def chunk_text(text: str) -> List[str]:
    """
    Split a large text string into semantically grouped chunks using LangChain.

    Parameters:
        text: The input text to chunk. Must be non-empty.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If text is empty or whitespace.
    """
    if not text or not text.strip():
        raise ValueError("chunk_text: text must not be empty.")

    # Get the embedding model to use for semantic chunking
    embedder = get_embedder()
    
    # Initialize the semantic chunker
    splitter = SemanticChunker(embedder)
    
    # SemanticChunker creates Document objects, we need to extract the text
    docs = splitter.create_documents([text])
    chunks = [doc.page_content for doc in docs]
    
    if not chunks:
        chunks = [text.strip()]
        
    return chunks
