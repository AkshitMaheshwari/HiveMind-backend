"""
RAG embedder — provides LangChain OpenAIEmbeddings.
"""
import os
import logging
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

def get_embedder() -> OpenAIEmbeddings:
    """
    Returns a configured LangChain OpenAIEmbeddings instance.
    Uses OPENAI_API_KEY and optionally OPENAI_BASE_URL.
    """
    from rag.config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, MAX_EMBED_RETRIES
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    
    if not api_key:
        logger.warning("Embedder: OPENAI_API_KEY is not set.")
        
    kwargs = {
        "model": EMBEDDING_MODEL,
        "openai_api_key": api_key,
        "max_retries": MAX_EMBED_RETRIES,
        "dimensions": EMBEDDING_DIMENSIONS,
    }
    
    if base_url:
        kwargs["openai_api_base"] = base_url
        
    return OpenAIEmbeddings(**kwargs)
