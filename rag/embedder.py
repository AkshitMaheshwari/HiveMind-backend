"""
RAG embedder — provides LangChain HuggingFaceEndpointEmbeddings.
"""
import os
import logging
from langchain_huggingface import HuggingFaceEndpointEmbeddings

logger = logging.getLogger(__name__)

def get_embedder() -> HuggingFaceEndpointEmbeddings:
    """
    Returns a configured LangChain HuggingFaceEndpointEmbeddings instance.
    Uses HF_TOKEN and EMBEDDING_MODEL.
    """
    from rag.config import EMBEDDING_MODEL
    
    hf_token = os.getenv("HF_TOKEN", "")
    
    if not hf_token:
        logger.warning("Embedder: HF_TOKEN is not set.")
        
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=hf_token,
    )
