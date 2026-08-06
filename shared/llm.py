"""
Centralized dynamic LLM factory for the Universal Multi-Agent Orchestrator.
Supports user-provided Gemini, OpenAI, or Groq API keys with automatic fallback to Groq.
"""
import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("llm_factory")


def get_llm(api_keys: Optional[Dict[str, str]] = None, model_type: str = "general"):
    """
    Selects LLM based on user-provided keys or system defaults.
    Priority:
    1. User Google Gemini Key -> Gemini 2.0 Flash
    2. User OpenAI Key -> GPT-4o
    3. User Groq Key -> Llama 3.3 70B
    4. Default Server Groq Key -> Llama 3.3 70B
    5. Default Server Google Key -> Gemini 2.0 Flash
    """
    api_keys = api_keys or {}
    google_key = api_keys.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
    openai_key = api_keys.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    groq_key = api_keys.get("groq_api_key") or os.getenv("GROQ_API_KEY")

    # 1. User-supplied Google Gemini key
    if api_keys.get("google_api_key"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_keys["google_api_key"].strip(),
                temperature=0.2 if model_type in ("routing", "verification") else 0.4,
            )
        except Exception as e:
            logger.warning(f"User Google Gemini key initialization failed: {e}. Falling back to Groq.")

    # 2. User-supplied OpenAI key
    if api_keys.get("openai_api_key"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o",
                api_key=api_keys["openai_api_key"].strip(),
                temperature=0.2 if model_type in ("routing", "verification") else 0.4,
            )
        except Exception as e:
            logger.warning(f"User OpenAI key initialization failed: {e}. Falling back to Groq.")

    # 3. User-supplied Groq key
    if api_keys.get("groq_api_key"):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=api_keys["groq_api_key"].strip(),
                temperature=0.1,
                max_tokens=2000,
            )
        except Exception as e:
            logger.warning(f"User Groq key initialization failed: {e}. Falling back to default server key.")

    # 4. Default Server Groq key
    if groq_key and not groq_key.startswith("your-"):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_key.strip(),
                temperature=0.1,
                max_tokens=2000,
            )
        except Exception as e:
            logger.warning(f"Server Groq key failed: {e}")

    # 5. Default Server Google key
    if google_key and not google_key.startswith("your-"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=google_key.strip(),
                temperature=0.4,
            )
        except Exception as e:
            logger.warning(f"Server Google key failed: {e}")

    raise ValueError("No valid API key available! Please enter a Gemini, OpenAI, or Groq API key in Settings.")


# ─── Convenience Aliases ──────────────────────────────────────────────────────

def ceo_llm(api_keys: Optional[Dict[str, str]] = None):
    """CEO Agent LLM for routing."""
    return get_llm(api_keys, model_type="routing")


def head_llm(api_keys: Optional[Dict[str, str]] = None):
    """Department Head LLM for coordination."""
    return get_llm(api_keys, model_type="coordination")


def worker_llm(api_keys: Optional[Dict[str, str]] = None):
    """Worker Agent LLM for generation tasks."""
    return get_llm(api_keys, model_type="generation")


def fast_llm(api_keys: Optional[Dict[str, str]] = None):
    """Fast LLM for verification tasks."""
    return get_llm(api_keys, model_type="verification")


