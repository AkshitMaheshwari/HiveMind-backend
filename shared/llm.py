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


class ResilientLLM:
    """Wraps multiple LLMs in a fallback chain to handle 429 quota / rate limit errors automatically."""
    def __init__(self, llms: list):
        self.llms = [l for l in llms if l is not None]

    def invoke(self, messages, **kwargs):
        last_err = None
        for llm in self.llms:
            try:
                return llm.invoke(messages, **kwargs)
            except Exception as e:
                logger.warning(f"LLM invocation failed on model {getattr(llm, 'model', 'unknown')}: {e}. Retrying with next model...")
                last_err = e
        raise last_err or RuntimeError("All LLM models in resilient chain failed.")

    def with_structured_output(self, schema, **kwargs):
        structured_llms = []
        for l in self.llms:
            try:
                structured_llms.append(l.with_structured_output(schema, **kwargs))
            except Exception:
                pass
        return ResilientLLM(structured_llms)


def get_llm(api_keys: Optional[Dict[str, str]] = None, model_type: str = "general"):
    """
    Builds a ResilientLLM chain based on user-provided keys or system defaults.
    Candidates across independent quota buckets:
    1. Google Gemini 2.0 Flash
    2. Google Gemini 2.0 Flash Lite
    3. Google Gemini 1.5 Pro
    4. Groq Llama 3.1 8B Instant (500k TPD limit)
    5. Groq Llama 3.3 70B Versatile (100k TPD limit)
    6. OpenAI GPT-4o (if sk- key provided)
    """
    api_keys = api_keys or {}
    google_key = api_keys.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
    openai_key = api_keys.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    groq_key = api_keys.get("groq_api_key") or os.getenv("GROQ_API_KEY")

    candidates = []
    temp = 0.2 if model_type in ("routing", "verification") else 0.4

    # 1. Google Gemini Candidates
    if google_key and not google_key.startswith("your-"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            g_key = google_key.strip()
            candidates.append(ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=g_key, temperature=temp))
            candidates.append(ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=g_key, temperature=temp))
            candidates.append(ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=g_key, temperature=temp))
        except Exception as e:
            logger.warning(f"Google Gemini initialization error: {e}")

    # 2. Groq Candidates (8b instant has 500k TPD bucket; 70b has 100k TPD bucket)
    if groq_key and not groq_key.startswith("your-"):
        try:
            from langchain_groq import ChatGroq
            gq_key = groq_key.strip()
            candidates.append(ChatGroq(model="llama-3.1-8b-instant", groq_api_key=gq_key, temperature=0.1, max_tokens=2000))
            candidates.append(ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=gq_key, temperature=0.1, max_tokens=2000))
        except Exception as e:
            logger.warning(f"Groq initialization error: {e}")

    # 3. OpenAI Candidate (requires valid sk- key)
    if openai_key and openai_key.startswith("sk-"):
        try:
            from langchain_openai import ChatOpenAI
            candidates.append(ChatOpenAI(model="gpt-4o", api_key=openai_key.strip(), temperature=temp))
        except Exception as e:
            logger.warning(f"OpenAI initialization error: {e}")

    if candidates:
        return ResilientLLM(candidates)

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


