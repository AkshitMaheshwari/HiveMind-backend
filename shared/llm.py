"""
Centralized LLM factory for the Universal Multi-Agent Orchestrator.
- Gemini 2.0 Flash → CEO routing, department heads (fast + cheap)
- GPT-4o via Azure → Content/Code generation workers (high quality)
- Groq Llama-3 → Quick summarization fallback
"""
import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()


@lru_cache(maxsize=None)
def get_gemini_flash():
    """Fast Gemini model for routing and lightweight tasks."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
    )


@lru_cache(maxsize=None)
def get_gemini_pro():
    """Gemini 2.5 Pro for high-quality generation tasks."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.4,
    )


@lru_cache(maxsize=None)
def get_openai_gpt4o():
    """GPT-4o via Azure GitHub Models as fallback for content workers."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2000,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://models.inference.ai.azure.com",
    )


@lru_cache(maxsize=None)
def get_groq_llama():
    """Groq Llama-3 for fast summarization."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=1500,
    )


# ─── Convenience aliases used across all departments ──────────────────────────
def ceo_llm():
    """CEO uses Gemini Flash for fast routing decisions."""
    return get_gemini_flash()


def head_llm():
    """Department Heads use Gemini Flash for coordination."""
    return get_gemini_flash()


def worker_llm():
    """Production workers use Gemini Pro for quality output."""
    return get_gemini_pro()


def fast_llm():
    """Fast LLM for verification/checking tasks."""
    return get_gemini_flash()
