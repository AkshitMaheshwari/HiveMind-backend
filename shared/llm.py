"""
Centralized dynamic LLM factory for the Universal Multi-Agent Orchestrator.
Supports user-provided Gemini, OpenAI, or Groq API keys with automatic fallback.
"""
import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("llm_factory")


# ─── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gemini": [
        {"id": "gemini-3.5-flash",              "name": "Gemini 3.5 Flash",              "description": "Next-gen flagship multimodal model, high speed & intelligence"},
        {"id": "gemini-3.1-flash-lite",         "name": "Gemini 3.1 Flash Lite",         "description": "Ultra-lightweight & lightning fast for high-throughput tasks"},
        {"id": "gemini-3-flash-preview",        "name": "Gemini 3 Flash Preview",        "description": "Next-gen preview model with enhanced reasoning"},
        {"id": "gemini-3.1-flash-live-preview",   "name": "Gemini 3.1 Flash Live Preview", "description": "Real-time low latency multimodal streaming preview"},
        {"id": "gemini-2.5-pro",                "name": "Gemini 2.5 Pro",                "description": "Deep reasoning & complex multi-step analysis"},
        {"id": "gemini-2.5-flash",              "name": "Gemini 2.5 Flash",              "description": "Fast and versatile general-purpose performance"},
        {"id": "gemini-2.5-flash-lite",         "name": "Gemini 2.5 Flash Lite",         "description": "Cost-effective, low latency execution"},
    ],
    "groq": [
        {"id": "openai/gpt-oss-120b", "name": "GPT OSS 120B", "description": "Flagship open-weights on Groq, 500 T/s, 131k context"},
        {"id": "openai/gpt-oss-20b",  "name": "GPT OSS 20B",  "description": "Ultra-fast open weights on Groq, 1000 T/s, 131k context"},
        {"id": "qwen/qwen3.6-27b",    "name": "Qwen 3.6 27B",  "description": "Alibaba reasoning model, 131k context, tools support"},
        {"id": "groq/compound",       "name": "Groq Compound", "description": "Groq engineered multi-agent system, 131k context"},
        {"id": "groq/compound-mini",  "name": "Compound Mini", "description": "Fast lightweight compound model, 131k context"},
        {"id": "allam-2-7b",          "name": "ALLaM 2 7B",    "description": "SDAIA bilingual Arabic/English model, 4k context"},
    ],
    "openai": [
        {"id": "gpt-4o",       "name": "GPT-4o",       "description": "Best OpenAI model, multimodal"},
        {"id": "gpt-4o-mini",  "name": "GPT-4o Mini",  "description": "Affordable, fast, intelligent"},
        {"id": "gpt-4-turbo",  "name": "GPT-4 Turbo",  "description": "GPT-4 with 128k context"},
        {"id": "gpt-3.5-turbo","name": "GPT-3.5 Turbo","description": "Fast and cost-effective"},
        {"id": "o1-mini",      "name": "o1 Mini",       "description": "Reasoning model, STEM tasks"},
    ],
}


def get_provider_for_model(model_id: str) -> Optional[str]:
    """Determine which provider a model belongs to."""
    for provider, models in MODEL_REGISTRY.items():
        if any(m["id"] == model_id for m in models):
            return provider
    return None


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

    async def ainvoke(self, messages, **kwargs):
        last_err = None
        for llm in self.llms:
            try:
                return await llm.ainvoke(messages, **kwargs)
            except Exception as e:
                logger.warning(f"LLM async invocation failed on model {getattr(llm, 'model', 'unknown')}: {e}. Retrying with next model...")
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


def _build_gemini(model_id: str, api_key: str, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model_id, google_api_key=api_key.strip(), temperature=temperature, max_retries=1, timeout=15)


def _build_groq(model_id: str, api_key: str, temperature: float):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model_id, groq_api_key=api_key.strip(), temperature=temperature, max_tokens=4096, max_retries=1, timeout=15)


def _build_openai(model_id: str, api_key: str, temperature: float):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_id, api_key=api_key.strip(), temperature=temperature, max_retries=1, timeout=15)


def get_llm(
    api_keys: Optional[Dict[str, str]] = None,
    model_type: str = "general",
    selected_model: Optional[str] = None,
):
    """
    Builds a ResilientLLM chain based on user-provided keys or system defaults.

    If `selected_model` is provided, ONLY that model + its provider's key is used.
    The user's explicitly provided api_keys always take priority over .env defaults.
    """
    api_keys = api_keys or {}

    temp = 0.2 if model_type in ("routing", "verification") else 0.4

    # ── User selected a specific model ─────────────────────────────────────────
    if selected_model:
        provider = get_provider_for_model(selected_model)

        # When a model is explicitly selected, ONLY use the user-provided key
        # Never fall back to .env keys — that would defeat the purpose of selection
        if provider == "gemini":
            key = api_keys.get("google_api_key")
            if not key:
                key = os.getenv("GOOGLE_API_KEY")
            if key and not key.startswith("your-"):
                try:
                    return ResilientLLM([_build_gemini(selected_model, key, temp)])
                except Exception as e:
                    logger.warning(f"Failed to init selected Gemini model '{selected_model}': {e}")

        elif provider == "groq":
            key = api_keys.get("groq_api_key")
            if not key:
                key = os.getenv("GROQ_API_KEY")
            if key and not key.startswith("your-"):
                try:
                    return ResilientLLM([_build_groq(selected_model, key, temp)])
                except Exception as e:
                    logger.warning(f"Failed to init selected Groq model '{selected_model}': {e}")

        elif provider == "openai":
            key = api_keys.get("openai_api_key")
            if not key:
                key = os.getenv("OPENAI_API_KEY")
            if key and key.startswith("sk-"):
                try:
                    return ResilientLLM([_build_openai(selected_model, key, temp)])
                except Exception as e:
                    logger.warning(f"Failed to init selected OpenAI model '{selected_model}': {e}")

        logger.warning(
            f"Selected model '{selected_model}' could not be initialized "
            f"(provider={provider}, key provided={bool(api_keys)}). Falling back to default chain."
        )

    # ── Default fallback chain — only used when NO model is selected ───────────
    # Priority: user-provided keys FIRST, then .env fallbacks
    google_key = api_keys.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
    openai_key = api_keys.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    groq_key   = api_keys.get("groq_api_key")   or os.getenv("GROQ_API_KEY")

    candidates = []

    if google_key and not google_key.startswith("your-"):
        try:
            candidates.append(_build_gemini("gemini-2.5-flash", google_key, temp))
            candidates.append(_build_gemini("gemini-3.5-flash", google_key, temp))
            candidates.append(_build_gemini("gemini-2.5-pro", google_key, temp))
        except Exception as e:
            logger.warning(f"Google Gemini initialization error: {e}")

    if groq_key and not groq_key.startswith("your-"):
        try:
            candidates.append(_build_groq("openai/gpt-oss-20b", groq_key, 0.1))
            candidates.append(_build_groq("openai/gpt-oss-120b", groq_key, 0.1))
        except Exception as e:
            logger.warning(f"Groq initialization error: {e}")

    if openai_key and openai_key.startswith("sk-"):
        try:
            candidates.append(_build_openai("gpt-4o", openai_key, temp))
        except Exception as e:
            logger.warning(f"OpenAI initialization error: {e}")

    if candidates:
        return ResilientLLM(candidates)

    raise ValueError("No valid API key available! Please enter a Gemini, OpenAI, or Groq API key in Settings.")


# ─── Convenience Aliases ──────────────────────────────────────────────────────

def ceo_llm(api_keys: Optional[Dict[str, str]] = None, selected_model: Optional[str] = None):
    """CEO Agent LLM for routing."""
    return get_llm(api_keys, model_type="routing", selected_model=selected_model)


def head_llm(api_keys: Optional[Dict[str, str]] = None, selected_model: Optional[str] = None):
    """Department Head LLM for coordination."""
    return get_llm(api_keys, model_type="coordination", selected_model=selected_model)


def worker_llm(api_keys: Optional[Dict[str, str]] = None, selected_model: Optional[str] = None):
    """Worker Agent LLM for generation tasks."""
    return get_llm(api_keys, model_type="generation", selected_model=selected_model)


def fast_llm(api_keys: Optional[Dict[str, str]] = None, selected_model: Optional[str] = None):
    """Fast LLM for verification tasks."""
    return get_llm(api_keys, model_type="verification", selected_model=selected_model)
