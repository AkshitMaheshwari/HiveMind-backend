"""
Centralized dynamic LLM factory for the Universal Multi-Agent Orchestrator.
Supports user-provided Gemini, OpenAI, or Groq API keys with automatic fallback.
"""
import hashlib
import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("llm_factory")

# ─── Module-level LLM Instance Cache ─────────────────────────────────────────
# Keyed on (api_key_hash, model_id, temperature) so LLM HTTP clients are
# created once per process and reused across all _make_agents() calls.
# This eliminates the 85-130s per-node overhead caused by re-instantiation.
_LLM_CACHE: Dict[str, "ResilientLLM"] = {}


def _cache_key(api_keys: Dict, model_type: str, selected_model: Optional[str]) -> str:
    """Stable cache key from api key fingerprints + model selection."""
    google = (api_keys.get("google_api_key") or os.getenv("GOOGLE_API_KEY") or "")[-8:]
    groq   = (api_keys.get("groq_api_key")   or os.getenv("GROQ_API_KEY")   or "")[-8:]
    openai = (api_keys.get("openai_api_key") or os.getenv("OPENAI_API_KEY") or "")[-8:]
    raw = f"{google}|{groq}|{openai}|{model_type}|{selected_model or ''}"
    return hashlib.md5(raw.encode()).hexdigest()


# ─── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gemini": [
        {"id": "gemini-3.5-flash-lite",         "name": "Gemini 3.5 Flash Lite",         "description": "Ultra-low latency execution for lightning fast throughput"},
        {"id": "gemini-3.5-flash",              "name": "Gemini 3.5 Flash",              "description": "Next-gen flagship multimodal model, high speed"},
        {"id": "gemini-3.7-flash",              "name": "Gemini 3.7 Flash",              "description": "Flagship hybrid reasoning & multimodal model, ultra-fast & intelligent"},
        {"id": "gemini-3.6-flash",              "name": "Gemini 3.6 Flash",              "description": "High-performance production model for complex workflows"},
        {"id": "gemini-flash-latest",           "name": "Gemini Flash Latest",           "description": "Latest versatile general-purpose performance"},
        {"id": "gemini-flash-lite-latest",      "name": "Gemini Flash Lite Latest",      "description": "Ultra-low latency execution for high throughput"},
        {"id": "gemini-3.1-pro-preview",        "name": "Gemini 3.1 Pro Preview",        "description": "Deep reasoning & complex multi-step analysis"},
        {"id": "gemini-3.1-flash-lite",         "name": "Gemini 3.1 Flash Lite",         "description": "Ultra-lightweight preview for fast tasks"},
        {"id": "gemini-3-flash-preview",        "name": "Gemini 3 Flash Preview",        "description": "Next-gen preview model with enhanced reasoning"},
        {"id": "gemini-3.1-flash-live-preview", "name": "Gemini 3.1 Flash Live Preview", "description": "Real-time low latency multimodal streaming preview"},
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
    if not model_id:
        return None
    for provider, models in MODEL_REGISTRY.items():
        if any(m["id"] == model_id for m in models):
            return provider
    if model_id.startswith("gemini"):
        return "gemini"
    if model_id.startswith("gpt") or model_id.startswith("o1"):
        return "openai"
    if "qwen" in model_id or "oss" in model_id or "compound" in model_id or "allam" in model_id:
        return "groq"
    return None


def _get_model_name(llm_obj) -> str:
    """Extract human-readable model identifier from raw LLM or wrapped Runnable."""
    if hasattr(llm_obj, "model"):
        return str(llm_obj.model)
    if hasattr(llm_obj, "model_name"):
        return str(llm_obj.model_name)
    bound = getattr(llm_obj, "bound", None)
    if bound and hasattr(bound, "model"):
        return str(bound.model)
    first = getattr(llm_obj, "first", None)
    if first and hasattr(first, "model"):
        return str(first.model)
    return "model"


class ResilientLLM:
    """Wraps multiple LLMs in a fallback chain to handle timeouts, 504 deadline exceeded, and 429 rate limit errors automatically."""
    def __init__(self, llms: list):
        self.llms = [l for l in llms if l is not None]

    def invoke(self, messages, **kwargs):
        last_err = None
        for llm in self.llms:
            try:
                return llm.invoke(messages, **kwargs)
            except Exception as e:
                m_name = _get_model_name(llm)
                logger.warning(f"LLM invocation failed on model '{m_name}': {e}. Retrying with next fallback model...")
                last_err = e
        raise last_err or RuntimeError("All LLM models in resilient chain failed.")

    async def ainvoke(self, messages, **kwargs):
        last_err = None
        for llm in self.llms:
            try:
                return await llm.ainvoke(messages, **kwargs)
            except Exception as e:
                m_name = _get_model_name(llm)
                logger.warning(f"LLM async invocation failed on model '{m_name}': {e}. Retrying with next fallback model...")
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
    return ChatGoogleGenerativeAI(model=model_id, google_api_key=api_key.strip(), temperature=temperature, max_retries=0, timeout=18)


def _build_groq(model_id: str, api_key: str, temperature: float):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model_id, groq_api_key=api_key.strip(), temperature=temperature, max_tokens=4096, max_retries=0, timeout=18)


def _build_openai(model_id: str, api_key: str, temperature: float):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_id, api_key=api_key.strip(), temperature=temperature, max_retries=0, timeout=18)


def get_llm(
    api_keys: Optional[Dict[str, str]] = None,
    model_type: str = "general",
    selected_model: Optional[str] = None,
    bypass_cache: bool = False,
):
    """
    Builds a ResilientLLM chain based on user-provided keys or system defaults.

    If `selected_model` is provided, that model is tried first, backed by same-provider fallbacks.
    The user's explicitly provided api_keys always take priority over .env defaults.
    """
    api_keys = api_keys or {}
    temp = 0.2 if model_type in ("routing", "verification") else 0.4

    # ── Cache lookup — skip build if we already have a live LLM for this config ─
    if not bypass_cache:
        key = _cache_key(api_keys, model_type, selected_model)
        if key in _LLM_CACHE:
            return _LLM_CACHE[key]

    # ── User selected a specific model ─────────────────────────────────────────
    if selected_model:
        provider = get_provider_for_model(selected_model)

        if provider == "gemini":
            api_key = api_keys.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
            if api_key and not api_key.startswith("your-"):
                models_to_try = [selected_model]
                # Add stable fallback models if primary model encounters 504 / quota issues
                for fallback in ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.6-flash"]:
                    if fallback not in models_to_try:
                        models_to_try.append(fallback)
                candidates = []
                for m in models_to_try:
                    try:
                        candidates.append(_build_gemini(m, api_key, temp))
                    except Exception as e:
                        logger.warning(f"Failed to build Gemini candidate '{m}': {e}")
                if candidates:
                    llm = ResilientLLM(candidates)
                    if not bypass_cache:
                        key = _cache_key(api_keys, model_type, selected_model)
                        _LLM_CACHE[key] = llm
                    return llm

        elif provider == "groq":
            api_key = api_keys.get("groq_api_key") or os.getenv("GROQ_API_KEY")
            if api_key and not api_key.startswith("your-"):
                models_to_try = [selected_model]
                for fallback in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
                    if fallback not in models_to_try:
                        models_to_try.append(fallback)
                candidates = []
                for m in models_to_try:
                    try:
                        candidates.append(_build_groq(m, api_key, temp))
                    except Exception as e:
                        logger.warning(f"Failed to build Groq candidate '{m}': {e}")
                if candidates:
                    llm = ResilientLLM(candidates)
                    if not bypass_cache:
                        key = _cache_key(api_keys, model_type, selected_model)
                        _LLM_CACHE[key] = llm
                    return llm

        elif provider == "openai":
            api_key = api_keys.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if api_key and api_key.startswith("sk-"):
                models_to_try = [selected_model]
                for fallback in ["gpt-4o", "gpt-4o-mini"]:
                    if fallback not in models_to_try:
                        models_to_try.append(fallback)
                candidates = []
                for m in models_to_try:
                    try:
                        candidates.append(_build_openai(m, api_key, temp))
                    except Exception as e:
                        logger.warning(f"Failed to build OpenAI candidate '{m}': {e}")
                if candidates:
                    llm = ResilientLLM(candidates)
                    if not bypass_cache:
                        key = _cache_key(api_keys, model_type, selected_model)
                        _LLM_CACHE[key] = llm
                    return llm

        logger.warning(
            f"Selected model '{selected_model}' could not be initialized "
            f"(provider={provider}, key provided={bool(api_keys)}). Falling back to default chain."
        )

    # ── Default fallback chain — used when NO model is selected or selection failed ─
    google_key = api_keys.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
    groq_key   = api_keys.get("groq_api_key")   or os.getenv("GROQ_API_KEY")
    openai_key = api_keys.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

    candidates = []

    if google_key and not google_key.startswith("your-"):
        for m in ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.6-flash"]:
            try:
                candidates.append(_build_gemini(m, google_key, temp))
            except Exception as e:
                logger.warning(f"Google Gemini initialization error for {m}: {e}")

    if groq_key and not groq_key.startswith("your-"):
        try:
            candidates.append(_build_groq("openai/gpt-oss-120b", groq_key, 0.1))
            candidates.append(_build_groq("openai/gpt-oss-20b", groq_key, 0.1))
        except Exception as e:
            logger.warning(f"Groq initialization error: {e}")

    if openai_key and openai_key.startswith("sk-"):
        try:
            candidates.append(_build_openai("gpt-4o", openai_key, temp))
            candidates.append(_build_openai("gpt-4o-mini", openai_key, temp))
        except Exception as e:
            logger.warning(f"OpenAI initialization error: {e}")

    if candidates:
        llm = ResilientLLM(candidates)
        if not bypass_cache:
            _LLM_CACHE[key] = llm
        return llm

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
