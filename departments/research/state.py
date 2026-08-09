"""
Research Department State — flows through the Research subgraph.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchDeptState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    task: str                      # The research task from CEO
    original_request: str          # Full original user request
    user_id: str                   # Required for user-scoped RAG queries
    api_keys: Optional[Dict[str, str]]   # User-provided API keys
    selected_model: Optional[str]        # User's chosen model id

    # ── Internal pipeline ──────────────────────────────────────────
    active_sources: List[str]          # Sources selected by router (e.g. 'rag', 'arxiv', 'wikipedia', 'web')
    routing_reasoning: str             # Explanation of why the router selected these sources
    rag_fallback_triggered: bool       # True if RAG retrieved poor results and fallback to web is needed
    search_results: str                # Raw web search output
    arxiv_evidence: List[Dict[str, Any]]  # Scientific paper findings from arXiv
    wikipedia_evidence: List[Dict[str, Any]] # Domain overview & background from Wikipedia
    web_evidence: List[Dict[str, Any]] # General web & industry search findings
    evidence: List[Dict[str, Any]]     # Aggregated evidence list across all sources
    draft_answer: str                  # Initial synthesis
    fact_check_verdict: str            # "verified" | "needs_more"
    missing_info: List[str]            # Gaps identified by fact-checker

    # ── Output ─────────────────────────────────────────────────────
    final_research: str                # Final polished research output

    # ── Events ─────────────────────────────────────────────────────
    events: List[Dict[str, Any]]       # Events to bubble up to orchestrator
