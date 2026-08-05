"""
Research Department State — flows through the Research subgraph.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchDeptState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    task: str                      # The research task from CEO
    original_request: str          # Full original user request

    # ── Internal pipeline ──────────────────────────────────────────
    search_results: str            # Raw search output
    evidence: List[Dict[str, str]] # Structured evidence list
    draft_answer: str              # Initial synthesis
    fact_check_verdict: str        # "verified" | "needs_more"
    missing_info: List[str]        # Gaps identified by fact-checker

    # ── Output ─────────────────────────────────────────────────────
    final_research: str            # Final polished research output

    # ── Events ─────────────────────────────────────────────────────
    events: List[Dict[str, Any]]   # Events to bubble up to orchestrator
