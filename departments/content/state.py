"""
Content Department State
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ContentDeptState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    task: str                          # Content task from CEO
    original_request: str
    research_context: str              # Research output (if available)

    # ── Internal pipeline ──────────────────────────────────────────
    draft_content: str                 # Copywriter's draft
    seo_keywords: List[str]            # Keywords from SEO agent
    meta_description: str             # SEO meta description
    seo_optimized_content: str        # After SEO optimization
    edited_content: str               # After editor review

    # ── Output ─────────────────────────────────────────────────────
    final_content: str

    # ── Events ─────────────────────────────────────────────────────
    events: List[Dict[str, Any]]
