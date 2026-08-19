"""
Analytics Department State — flows through the Analytics subgraph.
"""
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from shared.state_utils import merge_events


class AnalyticsDeptState(TypedDict):
    """State for the Analytics Department subgraph."""
    # ── Input ──────────────────────────────────────────────────────
    task: str                           # What to analyze / what question to answer
    data_source: str                    # CSV/JSON string or empty (will check RAG for uploaded files)
    user_id: Optional[str]             # For scoped RAG retrieval of uploaded CSVs
    api_keys: Optional[Dict[str, str]]
    selected_model: Optional[str]

    # ── Intermediate outputs ────────────────────────────────────────
    resolved_data: str                  # The actual data string after resolution
    profile_data: Dict[str, Any]        # Shape, dtypes, nulls, value counts
    cleaned_data: str                   # Data after outlier handling
    statistics: Dict[str, Any]          # Descriptive stats, correlation, KPIs
    charts_json: List[Dict[str, Any]]   # Chart specs for frontend renderer

    # ── Output ─────────────────────────────────────────────────────
    insights: str                       # Plain-English narrative of findings
    analysis_result: str                # Final combined output for the orchestrator

    # ── Streaming ──────────────────────────────────────────────────
    events: Annotated[List[Dict[str, Any]], merge_events]
