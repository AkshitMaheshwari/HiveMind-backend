"""
Data Analyst Department State
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class DataAnalystDeptState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    task: str
    original_request: str
    api_keys: Optional[Dict[str, str]]   # User-provided API keys
    selected_model: Optional[str]        # User's chosen model id

    # ── Internal pipeline ──────────────────────────────────────────
    analysis_plan: str            # Output of DataPlannerAgent
    eda_code: str                 # Generated EDA python code
    execution_stdout: str         # Output of EDA python code
    execution_stderr: str         # Error of EDA python code
    execution_success: bool       # Did EDA run cleanly?
    insights: str                 # Textual insights generated from execution
    dashboard_code: str           # Generated dashboard code (e.g. Streamlit or HTML/JS)

    # ── Output ─────────────────────────────────────────────────────
    final_report: str             # Final markdown report with embedded dashboard/insights

    # ── Events ─────────────────────────────────────────────────────
    events: List[Dict[str, Any]]
