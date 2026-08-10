"""
Data Analyst Department State
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class DataAnalystDeptState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    task: str
    original_request: str
    user_id: Optional[str]               # ID of the user (for accessing uploaded files)
    api_keys: Optional[Dict[str, str]]   # User-provided API keys
    selected_model: Optional[str]        # User's chosen model id

    # ── Internal pipeline ──────────────────────────────────────────
    analysis_plan: Dict[str, Any] # Structured output of DataPlannerAgent
    eda_code: str                 # Generated EDA python code
    execution_stdout: str         # Output of EDA python code
    execution_stderr: str         # Error of EDA python code
    execution_success: bool       # Did EDA run cleanly?
    dataset_info: Dict[str, Any]  # Parsed JSON of dataset info from EDA stdout
    eda_results: Dict[str, Any]   # Parsed JSON of EDA results from EDA stdout
    insights: Dict[str, Any]      # Structured insights (anomalies, risks, recommendations)
    review: Dict[str, Any]        # Review output (feedback, scores)
    dashboard_code: str           # Generated dashboard code (HTML/JS)
    structured_output: Dict[str, Any] # The final massive structured JSON payload

    # ── Output ─────────────────────────────────────────────────────
    final_report: str             # Final markdown report with embedded dashboard/insights

    # ── Events ─────────────────────────────────────────────────────
    events: List[Dict[str, Any]]
