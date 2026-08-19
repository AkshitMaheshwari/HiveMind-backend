"""
Code Department State
"""
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from shared.state_utils import merge_events


class CodeDeptState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    task: str
    original_request: str
    api_keys: Optional[Dict[str, str]]   # User-provided API keys
    selected_model: Optional[str]        # User's chosen model id

    # ── Internal pipeline ──────────────────────────────────────────
    problem_description: str      # Parsed/clarified problem
    is_web_ui_task: bool          # Is this a front-end/UI task?
    ux_design_system: str         # Generated UX plan (color, typography, layout)
    generated_code: str           # Raw generated code
    execution_stdout: str         # Execution output
    execution_stderr: str         # Execution errors
    execution_success: bool       # Did it run cleanly?
    fixed_code: str               # Debugger's fix (if errors)
    ui_feedback: str              # Feedback from UIReviewerAgent
    ui_approved: bool             # Has UIReviewerAgent approved the UI?
    documentation: str            # Generated README/docstrings

    # ── Output ─────────────────────────────────────────────────────
    final_report: str

    # ── Events ─────────────────────────────────────────────────────
    events: Annotated[List[Dict[str, Any]], merge_events]
