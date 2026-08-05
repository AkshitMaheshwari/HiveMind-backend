"""
Code Department State
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class CodeDeptState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    task: str
    original_request: str

    # ── Internal pipeline ──────────────────────────────────────────
    problem_description: str      # Parsed/clarified problem
    generated_code: str           # Raw generated code
    execution_stdout: str         # Execution output
    execution_stderr: str         # Execution errors
    execution_success: bool       # Did it run cleanly?
    fixed_code: str               # Debugger's fix (if errors)
    documentation: str            # Generated README/docstrings

    # ── Output ─────────────────────────────────────────────────────
    final_report: str

    # ── Events ─────────────────────────────────────────────────────
    events: List[Dict[str, Any]]
