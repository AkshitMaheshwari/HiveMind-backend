"""
Master state schema for the Universal Multi-Agent Orchestrator.
This state flows through the entire root LangGraph.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class TaskPlan(TypedDict):
    """CEO's routing plan — structured JSON output."""
    departments: List[str]           # e.g. ["research", "content"]
    sequence: str                    # "sequential" | "parallel"
    subtasks: List[Dict[str, Any]]  # [{department, task, depends_on}]
    reasoning: str                   # CEO's chain-of-thought


class AgentEvent(TypedDict):
    """Real-time streaming event sent via WebSocket."""
    event: str       # e.g. "department_started", "agent_working", "final_output"
    department: Optional[str]
    agent: Optional[str]
    data: Optional[str]
    timestamp: Optional[str]


class OrchestratorState(TypedDict):
    """
    The single source of truth flowing through the entire graph.
    All department subgraphs read from and write to this.
    """
    # ── Input ──────────────────────────────────────────────────────
    user_request: str
    conversation_id: str
    api_keys: Optional[Dict[str, str]]
    selected_model: Optional[str]  # e.g. "gemini-2.0-flash", "llama-3.3-70b-versatile"

    # ── CEO Routing ────────────────────────────────────────────────
    task_plan: Optional[Dict[str, Any]]       # CEO's JSON task plan
    active_departments: List[str]             # departments to run
    completed_departments: List[str]          # departments that finished

    # ── Department Outputs ─────────────────────────────────────────
    department_outputs: Dict[str, Any]        # {"research": "...", "content": "..."}

    # ── Streaming Events ───────────────────────────────────────────
    agent_events: List[Dict[str, Any]]        # All events for WebSocket stream

    # ── Output ─────────────────────────────────────────────────────
    final_output: str

    # ── Error / Clarification ──────────────────────────────────────
    clarification_needed: bool
    clarification_question: Optional[str]
    error: Optional[str]
