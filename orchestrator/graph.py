"""
Root Orchestrator Graph — wires CEO + all department subgraphs + aggregator.

Graph structure:
  START → ceo_router → [dept_nodes...] → aggregator → END

Supports all 10 departments:
  Tier 1: research, content, code, document, financial, analytics, strategy
  Tier 2: legal, sales, design

For sequential tasks: CEO routes to first dept, then re-routes after each completes.
For parallel tasks: All departments run simultaneously via asyncio.gather,
  cutting total time from sum(dept_times) to max(dept_times).
"""
import sys
from pathlib import Path

# Ensure backend root is in sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import asyncio
from typing import Any, Dict, List, Literal

from langgraph.graph import StateGraph, START, END

from orchestrator.state import OrchestratorState
from orchestrator.ceo_agent import ceo_router_node, clarification_node
from orchestrator.aggregator import aggregator_node
from departments.research.graph import research_department_node
from departments.content.graph import content_department_node
from departments.code.graph import code_department_node
from departments.document.graph import document_department_node
from departments.financial.graph import financial_department_node
from departments.analytics.graph import analytics_department_node
from departments.strategy.graph import strategy_department_node
from departments.legal.graph import legal_department_node
from departments.sales.graph import sales_department_node
from departments.design.graph import design_department_node


# All departments supported by the orchestrator
_ALL_DEPTS = [
    "research", "content", "code", "document", "financial",
    "analytics", "strategy", "legal", "sales", "design",
]

# Map department name → its async node function (for parallel fan-out)
_DEPT_FN_MAP: Dict[str, Any] = {
    "research":  research_department_node,
    "content":   content_department_node,
    "code":      code_department_node,
    "document":  document_department_node,
    "financial": financial_department_node,
    "analytics": analytics_department_node,
    "strategy":  strategy_department_node,
    "legal":     legal_department_node,
    "sales":     sales_department_node,
    "design":    design_department_node,
}


async def _run_departments_parallel(state: OrchestratorState, depts: List[str]) -> OrchestratorState:
    """
    Run multiple departments concurrently via asyncio.gather.
    Each dept node receives the current state and returns a partial update.
    We merge all partial updates into a single combined state update.

    This is the core fix for Bug 2: instead of Research (409s) → Strategy (402s) = 811s,
    we run them simultaneously: max(409s, 402s) ≈ 409s.
    """
    dept_fns = [_DEPT_FN_MAP[d] for d in depts if d in _DEPT_FN_MAP]
    if not dept_fns:
        return state

    results = await asyncio.gather(
        *[fn(state) for fn in dept_fns],
        return_exceptions=True
    )

    # Merge all partial state updates
    merged_outputs = dict(state.get("department_outputs", {}))
    merged_completed = list(state.get("completed_departments", []))
    merged_events = list(state.get("agent_events", []))

    for dept, result in zip(depts, results):
        if isinstance(result, Exception):
            merged_events.append({
                "event": "department_error", "department": dept,
                "agent": "Orchestrator", "data": str(result),
            })
            continue
        # Absorb each dept's output into the merged state
        merged_outputs.update(result.get("department_outputs", {}))
        for c in result.get("completed_departments", []):
            if c not in merged_completed:
                merged_completed.append(c)
        merged_events.extend(result.get("agent_events", []))

    return {
        **state,
        "department_outputs": merged_outputs,
        "completed_departments": merged_completed,
        "agent_events": merged_events,
    }


# ─── Routing function: CEO → Departments ─────────────────────────────────────

def route_after_ceo(state: OrchestratorState) -> str:
    """After CEO plans, determine routing mode (clarify / parallel / sequential)."""
    if state.get("clarification_needed"):
        return "clarification_node"

    active = state.get("active_departments", [])
    if not active:
        return "aggregator_node"

    task_plan = state.get("task_plan", {})
    sequence = task_plan.get("sequence", "sequential")
    subtasks = task_plan.get("subtasks", [])

    # Check whether any subtask has an explicit dependency on another dept.
    # If none do, and we have multiple depts, run them in parallel.
    has_dependencies = any(st.get("depends_on") for st in subtasks)

    if sequence != "sequential" or (len(active) > 1 and not has_dependencies):
        # ── Parallel mode: all departments run simultaneously ──────────────────
        # route_after_ceo sends to a single parallel_departments_node which uses
        # asyncio.gather internally. This is the core of Bug 2's fix.
        return "parallel_departments_node"
    else:
        # ── Sequential mode: run one dept at a time ───────────────────────────
        completed = set(state.get("completed_departments", []))
        for st in subtasks:
            dept = st["department"]
            depends = st.get("depends_on")
            if dept in completed:
                continue
            if depends is None or depends in completed:
                return f"{dept}_department_node"
        return "aggregator_node"


async def parallel_departments_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Runs ALL active departments concurrently using asyncio.gather.
    Total time = max(dept_times) instead of sum(dept_times).
    """
    active = state.get("active_departments", [])
    completed = set(state.get("completed_departments", []))
    pending = [d for d in active if d not in completed]
    return await _run_departments_parallel(state, pending)


def route_after_department(state: OrchestratorState) -> str:
    """After any department completes, check if more sequential departments need to run."""
    task_plan = state.get("task_plan", {})
    active = state.get("active_departments", [])
    completed = set(state.get("completed_departments", []))
    sequence = task_plan.get("sequence", "sequential")
    subtasks = task_plan.get("subtasks", [])

    # Check if all departments are done
    all_done = all(dept in completed for dept in active)
    if all_done:
        return "aggregator_node"

    if sequence == "sequential":
        # Find next department to run
        for st in subtasks:
            dept = st["department"]
            depends = st.get("depends_on")
            if dept in completed:
                continue
            if depends is None or depends in completed:
                return f"{dept}_department_node"

    return "aggregator_node"


# Build routing map for ALL departments + the new parallel node
_DEPT_ROUTE_MAP = {f"{d}_department_node": f"{d}_department_node" for d in _ALL_DEPTS}
_DEPT_ROUTE_MAP["clarification_node"]      = "clarification_node"
_DEPT_ROUTE_MAP["aggregator_node"]         = "aggregator_node"
_DEPT_ROUTE_MAP["parallel_departments_node"] = "parallel_departments_node"


# ─── Build the Root Graph ─────────────────────────────────────────────────────

def build_orchestrator_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("ceo_router_node", ceo_router_node)
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("aggregator_node", aggregator_node)
    graph.add_node("parallel_departments_node", parallel_departments_node)

    # Register all department nodes (used for sequential mode)
    graph.add_node("research_department_node", research_department_node)
    graph.add_node("content_department_node", content_department_node)
    graph.add_node("code_department_node", code_department_node)
    graph.add_node("document_department_node", document_department_node)
    graph.add_node("financial_department_node", financial_department_node)
    graph.add_node("analytics_department_node", analytics_department_node)
    graph.add_node("strategy_department_node", strategy_department_node)
    graph.add_node("legal_department_node", legal_department_node)
    graph.add_node("sales_department_node", sales_department_node)
    graph.add_node("design_department_node", design_department_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.add_edge(START, "ceo_router_node")

    # CEO → parallel node, sequential first dept, or clarification
    graph.add_conditional_edges(
        "ceo_router_node",
        route_after_ceo,
        _DEPT_ROUTE_MAP,
    )

    # Parallel node always → aggregator (all depts done in one shot)
    graph.add_edge("parallel_departments_node", "aggregator_node")

    # After each sequential department → next department or aggregator
    for dept in _ALL_DEPTS:
        graph.add_conditional_edges(
            f"{dept}_department_node",
            route_after_department,
            _DEPT_ROUTE_MAP,
        )

    graph.add_edge("clarification_node", END)
    graph.add_edge("aggregator_node", END)

    return graph


# ─── Compile and export ───────────────────────────────────────────────────────
orchestrator_graph = build_orchestrator_graph()
compiled_graph = orchestrator_graph.compile()


async def run_orchestrator(user_request: str, conversation_id: str = "default") -> Dict[str, Any]:
    """
    Convenience function to run the full orchestrator pipeline.
    Returns the final OrchestratorState.
    """
    initial_state: OrchestratorState = {
        "user_request": user_request,
        "conversation_id": conversation_id,
        "task_plan": None,
        "active_departments": [],
        "completed_departments": [],
        "department_outputs": {},
        "agent_events": [],
        "final_output": "",
        "clarification_needed": False,
        "clarification_question": None,
        "error": None,
    }

    return await compiled_graph.ainvoke(initial_state)
