"""
Root Orchestrator Graph — wires CEO + all department subgraphs + aggregator.

Graph structure:
  START → ceo_router → [dept_nodes...] → aggregator → END

Supports all 10 departments:
  Tier 1: research, content, code, document, financial, analytics, strategy
  Tier 2: legal, sales, design

For sequential tasks: CEO routes to first dept, then re-routes after each completes.
For parallel tasks: CEO fans out to all depts simultaneously.
"""
import sys
from pathlib import Path

# Ensure backend root is in sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

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


# ─── Routing function: CEO → Departments ─────────────────────────────────────

def route_after_ceo(state: OrchestratorState) -> str:
    """After CEO plans, determine which department to run first (or clarify)."""
    if state.get("clarification_needed"):
        return "clarification_node"

    active = state.get("active_departments", [])
    if not active:
        return "aggregator_node"

    task_plan = state.get("task_plan", {})
    sequence = task_plan.get("sequence", "sequential")
    completed = set(state.get("completed_departments", []))
    subtasks = task_plan.get("subtasks", [])

    if sequence == "sequential" and subtasks:
        # Run the first uncompleted department whose dependency is met
        for st in subtasks:
            dept = st["department"]
            depends = st.get("depends_on")
            if dept in completed:
                continue
            if depends is None or depends in completed:
                return f"{dept}_department_node"

        # All done
        return "aggregator_node"
    else:
        # Parallel OR no subtasks: route to first uncompleted active department
        for dept in active:
            if dept not in completed:
                return f"{dept}_department_node"
        return "aggregator_node"


def route_after_department(state: OrchestratorState) -> str:
    """After any department completes, check if more departments need to run."""
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


# Build routing map for ALL departments
_DEPT_ROUTE_MAP = {f"{d}_department_node": f"{d}_department_node" for d in _ALL_DEPTS}
_DEPT_ROUTE_MAP["clarification_node"] = "clarification_node"
_DEPT_ROUTE_MAP["aggregator_node"] = "aggregator_node"


# ─── Build the Root Graph ─────────────────────────────────────────────────────

def build_orchestrator_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("ceo_router_node", ceo_router_node)
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("aggregator_node", aggregator_node)

    # Register all department nodes
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

    # CEO → First department (or clarification)
    graph.add_conditional_edges(
        "ceo_router_node",
        route_after_ceo,
        _DEPT_ROUTE_MAP,
    )

    # After each department → next department or aggregator
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
