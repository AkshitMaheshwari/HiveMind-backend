"""
Research Department LangGraph subgraph.
Flow: web_search_node → fact_checker_node → synthesizer_node → [done]
"""
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from departments.research.state import ResearchDeptState
from departments.research.agents import WebSearchAgent, SummarizerAgent, FactCheckerAgent


# ─── Instantiate agents ───────────────────────────────────────────────────────
_web_search_agent = None
_summarizer_agent = None
_fact_checker_agent = None


def _get_agents():
    global _web_search_agent, _summarizer_agent, _fact_checker_agent
    if _web_search_agent is None:
        _web_search_agent = WebSearchAgent()
        _summarizer_agent = SummarizerAgent()
        _fact_checker_agent = FactCheckerAgent()
    return _web_search_agent, _summarizer_agent, _fact_checker_agent


def _emit(state: ResearchDeptState, event: str, agent: str, data: str = "") -> list:
    events = list(state.get("events", []))
    events.append({
        "event": event,
        "department": "research",
        "agent": agent,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return events


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

def web_search_node(state: ResearchDeptState) -> Dict[str, Any]:
    """Web Search Agent — finds information from the web."""
    web_search_agent, _, _ = _get_agents()

    events = _emit(state, "agent_working", "WebSearchAgent", "Searching the web...")

    output = web_search_agent.execute(state["task"])

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "WebSearchAgent",
        "Search complete",
    )

    return {
        "search_results": output.metadata.get("raw_results", ""),
        "evidence": output.metadata.get("evidence", []),
        "draft_answer": output.content,
        "events": events,
    }


def fact_checker_node(state: ResearchDeptState) -> Dict[str, Any]:
    """Fact Checker Agent — verifies the draft and identifies gaps."""
    _, _, fact_checker_agent = _get_agents()

    events = _emit(state, "agent_working", "FactCheckerAgent", "Verifying claims...")

    output = fact_checker_agent.execute(
        state["task"],
        context={
            "draft_answer": state.get("draft_answer", ""),
            "evidence": state.get("evidence", []),
        },
    )

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "FactCheckerAgent",
        f"Verification: {output.content}",
    )

    return {
        "fact_check_verdict": output.metadata.get("verdict", "verified"),
        "missing_info": output.metadata.get("gaps", []),
        "events": events,
    }


def synthesizer_node(state: ResearchDeptState) -> Dict[str, Any]:
    """Summarizer Agent — creates the final polished research report."""
    _, summarizer_agent, _ = _get_agents()

    events = _emit(state, "agent_working", "SummarizerAgent", "Synthesizing research...")

    output = summarizer_agent.execute(
        state["task"],
        context={
            "draft_answer": state.get("draft_answer", ""),
            "gaps": state.get("missing_info", []),
        },
    )

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "SummarizerAgent",
        "Research complete",
    )

    return {
        "final_research": output.content,
        "events": events,
    }


# ─── Build the Research Subgraph ──────────────────────────────────────────────

research_graph = StateGraph(ResearchDeptState)

research_graph.add_node("web_search_node", web_search_node)
research_graph.add_node("fact_checker_node", fact_checker_node)
research_graph.add_node("synthesizer_node", synthesizer_node)

research_graph.add_edge(START, "web_search_node")
research_graph.add_edge("web_search_node", "fact_checker_node")
research_graph.add_edge("fact_checker_node", "synthesizer_node")
research_graph.add_edge("synthesizer_node", END)

research_subgraph = research_graph.compile()


# ─── Outer node — plugs into root orchestrator graph ─────────────────────────

def research_department_node(state: "OrchestratorState") -> Dict[str, Any]:
    """
    Outer node that plugs into the main orchestrator graph.
    Extracts the research task from the CEO's plan, runs the subgraph,
    and writes results back into OrchestratorState.
    """
    from datetime import datetime

    # Find the research subtask from CEO plan
    subtasks = state.get("task_plan", {}).get("subtasks", [])
    research_task = state["user_request"]  # fallback
    for st in subtasks:
        if st.get("department") == "research":
            research_task = st.get("task", state["user_request"])
            break

    # Emit department started event
    events = list(state.get("agent_events", []))
    events.append({
        "event": "department_started",
        "department": "research",
        "agent": "Research Head",
        "data": f"Starting research: {research_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Run the subgraph
    initial_state = {
        "task": research_task,
        "original_request": state["user_request"],
        "events": [],
    }

    final_state = research_subgraph.invoke(initial_state)

    # Merge subgraph events into orchestrator events
    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done",
        "department": "research",
        "agent": "Research Head",
        "data": "Research department completed successfully",
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Update department outputs and completed list
    department_outputs = dict(state.get("department_outputs", {}))
    department_outputs["research"] = final_state.get("final_research", "Research completed.")

    completed = list(state.get("completed_departments", []))
    if "research" not in completed:
        completed.append("research")

    return {
        "department_outputs": department_outputs,
        "completed_departments": completed,
        "agent_events": events,
    }


# Resolve forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from orchestrator.state import OrchestratorState
