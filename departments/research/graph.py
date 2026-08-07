"""
Research Department LangGraph subgraph.
Flow: web_search_node → fact_checker_node → synthesizer_node → [done]
"""
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from departments.research.state import ResearchDeptState
from departments.research.agents import (
    ArxivResearchAgent,
    WikipediaAgent,
    WebSearchAgent,
    SummarizerAgent,
    FactCheckerAgent,
)


# ─── Agent factory ───────────────────────────────────────────────────────────

def _make_agents(api_keys=None, selected_model=None):
    """Create fresh agent instances with the given API keys and model."""
    return (
        ArxivResearchAgent(api_keys=api_keys, selected_model=selected_model),
        WikipediaAgent(api_keys=api_keys, selected_model=selected_model),
        WebSearchAgent(api_keys=api_keys, selected_model=selected_model),
        SummarizerAgent(api_keys=api_keys, selected_model=selected_model),
        FactCheckerAgent(api_keys=api_keys, selected_model=selected_model),
    )


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

def arxiv_node(state: ResearchDeptState) -> Dict[str, Any]:
    """Arxiv Research Agent — searches arXiv for scientific preprints and papers."""
    arxiv_agent, _, _, _, _ = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "ArxivResearchAgent", "Searching arXiv scientific papers...")
    output = arxiv_agent.execute(state["task"])

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "ArxivResearchAgent",
        "arXiv paper analysis complete",
    )

    return {
        "arxiv_evidence": output.metadata.get("evidence", []),
        "events": events,
    }


def wikipedia_node(state: ResearchDeptState) -> Dict[str, Any]:
    """Wikipedia Agent — searches Wikipedia for background domain context."""
    _, wikipedia_agent, _, _, _ = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "WikipediaAgent", "Searching Wikipedia knowledge base...")
    output = wikipedia_agent.execute(state["task"])

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "WikipediaAgent",
        "Wikipedia background search complete",
    )

    return {
        "wikipedia_evidence": output.metadata.get("evidence", []),
        "events": events,
    }


def web_search_node(state: ResearchDeptState) -> Dict[str, Any]:
    """Web Search Agent — finds real-time web intelligence and documentation."""
    _, _, web_search_agent, _, _ = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "WebSearchAgent", "Searching web and technical docs...")
    output = web_search_agent.execute(state["task"])

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "WebSearchAgent",
        "Web intelligence search complete",
    )

    # Combine evidence lists
    arxiv_ev = state.get("arxiv_evidence", [])
    wiki_ev = state.get("wikipedia_evidence", [])
    web_ev = output.metadata.get("evidence", [])
    all_evidence = arxiv_ev + wiki_ev + web_ev

    return {
        "search_results": output.metadata.get("raw_results", ""),
        "web_evidence": web_ev,
        "evidence": all_evidence,
        "draft_answer": output.content,
        "events": events,
    }


def fact_checker_node(state: ResearchDeptState) -> Dict[str, Any]:
    """Fact Checker Agent — verifies draft and checks academic & web evidence."""
    _, _, _, _, fact_checker_agent = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "FactCheckerAgent", "Cross-verifying evidence & claims...")

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
    """Summarizer Agent — synthesizes arXiv, Wikipedia, and Web evidence into a clear answer."""
    _, _, _, summarizer_agent, _ = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "SummarizerAgent", "Synthesizing Deep Research Report...")

    # Formulate contextual drafts
    arxiv_text = "\n".join([f"• [{e.get('source')}]: {e.get('summary')}" for e in state.get("arxiv_evidence", [])])
    wiki_text = "\n".join([f"• [{e.get('source')}]: {e.get('summary')}" for e in state.get("wikipedia_evidence", [])])

    output = summarizer_agent.execute(
        state["task"],
        context={
            "arxiv_draft": arxiv_text,
            "wiki_draft": wiki_text,
            "draft_answer": state.get("draft_answer", ""),
            "raw_results": state.get("search_results", ""),
            "gaps": state.get("missing_info", []),
        },
    )

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "SummarizerAgent",
        "Deep Research Report complete",
    )

    return {
        "final_research": output.content,
        "events": events,
    }


# ─── Build the Research Subgraph ──────────────────────────────────────────────

research_graph = StateGraph(ResearchDeptState)

research_graph.add_node("arxiv_node", arxiv_node)
research_graph.add_node("wikipedia_node", wikipedia_node)
research_graph.add_node("web_search_node", web_search_node)
research_graph.add_node("fact_checker_node", fact_checker_node)
research_graph.add_node("synthesizer_node", synthesizer_node)

research_graph.add_edge(START, "arxiv_node")
research_graph.add_edge("arxiv_node", "wikipedia_node")
research_graph.add_edge("wikipedia_node", "web_search_node")
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
        "api_keys": state.get("api_keys"),
        "selected_model": state.get("selected_model"),
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
