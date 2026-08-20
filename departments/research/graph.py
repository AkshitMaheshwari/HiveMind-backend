"""
Research Department LangGraph subgraph.
Clean synchronized flow:
  START → research_router_node → research_sources_node → fact_checker_node → synthesizer_node → END

Performance note: _make_agents() is called ONCE in the outer research_department_node
and agents are injected into inner nodes via closures — eliminating the per-node
re-instantiation overhead that previously caused 85-130s of wasted time per node.
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List

from langgraph.graph import StateGraph, START, END

from departments.research.state import ResearchDeptState
from departments.research.agents import (
    ArxivResearchAgent,
    WikipediaAgent,
    WebSearchAgent,
    SummarizerAgent,
    FactCheckerAgent,
    ResearchRouterAgent,
)


# ─── Agent factory ────────────────────────────────────────────────────────────
# Called ONCE per request from the outer department node.
# The LLM cache in shared/llm.py ensures the underlying HTTP clients are
# also reused across requests for the same api_key + model combination.

def _make_agents(api_keys=None, selected_model=None) -> Dict[str, Any]:
    """Build all research agents once. Returns a named dict for clarity."""
    return {
        "arxiv":      ArxivResearchAgent(api_keys=api_keys, selected_model=selected_model),
        "wiki":       WikipediaAgent(api_keys=api_keys, selected_model=selected_model),
        "web":        WebSearchAgent(api_keys=api_keys, selected_model=selected_model),
        "summarizer": SummarizerAgent(api_keys=api_keys, selected_model=selected_model),
        "fact":       FactCheckerAgent(api_keys=api_keys, selected_model=selected_model),
        "router":     ResearchRouterAgent(api_keys=api_keys, selected_model=selected_model),
    }


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


# ─── Closure-based node builders ──────────────────────────────────────────────
# Each builder accepts the pre-built agents dict and returns an async node fn.
# This is the key pattern that avoids per-node _make_agents() calls.

def _make_router_node(agents: Dict[str, Any]):
    async def research_router_node(state: ResearchDeptState) -> Dict[str, Any]:
        """Router Agent — decides which sources to search based on the query."""
        events = _emit(state, "agent_working", "ResearchRouterAgent",
                       "Analyzing query to route to appropriate knowledge sources...")
        output = await agents["router"].execute(state["task"])

        sources = output.metadata.get("sources", ["web_search_node"])
        reasoning = output.metadata.get("reasoning", "")

        events = _emit(
            {**state, "events": events}, "agent_done", "ResearchRouterAgent",
            f"Selected sources: {', '.join(sources)}. Reasoning: {reasoning}",
        )
        return {"active_sources": sources, "routing_reasoning": reasoning, "events": events}
    return research_router_node


def _make_sources_node(agents: Dict[str, Any]):
    async def research_sources_node(state: ResearchDeptState) -> Dict[str, Any]:
        """Runs all selected research sources concurrently and aggregates evidence."""
        sources = state.get("active_sources", ["web_search_node"])
        events = list(state.get("events", []))

        tasks = {}
        if "arxiv_node" in sources:
            events.append({"event": "agent_working", "department": "research",
                           "agent": "ArxivResearchAgent", "data": "Searching arXiv papers...",
                           "timestamp": datetime.utcnow().isoformat()})
            tasks["arxiv"] = agents["arxiv"].execute(state["task"])
        if "wikipedia_node" in sources:
            events.append({"event": "agent_working", "department": "research",
                           "agent": "WikipediaAgent", "data": "Searching Wikipedia...",
                           "timestamp": datetime.utcnow().isoformat()})
            tasks["wiki"] = agents["wiki"].execute(state["task"])
        if "web_search_node" in sources or not tasks:
            events.append({"event": "agent_working", "department": "research",
                           "agent": "WebSearchAgent", "data": "Searching web intelligence...",
                           "timestamp": datetime.utcnow().isoformat()})
            tasks["web"] = agents["web"].execute(state["task"])

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        task_keys = list(tasks.keys())

        arxiv_ev, wiki_ev, web_ev = [], [], []
        search_results = ""
        draft_answer = ""

        for idx, key in enumerate(task_keys):
            res = results[idx]
            if isinstance(res, Exception):
                continue
            if key == "arxiv":
                arxiv_ev = res.metadata.get("evidence", [])
            elif key == "wiki":
                wiki_ev = res.metadata.get("evidence", [])
            elif key == "web":
                web_ev = res.metadata.get("evidence", [])
                search_results = res.metadata.get("raw_results", "")
                draft_answer = res.content

        events.append({"event": "agent_done", "department": "research",
                       "agent": "WebSearchAgent", "data": "Knowledge source aggregation complete",
                       "timestamp": datetime.utcnow().isoformat()})
        return {
            "arxiv_evidence": arxiv_ev, "wikipedia_evidence": wiki_ev, "web_evidence": web_ev,
            "evidence": arxiv_ev + wiki_ev + web_ev,
            "search_results": search_results, "draft_answer": draft_answer, "events": events,
        }
    return research_sources_node


def _make_fact_checker_node(agents: Dict[str, Any]):
    async def fact_checker_node(state: ResearchDeptState) -> Dict[str, Any]:
        """Fact Checker Agent — verifies draft and checks academic & web evidence."""
        events = _emit(state, "agent_working", "FactCheckerAgent", "Cross-verifying evidence & claims...")
        output = await agents["fact"].execute(
            state["task"],
            context={"draft_answer": state.get("draft_answer", ""), "evidence": state.get("evidence", [])},
        )
        events = _emit({**state, "events": events}, "agent_done", "FactCheckerAgent",
                       f"Verification: {output.content}")
        return {
            "fact_check_verdict": output.metadata.get("verdict", "verified"),
            "missing_info": output.metadata.get("gaps", []),
            "events": events,
        }
    return fact_checker_node


def _make_synthesizer_node(agents: Dict[str, Any]):
    async def synthesizer_node(state: ResearchDeptState) -> Dict[str, Any]:
        """Summarizer Agent — synthesizes all evidence into a clear answer."""
        events = _emit(state, "agent_working", "SummarizerAgent", "Synthesizing Deep Research Report...")
        arxiv_text = "\n".join([f"• [{e.get('source')}]: {e.get('summary')}" for e in state.get("arxiv_evidence", [])])
        wiki_text  = "\n".join([f"• [{e.get('source')}]: {e.get('summary')}" for e in state.get("wikipedia_evidence", [])])
        output = await agents["summarizer"].execute(
            state["task"],
            context={
                "arxiv_draft": arxiv_text, "wiki_draft": wiki_text,
                "draft_answer": state.get("draft_answer", ""),
                "raw_results": f"Web Search:\n{state.get('search_results', '')}",
                "gaps": state.get("missing_info", []),
            },
        )
        events = _emit({**state, "events": events}, "agent_done", "SummarizerAgent",
                       "Deep Research Report complete")
        return {"final_research": output.content, "events": events}
    return synthesizer_node


# ─── Build the Research Subgraph ──────────────────────────────────────────────

def build_research_graph(agents: Dict[str, Any]) -> StateGraph:
    """
    Build a research subgraph wired with pre-instantiated agents.
    Agents are injected via closures so nodes never call _make_agents() themselves.
    """
    graph = StateGraph(ResearchDeptState)
    graph.add_node("research_router_node", _make_router_node(agents))
    graph.add_node("research_sources_node", _make_sources_node(agents))
    graph.add_node("fact_checker_node", _make_fact_checker_node(agents))
    graph.add_node("synthesizer_node", _make_synthesizer_node(agents))

    graph.add_edge(START, "research_router_node")
    graph.add_edge("research_router_node", "research_sources_node")
    graph.add_edge("research_sources_node", "fact_checker_node")
    graph.add_edge("fact_checker_node", "synthesizer_node")
    graph.add_edge("synthesizer_node", END)
    return graph.compile()


# ─── Outer node — plugs into root orchestrator graph ─────────────────────────

async def research_department_node(state: "OrchestratorState") -> Dict[str, Any]:
    """
    Outer node that plugs into the main orchestrator graph.
    Builds all agents ONCE here, constructs a fresh subgraph with those agents
    injected via closures, then runs it. This is the single point of instantiation.
    """
    subtasks = state.get("task_plan", {}).get("subtasks", [])
    research_task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "research":
            research_task = st.get("task", state["user_request"])
            break

    events = list(state.get("agent_events", []))
    events.append({
        "event": "department_started", "department": "research",
        "agent": "Research Head",
        "data": f"Starting research: {research_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    # ── Build agents ONCE — LLM cache in shared/llm.py deduplicates HTTP clients ──
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    subgraph = build_research_graph(agents)

    initial_state = {
        "task": research_task,
        "original_request": state["user_request"],
        "user_id": state.get("user_id"),
        "api_keys": state.get("api_keys"),
        "selected_model": state.get("selected_model"),
        "events": [],
    }

    final_state = await subgraph.ainvoke(initial_state)

    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done", "department": "research",
        "agent": "Research Head", "data": "Research department completed successfully",
        "timestamp": datetime.utcnow().isoformat(),
    })

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
