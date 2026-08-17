"""
Analytics Department LangGraph subgraph.

Flow:
  START → profiler_node → cleaner_node → [stats_node || chart_node] → narrator_node → END

Stats and Charts run in parallel (independent of each other).
Data source resolution: RAG-uploaded CSVs first, then inline data from user_request.
"""
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

from langgraph.graph import StateGraph, START, END

from departments.analytics.state import AnalyticsDeptState
from departments.analytics.agents import (
    DataProfilerAgent,
    DataCleanerAgent,
    StatisticsAgent,
    ChartGeneratorAgent,
    InsightNarratorAgent,
)


# ─── Agent factory ───────────────────────────────────────────────────────────

def _make_agents(api_keys=None, selected_model=None):
    return {
        "profiler": DataProfilerAgent(api_keys=api_keys, selected_model=selected_model),
        "cleaner": DataCleanerAgent(api_keys=api_keys, selected_model=selected_model),
        "stats": StatisticsAgent(api_keys=api_keys, selected_model=selected_model),
        "charts": ChartGeneratorAgent(api_keys=api_keys, selected_model=selected_model),
        "narrator": InsightNarratorAgent(api_keys=api_keys, selected_model=selected_model),
    }


def _emit(state: AnalyticsDeptState, event: str, agent: str, data: str = "") -> list:
    events = list(state.get("events", []))
    events.append({
        "event": event,
        "department": "analytics",
        "agent": agent,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return events


# ─── Data resolution helper ──────────────────────────────────────────────────

async def _resolve_data(state: AnalyticsDeptState) -> str:
    """
    Resolve the actual data to analyze.
    Priority:
    1. Inline data_source from state (user pasted CSV/JSON)
    2. Uploaded CSVs from RAG (if user_id is set)
    3. Empty string (agents will report no data)
    """
    inline = state.get("data_source", "").strip()
    if inline:
        return inline

    user_id = state.get("user_id")
    if user_id:
        try:
            from shared.tools.rag_retrieval import rag_document_search
            results = await asyncio.to_thread(
                rag_document_search,
                query=state.get("task", ""),
                user_id=user_id,
                top_k=3,
            )
            if results and results.strip():
                return results
        except Exception:
            pass

    return ""


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

async def data_resolver_node(state: AnalyticsDeptState) -> Dict[str, Any]:
    """Resolves the data source (inline CSV or uploaded file via RAG)."""
    events = _emit(state, "agent_working", "DataResolver", "Resolving data source...")
    resolved = await _resolve_data(state)
    data_summary = f"{len(resolved)} characters of data resolved" if resolved else "No data found"
    events = _emit({**state, "events": events}, "agent_done", "DataResolver", data_summary)
    return {"resolved_data": resolved, "events": events}


async def profiler_node(state: AnalyticsDeptState) -> Dict[str, Any]:
    """DataProfilerAgent — profiles the resolved dataset."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "DataProfilerAgent", "Profiling dataset...")
    output = await agents["profiler"].execute(
        state["task"],
        context={"data_source": state.get("resolved_data", state.get("data_source", ""))}
    )
    events = _emit({**state, "events": events}, "agent_done", "DataProfilerAgent", output.content[:120])
    return {"profile_data": output.metadata.get("profile", {}), "events": events}


async def cleaner_node(state: AnalyticsDeptState) -> Dict[str, Any]:
    """DataCleanerAgent — cleans and deduplicates the dataset."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "DataCleanerAgent", "Cleaning data...")
    output = await agents["cleaner"].execute(
        state["task"],
        context={"data_source": state.get("resolved_data", state.get("data_source", ""))}
    )
    events = _emit({**state, "events": events}, "agent_done", "DataCleanerAgent", output.content[:120])
    cleaned = output.metadata.get("cleaned_data", state.get("resolved_data", ""))
    return {"cleaned_data": cleaned, "events": events}


async def stats_node(state: AnalyticsDeptState) -> Dict[str, Any]:
    """StatisticsAgent — descriptive stats, correlation, outliers, KPIs."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "StatisticsAgent", "Computing statistics...")
    output = await agents["stats"].execute(
        state["task"],
        context={"cleaned_data": state.get("cleaned_data", ""), "data_source": state.get("resolved_data", "")}
    )
    events = _emit({**state, "events": events}, "agent_done", "StatisticsAgent", "Statistics complete")
    return {"statistics": output.metadata.get("statistics", {}), "events": events}


async def chart_node(state: AnalyticsDeptState) -> Dict[str, Any]:
    """ChartGeneratorAgent — produces charts_json for frontend rendering."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "ChartGeneratorAgent", "Generating charts...")
    output = await agents["charts"].execute(
        state["task"],
        context={"cleaned_data": state.get("cleaned_data", ""), "data_source": state.get("resolved_data", "")}
    )
    events = _emit({**state, "events": events}, "agent_done", "ChartGeneratorAgent", output.content[:80])
    return {"charts_json": output.metadata.get("charts_json", []), "events": events}


async def narrator_node(state: AnalyticsDeptState) -> Dict[str, Any]:
    """InsightNarratorAgent — synthesizes everything into a plain-English narrative."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "InsightNarratorAgent", "Narrating insights...")
    output = await agents["narrator"].execute(
        state["task"],
        context={
            "profile_data": state.get("profile_data", {}),
            "statistics": state.get("statistics", {}),
            "charts_json": state.get("charts_json", []),
            "cleaned_data": state.get("cleaned_data", ""),
        }
    )
    events = _emit({**state, "events": events}, "agent_done", "InsightNarratorAgent", "Analysis narrative ready")

    # Build final output: narrative + chart event
    charts = state.get("charts_json", [])
    if charts:
        events = _emit(
            {**state, "events": events},
            "charts_json",
            "ChartGeneratorAgent",
            json.dumps(charts)
        )

    return {
        "insights": output.content,
        "analysis_result": output.content,
        "events": events,
    }


# ─── Build the Analytics Subgraph ────────────────────────────────────────────

def build_analytics_graph() -> StateGraph:
    builder = StateGraph(AnalyticsDeptState)

    builder.add_node("data_resolver_node", data_resolver_node)
    builder.add_node("profiler_node", profiler_node)
    builder.add_node("cleaner_node", cleaner_node)
    builder.add_node("stats_node", stats_node)
    builder.add_node("chart_node", chart_node)
    builder.add_node("narrator_node", narrator_node)

    # Linear pipeline with parallel stats+charts fan-out
    builder.add_edge(START, "data_resolver_node")
    builder.add_edge("data_resolver_node", "profiler_node")
    builder.add_edge("profiler_node", "cleaner_node")

    # Parallel fan-out: stats and charts are independent
    builder.add_edge("cleaner_node", "stats_node")
    builder.add_edge("cleaner_node", "chart_node")

    # Fan-in to narrator
    builder.add_edge("stats_node", "narrator_node")
    builder.add_edge("chart_node", "narrator_node")

    builder.add_edge("narrator_node", END)

    return builder.compile()


analytics_subgraph = build_analytics_graph()


# ─── Outer node — plugs into root orchestrator graph ─────────────────────────

async def analytics_department_node(state) -> Dict[str, Any]:
    """
    Outer node that plugs into the main orchestrator graph.
    Extracts the analytics task from the CEO's plan, runs the subgraph,
    and writes results back into OrchestratorState.
    """
    import asyncio
    from shared.audit import log_event

    subtasks = state.get("task_plan", {}).get("subtasks", [])
    analytics_task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "analytics":
            analytics_task = st.get("task", state["user_request"])
            break

    user_id = state.get("user_id")
    events = list(state.get("agent_events", []))
    events.append({
        "event": "department_started",
        "department": "analytics",
        "agent": "Analytics Head",
        "data": f"Starting analytics: {analytics_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    asyncio.create_task(log_event(
        user_id=user_id, event_type="department_started",
        department="analytics", agent="Analytics Head",
        data={"task": analytics_task[:100]}
    ))

    initial_state = {
        "task": analytics_task,
        "data_source": "",  # resolver will check RAG for uploaded files
        "user_id": user_id,
        "api_keys": state.get("api_keys"),
        "selected_model": state.get("selected_model"),
        "events": [],
    }

    final_state = await analytics_subgraph.ainvoke(initial_state)

    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done",
        "department": "analytics",
        "agent": "Analytics Head",
        "data": "Analytics department completed",
        "timestamp": datetime.utcnow().isoformat(),
    })

    asyncio.create_task(log_event(
        user_id=user_id, event_type="department_done",
        department="analytics", agent="Analytics Head", data={}
    ))

    department_outputs = dict(state.get("department_outputs", {}))
    department_outputs["analytics"] = final_state.get("analysis_result", "Analytics completed.")

    completed = list(state.get("completed_departments", []))
    if "analytics" not in completed:
        completed.append("analytics")

    return {
        "department_outputs": department_outputs,
        "completed_departments": completed,
        "agent_events": events,
    }
