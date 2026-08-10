"""
Content Department LangGraph subgraph.
Flow: copywriter_node → seo_optimizer_node → editor_node → [done]
"""
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from departments.content.state import ContentDeptState
from departments.content.agents import CopywriterAgent, SEOOptimizerAgent, EditorAgent


# ─── Agent factory ───────────────────────────────────────────────────────────

def _make_agents(api_keys=None, selected_model=None):
    """Create fresh agent instances with the given API keys and model."""
    return (
        CopywriterAgent(api_keys=api_keys, selected_model=selected_model),
        SEOOptimizerAgent(api_keys=api_keys, selected_model=selected_model),
        EditorAgent(api_keys=api_keys, selected_model=selected_model),
    )


def _emit(state: ContentDeptState, event: str, agent: str, data: str = "") -> list:
    events = list(state.get("events", []))
    events.append({
        "event": event,
        "department": "content",
        "agent": agent,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return events


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

async def copywriter_node(state: ContentDeptState) -> Dict[str, Any]:
    """Copywriter Agent — creates the initial content draft."""
    copywriter, _, _ = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "CopywriterAgent", "Writing content draft...")

    output = await copywriter.execute(
        state["task"],
        context={"research_context": state.get("research_context", "")},
    )

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "CopywriterAgent",
        f"Draft ready ({output.metadata.get('word_count', 0)} words)",
    )

    return {
        "draft_content": output.content,
        "events": events,
    }


async def seo_optimizer_node(state: ContentDeptState) -> Dict[str, Any]:
    """SEO Optimizer Agent — keyword optimization."""
    _, seo_optimizer, _ = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "SEOOptimizerAgent", "Optimizing for search engines...")

    output = await seo_optimizer.execute(
        state["task"],
        context={"draft_content": state.get("draft_content", "")},
    )

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "SEOOptimizerAgent",
        f"SEO score: {output.metadata.get('seo_score', 'N/A')}",
    )

    return {
        "seo_keywords": output.metadata.get("secondary_keywords", []),
        "meta_description": output.metadata.get("meta_description", ""),
        "seo_optimized_content": output.content,
        "events": events,
    }


async def editor_node(state: ContentDeptState) -> Dict[str, Any]:
    """Editor Agent — final polish and proofreading."""
    _, _, editor = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "EditorAgent", "Editing and proofreading...")

    output = await editor.execute(
        state["task"],
        context={
            "seo_content": state.get("seo_optimized_content", ""),
            "draft_content": state.get("draft_content", ""),
        },
    )

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "EditorAgent",
        f"Quality score: {output.metadata.get('quality_score', 'N/A')}",
    )

    return {
        "edited_content": output.content,
        "final_content": output.content,
        "events": events,
    }


# ─── Build the Content Subgraph ───────────────────────────────────────────────

content_graph = StateGraph(ContentDeptState)

content_graph.add_node("copywriter_node", copywriter_node)
content_graph.add_node("seo_optimizer_node", seo_optimizer_node)
content_graph.add_node("editor_node", editor_node)

content_graph.add_edge(START, "copywriter_node")
content_graph.add_edge("copywriter_node", "seo_optimizer_node")
content_graph.add_edge("seo_optimizer_node", "editor_node")
content_graph.add_edge("editor_node", END)

content_subgraph = content_graph.compile()


# ─── Outer node — plugs into root orchestrator graph ─────────────────────────

async def content_department_node(state) -> Dict[str, Any]:
    """Outer node that plugs into the main orchestrator graph."""
    # Find the content subtask
    subtasks = state.get("task_plan", {}).get("subtasks", [])
    content_task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "content":
            content_task = st.get("task", state["user_request"])
            break

    # Get research output if available (for sequential tasks)
    research_context = state.get("department_outputs", {}).get("research", "")

    events = list(state.get("agent_events", []))
    events.append({
        "event": "department_started",
        "department": "content",
        "agent": "Content Head",
        "data": f"Starting content creation: {content_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    initial_state = {
        "task": content_task,
        "original_request": state["user_request"],
        "research_context": research_context,
        "api_keys": state.get("api_keys"),
        "selected_model": state.get("selected_model"),
        "events": [],
    }

    final_state = await content_subgraph.ainvoke(initial_state)

    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done",
        "department": "content",
        "agent": "Content Head",
        "data": "Content department completed successfully",
        "timestamp": datetime.utcnow().isoformat(),
    })

    department_outputs = dict(state.get("department_outputs", {}))
    department_outputs["content"] = final_state.get("final_content", "Content generation completed.")

    completed = list(state.get("completed_departments", []))
    if "content" not in completed:
        completed.append("content")

    return {
        "department_outputs": department_outputs,
        "completed_departments": completed,
        "agent_events": events,
    }
