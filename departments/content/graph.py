"""
Content Department LangGraph subgraph.

Flow:
  START → content_router_node → [blog/social/seo_copy/full_pipeline]
  - blog:         copywriter_node → seo_optimizer_node → editor_node → END
  - social:       copywriter_node → END (fast path, no SEO/editing needed)
  - seo_copy:     copywriter_node → seo_optimizer_node → END
  - full_pipeline: copywriter_node → seo_optimizer_node → editor_node → END

This implements the recursive Router pattern: CEO routes to Content dept,
Content dept routes internally to its own specialist pipeline.
"""
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from departments.content.state import ContentDeptState
from departments.content.agents import ContentRouterAgent, CopywriterAgent, SEOOptimizerAgent, EditorAgent


# ─── Agent factory ───────────────────────────────────────────────────────────

def _make_agents(api_keys=None, selected_model=None):
    """Create fresh agent instances with the given API keys and model."""
    return {
        "router": ContentRouterAgent(api_keys=api_keys, selected_model=selected_model),
        "copywriter": CopywriterAgent(api_keys=api_keys, selected_model=selected_model),
        "seo": SEOOptimizerAgent(api_keys=api_keys, selected_model=selected_model),
        "editor": EditorAgent(api_keys=api_keys, selected_model=selected_model),
    }


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


# ─── Content Router Node ──────────────────────────────────────────────────────

async def content_router_node(state: ContentDeptState) -> Dict[str, Any]:
    """ContentRouterAgent — decides which pipeline to run (blog/social/seo_copy/full_pipeline)."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "ContentRouterAgent", "Classifying content type...")
    output = await agents["router"].execute(state["task"])
    content_type = output.metadata.get("content_type", "full_pipeline")
    tone = output.metadata.get("tone", "professional")
    events = _emit({**state, "events": events}, "agent_done", "ContentRouterAgent",
                   f"Routing to: {content_type} pipeline ({tone} tone)")
    return {"content_type": content_type, "preferred_tone": tone, "events": events}


def route_by_content_type(state: ContentDeptState) -> str:
    """Conditional edge: route to the appropriate pipeline start."""
    content_type = state.get("content_type", "full_pipeline")
    if content_type == "social":
        return "copywriter_node_social"  # Skip SEO + Editor
    return "copywriter_node"  # blog, seo_copy, full_pipeline all start with copywriter


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

async def copywriter_node(state: ContentDeptState) -> Dict[str, Any]:
    """Copywriter Agent — creates the initial content draft."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    tone_hint = state.get("preferred_tone", "professional")

    events = _emit(state, "agent_working", "CopywriterAgent", "Writing content draft...")

    output = await agents["copywriter"].execute(
        state["task"],
        context={"research_context": state.get("research_context", ""), "tone": tone_hint},
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


async def copywriter_node_social(state: ContentDeptState) -> Dict[str, Any]:
    """Fast-path Copywriter for social media — no SEO or editing needed."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "CopywriterAgent", "Writing social post (fast path)...")
    output = await agents["copywriter"].execute(
        state["task"],
        context={"research_context": state.get("research_context", ""), "tone": "casual/engaging"},
    )
    events = _emit({**state, "events": events}, "agent_done", "CopywriterAgent", "Social post ready")
    return {"draft_content": output.content, "final_content": output.content, "events": events}


async def seo_optimizer_node(state: ContentDeptState) -> Dict[str, Any]:
    """SEO Optimizer Agent — keyword optimization."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "SEOOptimizerAgent", "Optimizing for search engines...")

    output = await agents["seo"].execute(
        state["task"],
        context={"draft_content": state.get("draft_content", "")},
    )

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "SEOOptimizerAgent",
        f"SEO score: {output.metadata.get('seo_score', 'N/A')}",
    )

    # For seo_copy pipeline: make this the final content (no editor)
    is_seo_only = state.get("content_type") == "seo_copy"
    return {
        "seo_keywords": output.metadata.get("secondary_keywords", []),
        "meta_description": output.metadata.get("meta_description", ""),
        "seo_optimized_content": output.content,
        **(({"final_content": output.content}) if is_seo_only else {}),
        "events": events,
    }


async def editor_node(state: ContentDeptState) -> Dict[str, Any]:
    """Editor Agent — final polish and proofreading."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "EditorAgent", "Editing and proofreading...")

    output = await agents["editor"].execute(
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


def route_after_seo(state: ContentDeptState) -> str:
    """After SEO: skip Editor for seo_copy pipeline, run Editor for blog/full_pipeline."""
    if state.get("content_type") == "seo_copy":
        return END
    return "editor_node"


# ─── Build the Content Subgraph ────────────────────────────────────────────────────

content_graph = StateGraph(ContentDeptState)

content_graph.add_node("content_router_node", content_router_node)
content_graph.add_node("copywriter_node", copywriter_node)
content_graph.add_node("copywriter_node_social", copywriter_node_social)
content_graph.add_node("seo_optimizer_node", seo_optimizer_node)
content_graph.add_node("editor_node", editor_node)

# START → Router → (conditional) pipeline entry point
content_graph.add_edge(START, "content_router_node")
content_graph.add_conditional_edges(
    "content_router_node",
    route_by_content_type,
    {"copywriter_node": "copywriter_node", "copywriter_node_social": "copywriter_node_social"}
)

# Social fast-path: Copywriter → END
content_graph.add_edge("copywriter_node_social", END)

# Full pipeline: Copywriter → SEO → (conditional) Editor or END
content_graph.add_edge("copywriter_node", "seo_optimizer_node")
content_graph.add_conditional_edges(
    "seo_optimizer_node",
    route_after_seo,
    {"editor_node": "editor_node", END: END}
)
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
