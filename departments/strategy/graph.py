"""
Business Strategy Department LangGraph subgraph.

Flow:
  START → router_node → [market_node || competitor_node] (parallel)
        → financial_model_node → swot_node
        → [business_plan_node || pitch_deck_node] (parallel if both needed)
        → synthesizer_node → END

Key hive-mind patterns demonstrated:
1. MarketAnalystAgent calls Research dept directly (inter-dept call, no CEO)
2. FinancialModelerAgent uses shared Code sandbox (execute_code)
3. SWOT receives inputs from 3 upstream agents (true fan-in)
"""
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

from langgraph.graph import StateGraph, START, END

from departments.strategy.state import StrategyDeptState
from departments.strategy.agents import (
    StrategyRouterAgent,
    MarketAnalystAgent,
    CompetitorAnalystAgent,
    FinancialModelerAgent,
    SWOTAgent,
    BusinessPlanAgent,
    PitchDeckAgent,
    StrategySynthesizerAgent,
)


# ─── Agent factory ───────────────────────────────────────────────────────────

def _make_agents(api_keys=None, selected_model=None):
    return {
        "router": StrategyRouterAgent(api_keys=api_keys, selected_model=selected_model),
        "market": MarketAnalystAgent(api_keys=api_keys, selected_model=selected_model),
        "competitor": CompetitorAnalystAgent(api_keys=api_keys, selected_model=selected_model),
        "financial": FinancialModelerAgent(api_keys=api_keys, selected_model=selected_model),
        "swot": SWOTAgent(api_keys=api_keys, selected_model=selected_model),
        "business_plan": BusinessPlanAgent(api_keys=api_keys, selected_model=selected_model),
        "pitch_deck": PitchDeckAgent(api_keys=api_keys, selected_model=selected_model),
        "synthesizer": StrategySynthesizerAgent(api_keys=api_keys, selected_model=selected_model),
    }


def _emit(state: StrategyDeptState, event: str, agent: str, data: str = "") -> list:
    events = list(state.get("events", []))
    events.append({
        "event": event,
        "department": "strategy",
        "agent": agent,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return events


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

async def router_node(state: StrategyDeptState) -> Dict[str, Any]:
    """StrategyRouterAgent — decides which agents to run."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "StrategyRouterAgent", "Planning strategy approach...")
    output = await agents["router"].execute(state["task"])
    required = output.metadata.get("required_agents", ["market", "competitor", "swot", "business_plan"])
    events = _emit({**state, "events": events}, "agent_done", "StrategyRouterAgent",
                   f"Deploying: {', '.join(required)}")
    return {"required_agents": required, "events": events}


async def market_node(state: StrategyDeptState) -> Dict[str, Any]:
    """MarketAnalystAgent — calls Research dept directly (hive-mind inter-dept pattern)."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "MarketAnalystAgent",
                   "Calling Research dept for market intelligence...")
    output = await agents["market"].execute(
        state["task"],
        context={"api_keys": state.get("api_keys"), "selected_model": state.get("selected_model"),
                 "user_id": state.get("user_id")}
    )
    events = _emit({**state, "events": events}, "agent_done", "MarketAnalystAgent", "Market analysis complete")
    return {"market_research": output.content, "events": events}


async def competitor_node(state: StrategyDeptState) -> Dict[str, Any]:
    """CompetitorAnalystAgent — competitive landscape mapping."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "CompetitorAnalystAgent", "Mapping competitive landscape...")
    output = await agents["competitor"].execute(state["task"])
    events = _emit({**state, "events": events}, "agent_done", "CompetitorAnalystAgent", "Competitor analysis done")
    return {"competitor_data": output.content, "events": events}


async def financial_model_node(state: StrategyDeptState) -> Dict[str, Any]:
    """FinancialModelerAgent — 3-year projections using shared Code sandbox."""
    if "financial_model" not in state.get("required_agents", []):
        return {"financial_model": "", "events": state.get("events", [])}

    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "FinancialModelerAgent",
                   "Building financial model (shared Code sandbox)...")
    output = await agents["financial"].execute(
        state["task"],
        context={"market_research": state.get("market_research", "")}
    )
    events = _emit({**state, "events": events}, "agent_done", "FinancialModelerAgent",
                   "Financial model complete")
    return {"financial_model": output.content, "events": events}


async def swot_node(state: StrategyDeptState) -> Dict[str, Any]:
    """SWOTAgent — structured SWOT synthesizing all upstream inputs."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "SWOTAgent", "Conducting SWOT analysis...")
    output = await agents["swot"].execute(
        state["task"],
        context={
            "market_research": state.get("market_research", ""),
            "competitor_data": state.get("competitor_data", ""),
            "financial_model": state.get("financial_model", ""),
        }
    )
    events = _emit({**state, "events": events}, "agent_done", "SWOTAgent", "SWOT complete")
    return {
        "swot_analysis": output.metadata.get("swot", {}),
        "events": events
    }


async def business_plan_node(state: StrategyDeptState) -> Dict[str, Any]:
    """BusinessPlanAgent — full executive business plan."""
    if "business_plan" not in state.get("required_agents", []):
        return {"business_plan": "", "events": state.get("events", [])}

    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "BusinessPlanAgent", "Writing business plan...")
    output = await agents["business_plan"].execute(
        state["task"],
        context={
            "market_research": state.get("market_research", ""),
            "competitor_data": state.get("competitor_data", ""),
            "financial_model": state.get("financial_model", ""),
            "swot_analysis": state.get("swot_analysis", {}),
        }
    )
    events = _emit({**state, "events": events}, "agent_done", "BusinessPlanAgent", "Business plan ready")
    return {"business_plan": output.content, "events": events}


async def pitch_deck_node(state: StrategyDeptState) -> Dict[str, Any]:
    """PitchDeckAgent — investor pitch deck (only if requested)."""
    if "pitch_deck" not in state.get("required_agents", []):
        return {"pitch_deck": "", "events": state.get("events", [])}

    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "PitchDeckAgent", "Building investor pitch deck...")
    output = await agents["pitch_deck"].execute(
        state["task"],
        context={
            "market_research": state.get("market_research", ""),
            "financial_model": state.get("financial_model", ""),
            "competitor_data": state.get("competitor_data", ""),
        }
    )
    events = _emit({**state, "events": events}, "agent_done", "PitchDeckAgent", "Pitch deck ready")
    return {"pitch_deck": output.content, "events": events}


async def synthesizer_node(state: StrategyDeptState) -> Dict[str, Any]:
    """StrategySynthesizerAgent — final consolidated strategy document."""
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "StrategySynthesizerAgent", "Synthesizing strategy package...")
    output = await agents["synthesizer"].execute(
        state["task"],
        context={
            "market_research": state.get("market_research", ""),
            "competitor_data": state.get("competitor_data", ""),
            "financial_model": state.get("financial_model", ""),
            "swot_analysis": state.get("swot_analysis", {}),
            "business_plan": state.get("business_plan", ""),
            "pitch_deck": state.get("pitch_deck", ""),
        }
    )
    events = _emit({**state, "events": events}, "agent_done", "StrategySynthesizerAgent",
                   "Strategy report complete")
    return {"final_strategy": output.content, "events": events}


# ─── Build the Strategy Subgraph ─────────────────────────────────────────────

def build_strategy_graph() -> StateGraph:
    builder = StateGraph(StrategyDeptState)

    builder.add_node("router_node", router_node)
    builder.add_node("market_node", market_node)
    builder.add_node("competitor_node", competitor_node)
    builder.add_node("financial_model_node", financial_model_node)
    builder.add_node("swot_node", swot_node)
    builder.add_node("business_plan_node", business_plan_node)
    builder.add_node("pitch_deck_node", pitch_deck_node)
    builder.add_node("synthesizer_node", synthesizer_node)

    builder.add_edge(START, "router_node")

    # Parallel fan-out: market + competitor are independent
    builder.add_edge("router_node", "market_node")
    builder.add_edge("router_node", "competitor_node")

    # Fan-in: both feed into financial model, then SWOT
    builder.add_edge("market_node", "financial_model_node")
    builder.add_edge("competitor_node", "financial_model_node")
    builder.add_edge("financial_model_node", "swot_node")

    # Parallel fan-out: business plan + pitch deck are independent
    builder.add_edge("swot_node", "business_plan_node")
    builder.add_edge("swot_node", "pitch_deck_node")

    # Fan-in to synthesizer
    builder.add_edge("business_plan_node", "synthesizer_node")
    builder.add_edge("pitch_deck_node", "synthesizer_node")

    builder.add_edge("synthesizer_node", END)

    return builder.compile()


strategy_subgraph = build_strategy_graph()


# ─── Outer node — plugs into root orchestrator graph ─────────────────────────

async def strategy_department_node(state) -> Dict[str, Any]:
    """
    Outer node that plugs into the main orchestrator graph.
    Demonstrates the hive-mind pattern: this dept calls Research dept
    internally via MarketAnalystAgent → inter_dept.call_research_dept().
    """
    import asyncio
    from shared.audit import log_event

    subtasks = state.get("task_plan", {}).get("subtasks", [])
    strategy_task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "strategy":
            strategy_task = st.get("task", state["user_request"])
            break

    user_id = state.get("user_id")
    events = list(state.get("agent_events", []))
    events.append({
        "event": "department_started",
        "department": "strategy",
        "agent": "Strategy Head",
        "data": f"Starting strategy work: {strategy_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    asyncio.create_task(log_event(
        user_id=user_id, event_type="department_started",
        department="strategy", agent="Strategy Head",
        data={"task": strategy_task[:100]}
    ))

    initial_state = {
        "task": strategy_task,
        "required_agents": [],
        "user_id": user_id,
        "api_keys": state.get("api_keys"),
        "selected_model": state.get("selected_model"),
        "market_research": "",
        "competitor_data": "",
        "financial_model": "",
        "swot_analysis": {},
        "business_plan": "",
        "pitch_deck": "",
        "final_strategy": "",
        "events": [],
    }

    final_state = await strategy_subgraph.ainvoke(initial_state)

    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done",
        "department": "strategy",
        "agent": "Strategy Head",
        "data": "Strategy department completed",
        "timestamp": datetime.utcnow().isoformat(),
    })

    asyncio.create_task(log_event(
        user_id=user_id, event_type="department_done",
        department="strategy", agent="Strategy Head", data={}
    ))

    department_outputs = dict(state.get("department_outputs", {}))
    department_outputs["strategy"] = final_state.get("final_strategy", "Strategy analysis completed.")

    completed = list(state.get("completed_departments", []))
    if "strategy" not in completed:
        completed.append("strategy")

    return {
        "department_outputs": department_outputs,
        "completed_departments": completed,
        "agent_events": events,
    }
