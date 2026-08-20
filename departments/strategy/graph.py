"""
Business Strategy Department LangGraph subgraph.

Optimized High-Performance Parallel Architecture:
  START → router_node 
        → parallel_analysis_node (runs Market, Competitor, Financial Modeler, and SWOT concurrently)
        → synthesizer_node (generates comprehensive strategy report & 9-slide pitch deck)
        → END

Performance note: _make_agents() is called ONCE in the outer strategy_department_node
and agents are injected into inner nodes via closures — same pattern as research/graph.py.
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
    StrategySynthesizerAgent,
)


# ─── Agent factory ────────────────────────────────────────────────────────────
# Called ONCE per request from the outer department node.

def _make_agents(api_keys=None, selected_model=None) -> Dict[str, Any]:
    """Build all strategy agents once. Returns a named dict."""
    return {
        "router":      StrategyRouterAgent(api_keys=api_keys, selected_model=selected_model),
        "market":      MarketAnalystAgent(api_keys=api_keys, selected_model=selected_model),
        "competitor":  CompetitorAnalystAgent(api_keys=api_keys, selected_model=selected_model),
        "financial":   FinancialModelerAgent(api_keys=api_keys, selected_model=selected_model),
        "swot":        SWOTAgent(api_keys=api_keys, selected_model=selected_model),
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


# ─── Closure-based node builders ──────────────────────────────────────────────

def _make_router_node(agents: Dict[str, Any]):
    async def router_node(state: StrategyDeptState) -> Dict[str, Any]:
        """StrategyRouterAgent — determines strategy plan."""
        events = _emit(state, "agent_working", "StrategyRouterAgent", "Planning strategic deployment...")
        output = await agents["router"].execute(state["task"])
        required = output.metadata.get("required_agents", [
            "market", "competitor", "financial_model", "swot", "business_plan", "pitch_deck"
        ])
        events = _emit({**state, "events": events}, "agent_done", "StrategyRouterAgent",
                       f"Deployed concurrent specialists: {', '.join(required)}")
        return {"required_agents": required, "events": events}
    return router_node


def _make_parallel_analysis_node(agents: Dict[str, Any]):
    async def parallel_analysis_node(state: StrategyDeptState) -> Dict[str, Any]:
        """
        Runs all 4 specialist analytical agents concurrently in parallel via asyncio.gather:
        - MarketAnalystAgent (TAM/SAM/SOM + market trends)
        - CompetitorAnalystAgent (Competitors & whitespace)
        - FinancialModelerAgent (Python code sandbox calculation)
        - SWOTAgent (Strengths/Weaknesses/Opportunities/Threats)
        """
        events = list(state.get("events", []))
        events.append({
            "event": "agent_working", "department": "strategy",
            "agent": "Strategy Specialist Team",
            "data": "Running concurrent market, competitor, financial modeling & SWOT analysis...",
            "timestamp": datetime.utcnow().isoformat()
        })

        market_out, comp_out, fin_out, swot_out = await asyncio.gather(
            agents["market"].execute(
                state["task"],
                context={"api_keys": state.get("api_keys"), "selected_model": state.get("selected_model"),
                         "user_id": state.get("user_id")}
            ),
            agents["competitor"].execute(state["task"]),
            agents["financial"].execute(state["task"]),
            agents["swot"].execute(state["task"]),
            return_exceptions=True
        )

        market_res = market_out.content if hasattr(market_out, "content") else ""
        comp_res   = comp_out.content   if hasattr(comp_out, "content")   else ""
        fin_res    = fin_out.content    if hasattr(fin_out, "content")    else ""
        swot_dict  = swot_out.metadata.get("swot", {}) if hasattr(swot_out, "metadata") else {}

        events.append({
            "event": "agent_done", "department": "strategy",
            "agent": "Strategy Specialist Team",
            "data": "Concurrent analysis complete (Market, Competitors, Financial Model, SWOT ready)",
            "timestamp": datetime.utcnow().isoformat()
        })
        return {
            "market_research": market_res,
            "competitor_data": comp_res,
            "financial_model": fin_res,
            "swot_analysis": swot_dict,
            "events": events,
        }
    return parallel_analysis_node


def _make_synthesizer_node(agents: Dict[str, Any]):
    async def synthesizer_node(state: StrategyDeptState) -> Dict[str, Any]:
        """StrategySynthesizerAgent — master unified strategy blueprint & pitch deck synthesis."""
        events = _emit(state, "agent_working", "StrategySynthesizerAgent",
                       "Synthesizing executive master strategy & pitch deck...")
        output = await agents["synthesizer"].execute(
            state["task"],
            context={
                "market_research": state.get("market_research", ""),
                "competitor_data": state.get("competitor_data", ""),
                "financial_model": state.get("financial_model", ""),
                "swot_analysis":   state.get("swot_analysis", {}),
                "business_plan":   state.get("business_plan", ""),
                "pitch_deck":      state.get("pitch_deck", ""),
            }
        )
        events = _emit({**state, "events": events}, "agent_done", "StrategySynthesizerAgent",
                       "Master strategy package complete")
        return {"final_strategy": output.content, "events": events}
    return synthesizer_node


# ─── Build the Strategy Subgraph ─────────────────────────────────────────────

def build_strategy_graph(agents: Dict[str, Any]) -> StateGraph:
    """
    Build a strategy subgraph wired with pre-instantiated agents.
    Agents are injected via closures so nodes never call _make_agents() themselves.
    """
    builder = StateGraph(StrategyDeptState)
    builder.add_node("router_node",            _make_router_node(agents))
    builder.add_node("parallel_analysis_node", _make_parallel_analysis_node(agents))
    builder.add_node("synthesizer_node",       _make_synthesizer_node(agents))

    builder.add_edge(START, "router_node")
    builder.add_edge("router_node", "parallel_analysis_node")
    builder.add_edge("parallel_analysis_node", "synthesizer_node")
    builder.add_edge("synthesizer_node", END)
    return builder.compile()


# ─── Outer node — plugs into root orchestrator graph ─────────────────────────

async def strategy_department_node(state) -> Dict[str, Any]:
    """
    Outer node that plugs into the main orchestrator graph.
    Builds all agents ONCE here, constructs the subgraph with those agents
    injected via closures, then runs it.
    """
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
        "event": "department_started", "department": "strategy",
        "agent": "Strategy Head",
        "data": f"Starting strategy analysis: {strategy_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    asyncio.create_task(log_event(
        user_id=user_id, event_type="department_started",
        department="strategy", agent="Strategy Head",
        data={"task": strategy_task[:100]}
    ))

    # ── Build agents ONCE — LLM cache in shared/llm.py deduplicates HTTP clients ──
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    subgraph = build_strategy_graph(agents)

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

    final_state = await subgraph.ainvoke(initial_state)

    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done", "department": "strategy",
        "agent": "Strategy Head", "data": "Strategy department completed",
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

