"""
Financial Department LangGraph subgraph.
Flow: router_node → [parallel specialized nodes] → synthesizer_node → [done]
"""
from datetime import datetime
import json
from typing import Any, Dict, List

from langgraph.graph import StateGraph, START, END

from departments.financial.state import FinancialDeptState
from departments.financial.agents import (
    FinancialRouterAgent,
    MarketDataAgent,
    FundamentalAnalysisAgent,
    TechnicalAnalysisAgent,
    NewsSentimentAgent,
    PortfolioAnalystAgent,
    ComparativeAnalysisAgent,
    SynthesizerAgent,
)


# ─── Agent factory ───────────────────────────────────────────────────────────

def _make_agents(api_keys=None, selected_model=None):
    return {
        "router": FinancialRouterAgent(api_keys=api_keys, selected_model=selected_model),
        "market": MarketDataAgent(api_keys=api_keys, selected_model=selected_model),
        "fundamental": FundamentalAnalysisAgent(api_keys=api_keys, selected_model=selected_model),
        "technical": TechnicalAnalysisAgent(api_keys=api_keys, selected_model=selected_model),
        "news": NewsSentimentAgent(api_keys=api_keys, selected_model=selected_model),
        "portfolio": PortfolioAnalystAgent(api_keys=api_keys, selected_model=selected_model),
        "comparative": ComparativeAnalysisAgent(api_keys=api_keys, selected_model=selected_model),
        "synthesizer": SynthesizerAgent(api_keys=api_keys, selected_model=selected_model),
    }


def _emit(state: FinancialDeptState, event: str, agent: str, data: str = "") -> list:
    events = list(state.get("events", []))
    events.append({
        "event": event,
        "department": "financial",
        "agent": agent,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return events


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

async def router_node(state: FinancialDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    router = agents["router"]

    events = _emit(state, "agent_working", "FinancialRouterAgent", "Analyzing query for financial intent...")
    output = await router.execute(state["task"])
    
    required_agents = output.metadata.get("required_agents", ["market_data"])
    tickers = output.metadata.get("tickers", [])
    
    events = _emit(
        {**state, "events": events},
        "agent_done",
        "FinancialRouterAgent",
        f"Routing to: {required_agents} for tickers: {tickers}"
    )

    return {
        "required_agents": required_agents,
        "tickers": tickers,
        "events": events,
    }


async def market_node(state: FinancialDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "MarketDataAgent", "Fetching market data...")
    output = await agents["market"].execute(state["task"], context={"tickers": state.get("tickers", [])})
    events = _emit({**state, "events": events}, "agent_done", "MarketDataAgent", "Market data fetched")
    return {"market_data": output.metadata.get("data", {}), "events": events}


async def fundamental_node(state: FinancialDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "FundamentalAnalysisAgent", "Analyzing fundamentals...")
    output = await agents["fundamental"].execute(state["task"], context={"tickers": state.get("tickers", [])})
    events = _emit({**state, "events": events}, "agent_done", "FundamentalAnalysisAgent", "Fundamentals analyzed")
    return {"fundamental_data": output.metadata.get("data", {}), "events": events}


async def technical_node(state: FinancialDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "TechnicalAnalysisAgent", "Computing technical indicators...")
    output = await agents["technical"].execute(state["task"], context={"tickers": state.get("tickers", [])})
    events = _emit({**state, "events": events}, "agent_done", "TechnicalAnalysisAgent", "Technical indicators computed")
    return {"technical_data": output.metadata.get("data", {}), "events": events}


async def news_node(state: FinancialDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "NewsSentimentAgent", "Analyzing news sentiment...")
    output = await agents["news"].execute(state["task"], context={"tickers": state.get("tickers", [])})
    events = _emit({**state, "events": events}, "agent_done", "NewsSentimentAgent", "Sentiment analyzed")
    return {"news_sentiment": output.metadata, "events": events}


async def portfolio_node(state: FinancialDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "PortfolioAnalystAgent", "Analyzing portfolio...")
    output = await agents["portfolio"].execute(state["task"])
    events = _emit({**state, "events": events}, "agent_done", "PortfolioAnalystAgent", "Portfolio analyzed")
    return {"portfolio_analysis": output.metadata.get("data", {}), "events": events}


async def comparative_node(state: FinancialDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "ComparativeAnalysisAgent", "Comparing tickers...")
    output = await agents["comparative"].execute(
        state["task"], 
        context={
            "market_data": state.get("market_data", {}), 
            "fundamental_data": state.get("fundamental_data", {})
        }
    )
    events = _emit({**state, "events": events}, "agent_done", "ComparativeAnalysisAgent", "Comparison complete")
    return {"comparison_data": output.metadata.get("comparison_data", {}), "events": events}


async def synthesizer_node(state: FinancialDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "SynthesizerAgent", "Synthesizing investment report...")
    output = await agents["synthesizer"].execute(state["task"], context=state)
    events = _emit({**state, "events": events}, "agent_done", "SynthesizerAgent", "Report generated")
    
    
    charts = output.metadata.get("charts_json", [])
    if charts:
        events = _emit({**state, "events": events}, "charts_json", "SynthesizerAgent", json.dumps(charts))

    return {
        "final_report": output.content,
        "charts_json": charts,
        "events": events,
    }


# ─── Conditional Routing ──────────────────────────────────────────────────────

# ─── Conditional Routing ──────────────────────────────────────────────────────

def route_financial_agents(state: FinancialDeptState) -> List[str]:
    """Determine which worker nodes to run from router."""
    required = state.get("required_agents", [])
    nodes = []
    if "market_data" in required:
        nodes.append("market_node")
    if "fundamental" in required:
        nodes.append("fundamental_node")
    if "technical" in required:
        nodes.append("technical_node")
    if "news" in required:
        nodes.append("news_node")
    if "portfolio" in required:
        nodes.append("portfolio_node")
    if "comparative" in required:
        if "market_node" not in nodes:
            nodes.append("market_node")
        if "fundamental_node" not in nodes:
            nodes.append("fundamental_node")

    if not nodes:
        nodes = ["market_node"]
    return nodes


def route_after_market_fundamental(state: FinancialDeptState) -> str:
    """If comparative analysis was requested, route through comparative_node."""
    if "comparative" in state.get("required_agents", []):
        return "comparative_node"
    return "synthesizer_node"


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_financial_graph() -> StateGraph:
    builder = StateGraph(FinancialDeptState)

    # Add nodes
    builder.add_node("router_node", router_node)
    builder.add_node("market_node", market_node)
    builder.add_node("fundamental_node", fundamental_node)
    builder.add_node("technical_node", technical_node)
    builder.add_node("news_node", news_node)
    builder.add_node("portfolio_node", portfolio_node)
    builder.add_node("comparative_node", comparative_node)
    builder.add_node("synthesizer_node", synthesizer_node)

    # Edge from START to router
    builder.add_edge(START, "router_node")

    # Fan-out conditional edges from router
    builder.add_conditional_edges(
        "router_node",
        route_financial_agents,
        ["market_node", "fundamental_node", "technical_node", "news_node", "portfolio_node"]
    )

    # Fan-in: Market & Fundamental conditionally go to comparative or synthesizer
    builder.add_conditional_edges(
        "market_node",
        route_after_market_fundamental,
        {"comparative_node": "comparative_node", "synthesizer_node": "synthesizer_node"}
    )
    builder.add_conditional_edges(
        "fundamental_node",
        route_after_market_fundamental,
        {"comparative_node": "comparative_node", "synthesizer_node": "synthesizer_node"}
    )

    builder.add_edge("technical_node", "synthesizer_node")
    builder.add_edge("news_node", "synthesizer_node")
    builder.add_edge("portfolio_node", "synthesizer_node")
    builder.add_edge("comparative_node", "synthesizer_node")

    # Final edge
    builder.add_edge("synthesizer_node", END)

    return builder.compile()

financial_subgraph = build_financial_graph()


# ─── Outer node — plugs into root orchestrator graph ─────────────────────────

async def financial_department_node(state) -> Dict[str, Any]:
    """Outer node that plugs into the main orchestrator graph."""
    subtasks = state.get("task_plan", {}).get("subtasks", [])
    financial_task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "financial":
            financial_task = st.get("task", state["user_request"])
            break

    events = list(state.get("agent_events", []))
    events.append({
        "event": "department_started",
        "department": "financial",
        "agent": "Financial Head",
        "data": f"Starting financial task: {financial_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    initial_state = {
        "task": financial_task,
        "original_request": state["user_request"],
        "user_id": state.get("user_id"),
        "api_keys": state.get("api_keys"),
        "selected_model": state.get("selected_model"),
        "events": [],
    }

    final_state = await financial_subgraph.ainvoke(initial_state)

    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done",
        "department": "financial",
        "agent": "Financial Head",
        "data": "Financial department completed successfully",
        "timestamp": datetime.utcnow().isoformat(),
    })


    department_outputs = dict(state.get("department_outputs", {}))
    department_outputs["financial"] = final_state.get("final_report", "Financial analysis completed.")

    completed = list(state.get("completed_departments", []))
    if "financial" not in completed:
        completed.append("financial")

    return {
        "department_outputs": department_outputs,
        "completed_departments": completed,
        "agent_events": events,
    }
