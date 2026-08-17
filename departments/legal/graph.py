"""Legal Department LangGraph subgraph."""
import asyncio
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END
from departments.legal.state import LegalDeptState
from departments.legal.agents import (
    LegalRouterAgent, ContractReviewAgent,
    ToSDrafterAgent, ComplianceChecklistAgent, LegalSynthesizerAgent
)

def _make_agents(api_keys=None, selected_model=None):
    return {
        "router": LegalRouterAgent(api_keys=api_keys, selected_model=selected_model),
        "contract": ContractReviewAgent(api_keys=api_keys, selected_model=selected_model),
        "tos": ToSDrafterAgent(api_keys=api_keys, selected_model=selected_model),
        "compliance": ComplianceChecklistAgent(api_keys=api_keys, selected_model=selected_model),
        "synthesizer": LegalSynthesizerAgent(api_keys=api_keys, selected_model=selected_model),
    }

def _emit(state, event, agent, data=""):
    events = list(state.get("events", []))
    events.append({"event": event, "department": "legal", "agent": agent,
                   "data": data, "timestamp": datetime.utcnow().isoformat()})
    return events

async def router_node(state: LegalDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "LegalRouterAgent", "Classifying legal request...")
    output = await agents["router"].execute(state["task"])
    required = output.metadata.get("required_agents", ["contract_review"])
    events = _emit({**state,"events":events}, "agent_done", "LegalRouterAgent", f"Agents: {required}")
    return {"required_agents": required, "events": events}

async def contract_review_node(state: LegalDeptState) -> Dict[str, Any]:
    if "contract_review" not in state.get("required_agents", []):
        return {"contract_review": "", "events": state.get("events", [])}
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "ContractReviewAgent", "Reviewing contract...")
    output = await agents["contract"].execute(state["task"], context={"user_id": state.get("user_id")})
    events = _emit({**state,"events":events}, "agent_done", "ContractReviewAgent", "Review complete")
    return {"contract_review": output.content, "events": events}

async def tos_draft_node(state: LegalDeptState) -> Dict[str, Any]:
    if "tos_draft" not in state.get("required_agents", []):
        return {"tos_draft": "", "events": state.get("events", [])}
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "ToSDrafterAgent", "Drafting legal document...")
    output = await agents["tos"].execute(state["task"])
    events = _emit({**state,"events":events}, "agent_done", "ToSDrafterAgent", "Draft complete")
    return {"tos_draft": output.content, "events": events}

async def compliance_node(state: LegalDeptState) -> Dict[str, Any]:
    if "compliance_checklist" not in state.get("required_agents", []):
        return {"compliance_checklist": "", "events": state.get("events", [])}
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "ComplianceChecklistAgent", "Building compliance checklist...")
    output = await agents["compliance"].execute(state["task"], context={"api_keys": state.get("api_keys")})
    events = _emit({**state,"events":events}, "agent_done", "ComplianceChecklistAgent", "Checklist ready")
    return {"compliance_checklist": output.content, "events": events}

async def synthesizer_node(state: LegalDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    events = _emit(state, "agent_working", "LegalSynthesizerAgent", "Synthesizing legal output...")
    output = await agents["synthesizer"].execute(state["task"], context={
        "contract_review": state.get("contract_review",""),
        "tos_draft": state.get("tos_draft",""),
        "compliance_checklist": state.get("compliance_checklist",""),
    })
    events = _emit({**state,"events":events}, "agent_done", "LegalSynthesizerAgent", "Legal report ready")
    return {"final_legal_output": output.content, "events": events}

def build_legal_graph():
    builder = StateGraph(LegalDeptState)
    builder.add_node("router_node", router_node)
    builder.add_node("contract_review_node", contract_review_node)
    builder.add_node("tos_draft_node", tos_draft_node)
    builder.add_node("compliance_node", compliance_node)
    builder.add_node("synthesizer_node", synthesizer_node)
    builder.add_edge(START, "router_node")
    builder.add_edge("router_node", "contract_review_node")
    builder.add_edge("router_node", "tos_draft_node")
    builder.add_edge("router_node", "compliance_node")
    builder.add_edge("contract_review_node", "synthesizer_node")
    builder.add_edge("tos_draft_node", "synthesizer_node")
    builder.add_edge("compliance_node", "synthesizer_node")
    builder.add_edge("synthesizer_node", END)
    return builder.compile()

legal_subgraph = build_legal_graph()

async def legal_department_node(state) -> Dict[str, Any]:
    """Outer node plugging into root orchestrator."""
    subtasks = state.get("task_plan", {}).get("subtasks", [])
    task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "legal":
            task = st.get("task", task); break
    events = list(state.get("agent_events", []))
    events.append({"event":"department_started","department":"legal","agent":"Legal Head",
                   "data":f"Legal: {task[:80]}","timestamp":datetime.utcnow().isoformat()})
    final = await legal_subgraph.ainvoke({
        "task": task, "required_agents": [], "user_id": state.get("user_id"),
        "api_keys": state.get("api_keys"), "selected_model": state.get("selected_model"),
        "contract_review":"","tos_draft":"","compliance_checklist":"","final_legal_output":"","events":[]
    })
    events.extend(final.get("events",[]))
    events.append({"event":"department_done","department":"legal","agent":"Legal Head",
                   "data":"Legal complete","timestamp":datetime.utcnow().isoformat()})
    dept_outputs = dict(state.get("department_outputs",{}))
    dept_outputs["legal"] = final.get("final_legal_output","Legal analysis complete.")
    completed = list(state.get("completed_departments",[]))
    if "legal" not in completed: completed.append("legal")
    return {"department_outputs": dept_outputs, "completed_departments": completed, "agent_events": events}
