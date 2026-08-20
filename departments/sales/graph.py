"""Sales Department LangGraph subgraph."""
from datetime import datetime
from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from departments.sales.state import SalesDeptState
from departments.sales.agents import (SalesRouterAgent, LeadResearchAgent,
    ColdEmailAgent, FollowUpSequencerAgent, SalesSynthesizerAgent)

def _make(api_keys=None, selected_model=None):
    return {"router": SalesRouterAgent(api_keys=api_keys, selected_model=selected_model),
            "lead": LeadResearchAgent(api_keys=api_keys, selected_model=selected_model),
            "email": ColdEmailAgent(api_keys=api_keys, selected_model=selected_model),
            "followup": FollowUpSequencerAgent(api_keys=api_keys, selected_model=selected_model),
            "synth": SalesSynthesizerAgent(api_keys=api_keys, selected_model=selected_model)}

def _emit(state, event, agent, data=""):
    evs = list(state.get("events",[]))
    evs.append({"event":event,"department":"sales","agent":agent,"data":data,"timestamp":datetime.utcnow().isoformat()})
    return evs

async def router_node(state):
    a = _make(state.get("api_keys"), state.get("selected_model"))
    evs = _emit(state,"agent_working","SalesRouterAgent","Planning sales strategy...")
    o = await a["router"].execute(state["task"])
    evs = _emit({**state,"events":evs},"agent_done","SalesRouterAgent",f"Agents: {o.metadata.get('required_agents')}")
    return {"required_agents": o.metadata.get("required_agents",["cold_email"]), "events": evs}

async def lead_research_node(state):
    if "lead_research" not in state.get("required_agents",[]): return {"lead_research":"","events":state.get("events",[])}
    a = _make(state.get("api_keys"), state.get("selected_model"))
    evs = _emit(state,"agent_working","LeadResearchAgent","Calling Research dept for prospect intel...")
    o = await a["lead"].execute(state["task"], context={"api_keys":state.get("api_keys"),"selected_model":state.get("selected_model")})
    evs = _emit({**state,"events":evs},"agent_done","LeadResearchAgent","Prospect research done")
    return {"lead_research": o.content, "events": evs}

async def cold_email_node(state):
    if "cold_email" not in state.get("required_agents",[]): return {"cold_emails":"","events":state.get("events",[])}
    a = _make(state.get("api_keys"), state.get("selected_model"))
    evs = _emit(state,"agent_working","ColdEmailAgent","Writing cold emails...")
    o = await a["email"].execute(state["task"], context={"lead_research":state.get("lead_research","")})
    evs = _emit({**state,"events":evs},"agent_done","ColdEmailAgent","Emails ready")
    return {"cold_emails": o.content, "events": evs}

async def followup_node(state):
    if "follow_up_sequence" not in state.get("required_agents",[]): return {"follow_up_sequence":"","events":state.get("events",[])}
    a = _make(state.get("api_keys"), state.get("selected_model"))
    evs = _emit(state,"agent_working","FollowUpSequencerAgent","Building follow-up sequence...")
    o = await a["followup"].execute(state["task"], context={"cold_emails":state.get("cold_emails","")})
    evs = _emit({**state,"events":evs},"agent_done","FollowUpSequencerAgent","Sequence ready")
    return {"follow_up_sequence": o.content, "events": evs}

async def synth_node(state):
    a = _make(state.get("api_keys"), state.get("selected_model"))
    evs = _emit(state,"agent_working","SalesSynthesizerAgent","Packaging outreach kit...")
    o = await a["synth"].execute(state["task"], context={
        "lead_research":state.get("lead_research",""), "cold_emails":state.get("cold_emails",""),
        "follow_up_sequence":state.get("follow_up_sequence","")})
    evs = _emit({**state,"events":evs},"agent_done","SalesSynthesizerAgent","Outreach kit ready")
    return {"final_outreach_kit": o.content, "events": evs}

def build_sales_graph():
    b = StateGraph(SalesDeptState)
    for n,f in [("router_node",router_node),("lead_research_node",lead_research_node),
                ("cold_email_node",cold_email_node),("followup_node",followup_node),("synth_node",synth_node)]:
        b.add_node(n, f)
    b.add_edge(START,"router_node")
    b.add_edge("router_node","lead_research_node")
    b.add_edge("lead_research_node","cold_email_node")
    b.add_edge("cold_email_node","followup_node")
    b.add_edge("followup_node","synth_node")
    b.add_edge("synth_node",END)
    return b.compile()

sales_subgraph = build_sales_graph()

async def sales_department_node(state) -> Dict[str, Any]:
    subtasks = state.get("task_plan",{}).get("subtasks",[])
    task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "sales": task = st.get("task", task); break
    evs = list(state.get("agent_events",[]))
    evs.append({"event":"department_started","department":"sales","agent":"Sales Head",
                "data":f"Sales: {task[:80]}","timestamp":datetime.utcnow().isoformat()})
    final = await sales_subgraph.ainvoke({
        "task":task,"required_agents":[],"user_id":state.get("user_id"),
        "api_keys":state.get("api_keys"),"selected_model":state.get("selected_model"),
        "lead_research":"","cold_emails":"","follow_up_sequence":"","final_outreach_kit":"","events":[]})
    evs.extend(final.get("events",[]))
    evs.append({"event":"department_done","department":"sales","agent":"Sales Head",
                "data":"Sales complete","timestamp":datetime.utcnow().isoformat()})
    dept_out = dict(state.get("department_outputs",{}))
    dept_out["sales"] = final.get("final_outreach_kit","Sales outreach kit complete.")
    completed = list(state.get("completed_departments",[]))
    if "sales" not in completed: completed.append("sales")
    return {"department_outputs":dept_out,"completed_departments":completed,"agent_events":evs}
