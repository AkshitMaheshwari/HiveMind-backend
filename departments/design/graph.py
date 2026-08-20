"""Design Department LangGraph subgraph."""
from datetime import datetime
from typing import Any, Dict, List
from langgraph.graph import StateGraph, START, END
from departments.design.state import DesignDeptState
from departments.design.agents import (DesignRouterAgent, BrandingAgent,
    LogoConceptAgent, PitchVisualsAgent, DesignSynthesizerAgent)

def _make(api_keys=None, selected_model=None):
    return {"router": DesignRouterAgent(api_keys=api_keys, selected_model=selected_model),
            "branding": BrandingAgent(api_keys=api_keys, selected_model=selected_model),
            "logo": LogoConceptAgent(api_keys=api_keys, selected_model=selected_model),
            "pitch": PitchVisualsAgent(api_keys=api_keys, selected_model=selected_model),
            "synth": DesignSynthesizerAgent(api_keys=api_keys, selected_model=selected_model)}

def _emit(state, event, agent, data=""):
    evs = list(state.get("events",[])); evs.append({"event":event,"department":"design","agent":agent,"data":data,"timestamp":datetime.utcnow().isoformat()}); return evs

async def router_node(state):
    a = _make(state.get("api_keys"),state.get("selected_model"))
    evs = _emit(state,"agent_working","DesignRouterAgent","Planning design work...")
    o = await a["router"].execute(state["task"])
    evs = _emit({**state,"events":evs},"agent_done","DesignRouterAgent",f"Agents: {o.metadata.get('required_agents')}")
    return {"required_agents":o.metadata.get("required_agents",["branding","logo_concept"]),"events":evs}

async def branding_node(state):
    if "branding" not in state.get("required_agents",[]): return {"branding_guide":"","events":state.get("events",[])}
    a = _make(state.get("api_keys"),state.get("selected_model"))
    evs = _emit(state,"agent_working","BrandingAgent","Creating brand identity...")
    o = await a["branding"].execute(state["task"])
    evs = _emit({**state,"events":evs},"agent_done","BrandingAgent","Brand guide ready")
    return {"branding_guide":o.content,"events":evs}

async def logo_node(state):
    if "logo_concept" not in state.get("required_agents",[]): return {"logo_concepts":[],"events":state.get("events",[])}
    a = _make(state.get("api_keys"),state.get("selected_model"))
    evs = _emit(state,"agent_working","LogoConceptAgent","Generating logo concepts + DALL-E 3 image...")
    o = await a["logo"].execute(state["task"], context={"branding_guide":state.get("branding_guide","")})
    evs = _emit({**state,"events":evs},"agent_done","LogoConceptAgent","Logo concepts ready")
    urls = o.metadata.get("image_urls",[])
    if urls:
        evs = _emit({**state,"events":evs},"image_generated","LogoConceptAgent",urls[0])
    return {"logo_concepts":[o.content],"visual_assets":urls,"events":evs}

async def pitch_visuals_node(state):
    if "pitch_visuals" not in state.get("required_agents",[]): return {"pitch_visuals":"","events":state.get("events",[])}
    a = _make(state.get("api_keys"),state.get("selected_model"))
    evs = _emit(state,"agent_working","PitchVisualsAgent","Creating visual directions...")
    o = await a["pitch"].execute(state["task"], context={"branding_guide":state.get("branding_guide","")})
    evs = _emit({**state,"events":evs},"agent_done","PitchVisualsAgent","Pitch visuals ready")
    return {"pitch_visuals":o.content,"events":evs}

async def synth_node(state):
    a = _make(state.get("api_keys"),state.get("selected_model"))
    evs = _emit(state,"agent_working","DesignSynthesizerAgent","Packaging design assets...")
    o = await a["synth"].execute(state["task"], context={
        "branding_guide":state.get("branding_guide",""),
        "logo_concepts":state.get("logo_concepts",[]),
        "visual_assets":state.get("visual_assets",[]),
        "pitch_visuals":state.get("pitch_visuals","")})
    evs = _emit({**state,"events":evs},"agent_done","DesignSynthesizerAgent","Design package ready")
    return {"final_design_package":o.content,"events":evs}

def build_design_graph():
    b = StateGraph(DesignDeptState)
    for n,f in [("router_node",router_node),("branding_node",branding_node),
                ("logo_node",logo_node),("pitch_visuals_node",pitch_visuals_node),("synth_node",synth_node)]:
        b.add_node(n,f)
    b.add_edge(START,"router_node")
    b.add_edge("router_node","branding_node")
    b.add_edge("branding_node","logo_node")
    b.add_edge("branding_node","pitch_visuals_node")
    b.add_edge("logo_node","synth_node")
    b.add_edge("pitch_visuals_node","synth_node")
    b.add_edge("synth_node",END)
    return b.compile()

design_subgraph = build_design_graph()

async def design_department_node(state) -> Dict[str, Any]:
    subtasks = state.get("task_plan",{}).get("subtasks",[])
    task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "design": task = st.get("task",task); break
    evs = list(state.get("agent_events",[]))
    evs.append({"event":"department_started","department":"design","agent":"Design Head",
                "data":f"Design: {task[:80]}","timestamp":datetime.utcnow().isoformat()})
    final = await design_subgraph.ainvoke({
        "task":task,"required_agents":[],"user_id":state.get("user_id"),
        "api_keys":state.get("api_keys"),"selected_model":state.get("selected_model"),
        "branding_guide":"","logo_concepts":[],"visual_assets":[],"pitch_visuals":"","final_design_package":"","events":[]})
    evs.extend(final.get("events",[]))
    evs.append({"event":"department_done","department":"design","agent":"Design Head",
                "data":"Design complete","timestamp":datetime.utcnow().isoformat()})
    dept_out = dict(state.get("department_outputs",{}))
    dept_out["design"] = final.get("final_design_package","Design package complete.")
    completed = list(state.get("completed_departments",[]))
    if "design" not in completed: completed.append("design")
    return {"department_outputs":dept_out,"completed_departments":completed,"agent_events":evs}
