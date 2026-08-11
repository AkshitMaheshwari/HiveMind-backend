"""
Data Analyst Department LangGraph subgraph.
Flow: data_planner_node → eda_node → execution_node → insights_node → dashboard_node → [done]
"""
import asyncio
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from departments.data_analyst.state import DataAnalystDeptState
from departments.data_analyst.agents import DataPlannerAgent, EDAAgent, InsightsAgent, ReviewAgent, DashboardAgent
from shared.tools import execute_code
import json


def _make_agents(api_keys=None, selected_model=None):
    return {
        "planner": DataPlannerAgent(api_keys=api_keys, selected_model=selected_model),
        "eda": EDAAgent(api_keys=api_keys, selected_model=selected_model),
        "insights": InsightsAgent(api_keys=api_keys, selected_model=selected_model),
        "review": ReviewAgent(api_keys=api_keys, selected_model=selected_model),
        "dashboard": DashboardAgent(api_keys=api_keys, selected_model=selected_model),
    }


def _emit(state: DataAnalystDeptState, event: str, agent: str, data: str = "") -> list:
    events = list(state.get("events", []))
    events.append({
        "event": event,
        "department": "data_analyst",
        "agent": agent,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return events


async def data_planner_node(state: DataAnalystDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    planner = agents["planner"]

    events = _emit(state, "agent_working", "DataPlannerAgent", "Formulating analysis and visualization plan...")
    output = await planner.execute(state["task"], context={"user_id": state.get("user_id")})

    events = _emit({**state, "events": events}, "agent_done", "DataPlannerAgent", "Analysis plan ready.")
    
    return {
        "analysis_plan": output.content,
        "events": events,
    }


async def eda_node(state: DataAnalystDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    eda = agents["eda"]

    events = _emit(state, "agent_working", "EDAAgent", "Writing data processing scripts...")
    
    context = {
        "analysis_plan": state.get("analysis_plan"),
        "user_id": state.get("user_id")
    }
    output = await eda.execute(state["task"], context=context)
    
    events = _emit({**state, "events": events}, "agent_done", "EDAAgent", "EDA script generated.")
    
    return {
        "eda_code": output.content,
        "events": events,
    }


async def execution_node(state: DataAnalystDeptState) -> Dict[str, Any]:
    events = _emit(state, "agent_working", "DataExecutionAgent", "Executing analysis in sandbox...")
    
    code = state.get("eda_code", "")
    exec_result = await asyncio.to_thread(execute_code, code)
    
    stdout = exec_result["stdout"]
    
    dataset_info = {}
    eda_results = {}
    try:
        # Extract the JSON block printed by the script
        import re
        json_match = re.search(r'(\{.*\})', stdout.replace('\n', ' '), re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
            dataset_info = parsed.get("dataset_info", {})
            eda_results = parsed.get("eda_results", {})
        else:
            # Fallback if the whole stdout is JSON
            parsed = json.loads(stdout)
            dataset_info = parsed.get("dataset_info", {})
            eda_results = parsed.get("eda_results", {})
    except Exception as e:
        pass  # Will be handled gracefully down the line

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "DataExecutionAgent",
        "Execution: " + ("✅ Success" if exec_result["success"] else f"❌ Error: {exec_result['stderr'][:100]}"),
    )
    
    return {
        "execution_stdout": exec_result["stdout"],
        "execution_stderr": exec_result["stderr"],
        "execution_success": exec_result["success"],
        "dataset_info": dataset_info,
        "eda_results": eda_results,
        "events": events,
    }


async def insights_node(state: DataAnalystDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    insights = agents["insights"]

    events = _emit(state, "agent_working", "InsightsAgent", "Extracting insights from data...")
    
    context = {"execution_stdout": state.get("execution_stdout", "")}
    output = await insights.execute(state["task"], context=context)
    
    events = _emit({**state, "events": events}, "agent_done", "InsightsAgent", "Insights generated.")
    
    return {
        "insights": output.metadata if output.success else {},
        "events": events,
    }


async def review_node(state: DataAnalystDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    review = agents["review"]

    events = _emit(state, "agent_working", "ReviewAgent", "Reviewing generated insights and EDA...")
    
    context = {
        "execution_stdout": state.get("execution_stdout", ""),
        "insights": state.get("insights", {})
    }
    output = await review.execute(state["task"], context=context)
    
    events = _emit({**state, "events": events}, "agent_done", "ReviewAgent", "Peer review completed.")
    
    return {
        "review": output.metadata if output.success else {},
        "events": events,
    }


async def dashboard_node(state: DataAnalystDeptState) -> Dict[str, Any]:
    agents = _make_agents(state.get("api_keys"), state.get("selected_model"))
    dashboard = agents["dashboard"]

    events = _emit(state, "agent_working", "DashboardAgent", "Building interactive HTML report...")
    
    context = {
        "analysis_plan": state.get("analysis_plan", ""),
        "insights": state.get("insights", ""),
        "execution_stdout": state.get("execution_stdout", ""),
        "user_id": state.get("user_id", "")
    }
    output = await dashboard.execute(state["task"], context=context)
    
    events = _emit({**state, "events": events}, "agent_done", "DashboardAgent", "Interactive report ready.")
    
    # Construct the massive structured payload exactly like the friend's agent
    structured_output = {
        "analysis_plan": state.get("analysis_plan", {}),
        "dataset_info": state.get("dataset_info", {}),
        "eda_results": state.get("eda_results", {}),
        "insights": state.get("insights", {}),
        "review": state.get("review", {}),
        "report": state.get("insights", {}).get("report", ""),
        "visualizations": [],
    }
    
    json_payload = json.dumps(structured_output, indent=2)

    dashboard_code = output.content
    final_report = f"""## 📊 Data Analyst Interactive Report

### 💡 Key Insights
{state.get("insights", {}).get("executive_summary", "")}

### 📈 Interactive Live Report
```html
{dashboard_code}
```

> [!TIP]
> **Live Preview:** Click the **👁️ Live Preview** tab at the top of this message (or **↗️ Fullscreen**) to view and interact with your report directly in the browser!

### 🧱 Raw Structured Output
```json
{json_payload}
```
"""
    
    charts = output.metadata.get("charts_json", [])
    if charts:
        events = _emit({**state, "events": events}, "charts_json", "DashboardAgent", json.dumps(charts))

    return {
        "dashboard_code": dashboard_code,
        "structured_output": structured_output,
        "final_report": final_report,
        "charts_json": charts,
        "events": events,
    }


# ─── Build Graph ─────────────────────────────────────────────────────────────

data_analyst_graph = StateGraph(DataAnalystDeptState)

data_analyst_graph.add_node("data_planner_node", data_planner_node)
data_analyst_graph.add_node("eda_node", eda_node)
data_analyst_graph.add_node("execution_node", execution_node)
data_analyst_graph.add_node("insights_node", insights_node)
data_analyst_graph.add_node("review_node", review_node)
data_analyst_graph.add_node("dashboard_node", dashboard_node)

data_analyst_graph.add_edge(START, "data_planner_node")
data_analyst_graph.add_edge("data_planner_node", "eda_node")
data_analyst_graph.add_edge("eda_node", "execution_node")
data_analyst_graph.add_edge("execution_node", "insights_node")
data_analyst_graph.add_edge("insights_node", "review_node")
data_analyst_graph.add_edge("review_node", "dashboard_node")
data_analyst_graph.add_edge("dashboard_node", END)

data_analyst_subgraph = data_analyst_graph.compile()


# ─── Outer node ─────────────────────────────────────────────────────────────

async def data_analyst_department_node(state) -> Dict[str, Any]:
    """Outer node that plugs into the main orchestrator graph."""
    subtasks = state.get("task_plan", {}).get("subtasks", [])
    da_task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "data_analyst":
            da_task = st.get("task", state["user_request"])
            break

    events = list(state.get("agent_events", []))
    events.append({
        "event": "department_started",
        "department": "data_analyst",
        "agent": "Data Analyst Head",
        "data": f"Starting data analysis: {da_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    initial_state = {
        "task": da_task,
        "original_request": state["user_request"],
        "user_id": state.get("user_id"),
        "api_keys": state.get("api_keys"),
        "selected_model": state.get("selected_model"),
        "events": [],
    }

    final_state = await data_analyst_subgraph.ainvoke(initial_state)

    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done",
        "department": "data_analyst",
        "agent": "Data Analyst Head",
        "data": "Data analysis completed successfully",
        "timestamp": datetime.utcnow().isoformat(),
    })

    department_outputs = dict(state.get("department_outputs", {}))
    department_outputs["data_analyst"] = final_state.get("final_report", "Data analysis completed.")

    completed = list(state.get("completed_departments", []))
    if "data_analyst" not in completed:
        completed.append("data_analyst")

    return {
        "department_outputs": department_outputs,
        "completed_departments": completed,
        "agent_events": events,
    }
