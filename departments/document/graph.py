"""
Document Department LangGraph subgraph.
Flow: document_qa_node → [done]
"""
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from departments.document.state import DocumentDeptState
from departments.document.agents import DocumentQAAgent


def _emit(state: DocumentDeptState, event: str, agent: str, data: str = "") -> list:
    events = list(state.get("events", []))
    events.append({
        "event": event,
        "department": "document",
        "agent": agent,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return events


def document_qa_node(state: DocumentDeptState) -> Dict[str, Any]:
    """Single node that retrieves documents and answers directly."""
    qa_agent = DocumentQAAgent(
        api_keys=state.get("api_keys"), 
        selected_model=state.get("selected_model")
    )

    events = _emit(state, "agent_working", "DocumentQAAgent", "Retrieving documents and extracting answer...")
    output = qa_agent.execute(state["task"], context={"user_id": state.get("user_id")})

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "DocumentQAAgent",
        f"Document analysis complete. Confidence: {output.metadata.get('confidence', 0.0):.2f}",
    )

    return {
        "final_answer": output.content,
        "events": events,
    }


# ─── Build the Document Subgraph ──────────────────────────────────────────────

document_graph = StateGraph(DocumentDeptState)

document_graph.add_node("document_qa_node", document_qa_node)

document_graph.add_edge(START, "document_qa_node")
document_graph.add_edge("document_qa_node", END)

document_subgraph = document_graph.compile()


# ─── Outer node — plugs into root orchestrator graph ─────────────────────────

def document_department_node(state: "OrchestratorState") -> Dict[str, Any]:
    """
    Outer node that plugs into the main orchestrator graph.
    """
    # Find the document subtask from CEO plan
    subtasks = state.get("task_plan", {}).get("subtasks", [])
    doc_task = state["user_request"]  # fallback
    for st in subtasks:
        if st.get("department") == "document":
            doc_task = st.get("task", state["user_request"])
            break

    # Emit department started event
    events = list(state.get("agent_events", []))
    events.append({
        "event": "department_started",
        "department": "document",
        "agent": "Document Analyst",
        "data": f"Starting document analysis: {doc_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    initial_state = {
        "task": doc_task,
        "original_request": state["user_request"],
        "user_id": state.get("user_id"),
        "api_keys": state.get("api_keys"),
        "selected_model": state.get("selected_model"),
        "events": [],
    }

    final_state = document_subgraph.invoke(initial_state)

    # Merge subgraph events into orchestrator events
    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done",
        "department": "document",
        "agent": "Document Analyst",
        "data": "Document analysis completed successfully",
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Update department outputs and completed list
    department_outputs = dict(state.get("department_outputs", {}))
    department_outputs["document"] = final_state.get("final_answer", "Document analysis completed.")

    completed = list(state.get("completed_departments", []))
    if "document" not in completed:
        completed.append("document")

    return {
        "department_outputs": department_outputs,
        "completed_departments": completed,
        "agent_events": events,
    }


# Resolve forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from orchestrator.state import OrchestratorState
