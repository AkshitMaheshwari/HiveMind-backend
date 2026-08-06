"""
Aggregator node — combines outputs from all departments into a final unified response.
"""
from datetime import datetime
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.state import OrchestratorState


AGGREGATOR_PROMPT = """You are the CEO of an AI company finalizing a client deliverable.

You have received outputs from your departments. Your job is to:
1. Synthesize all department outputs into ONE unified, polished final response
2. Maintain proper structure with clear sections for each department's contribution
3. Add an Executive Summary at the top
4. Ensure the tone is professional and the output is directly useful to the user

Format in clean Markdown. Start with "# 📋 Final Report" then the Executive Summary.
"""


def aggregator_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Combines all department_outputs into a single cohesive final response.
    """
    from shared.llm import ceo_llm
    from orchestrator.ceo_agent import emit_event

    department_outputs = state.get("department_outputs", {})
    events = list(state.get("agent_events", []))

    events.append({
        "event": "aggregating",
        "data": "CEO is synthesizing all department outputs...",
        "department": None,
        "agent": "CEO",
        "timestamp": datetime.utcnow().isoformat(),
    })

    # If only one department ran, return its output directly
    if len(department_outputs) == 1:
        dept, output = list(department_outputs.items())[0]
        output_str = str(output).strip() if output else ""
        if not output_str:
            output_str = f"No specific report content was returned by the {dept} department."
        final = output_str if output_str.startswith("#") else f"# 📋 Output\n\n{output_str}"
        events.append({
            "event": "final_output",
            "data": final,
            "department": None,
            "agent": "CEO",
            "timestamp": datetime.utcnow().isoformat(),
        })
        return {"final_output": final, "agent_events": events}

    # Multiple departments — synthesize
    combined_context = f"User Request: {state['user_request']}\n\n"
    for dept, output in department_outputs.items():
        combined_context += f"## {dept.upper()} DEPARTMENT OUTPUT:\n{output}\n\n---\n\n"

    try:
        llm = ceo_llm(state.get("api_keys"))
        response = llm.invoke([
            SystemMessage(content=AGGREGATOR_PROMPT),
            HumanMessage(content=combined_context),
        ])
        final = response.content
    except Exception as e:
        # Fallback: just concatenate
        final = f"# 📋 Final Report\n\n**User Request:** {state['user_request']}\n\n"
        for dept, output in department_outputs.items():
            final += f"## {dept.upper()} Department\n\n{output}\n\n---\n\n"

    events.append({
        "event": "final_output",
        "data": final,
        "department": None,
        "agent": "CEO",
        "timestamp": datetime.utcnow().isoformat(),
    })

    return {"final_output": final, "agent_events": events}
