"""
Aggregator node — combines outputs from all departments into a final unified response.
"""
from datetime import datetime
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.state import OrchestratorState
from shared.base_agent import normalize_llm_content


AGGREGATOR_PROMPT = """You are a knowledgeable AI assistant synthesizing research from multiple specialized agents.

Your job is to combine their findings into a single, clear, helpful response — like a smart friend explaining something to you.

Guidelines:
- Be direct and conversational. No corporate jargon or unnecessary formality.
- Use Markdown for structure: headings, bullet points, bold for key terms, code blocks for code.
- DO NOT start with "# 📋 Final Report" or "Executive Summary" headers — just answer the question naturally.
- Lead with the most important insight. Then add supporting details.
- If there are sources or links, include them inline or in a "Sources" section at the end.
- Keep it scannable: use short paragraphs and bullets where appropriate.
- Match the length to the complexity — simple questions get short answers, deep questions get detailed ones.
"""


async def aggregator_node(state: OrchestratorState) -> Dict[str, Any]:
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
        final = normalize_llm_content(output)
        if not final:
            final = f"I wasn't able to find specific information from the {dept} department. Please try rephrasing your question."

        # Artificially stream the output to the frontend for the ChatGPT typing effect
        task_id = state.get("task_id")
        if task_id:
            try:
                from api.websocket.stream import manager
                import asyncio
                
                chunk_size = 120  # characters per chunk for high throughput and fluid rendering
                accumulated = ""
                for i in range(0, len(final), chunk_size):
                    accumulated += final[i:i+chunk_size]
                    await manager.broadcast(task_id, {
                        "event": "partial_output",
                        "data": accumulated,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    await asyncio.sleep(0.005)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to artificially stream final output: {e}")

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
        cleaned_dept_out = normalize_llm_content(output)
        combined_context += f"## {dept.upper()} DEPARTMENT OUTPUT:\n{cleaned_dept_out}\n\n---\n\n"

    try:
        from api.websocket.stream import manager
        
        llm = ceo_llm(state.get("api_keys"), selected_model=state.get("selected_model"))
        
        accumulated = ""
        task_id = state.get("task_id")
        
        async for chunk in llm.astream([
            SystemMessage(content=AGGREGATOR_PROMPT),
            HumanMessage(content=combined_context),
        ]):
            content = normalize_llm_content(chunk.content)
            if content:
                accumulated += content
                if task_id:
                    await manager.broadcast(task_id, {
                        "event": "partial_output",
                        "data": accumulated,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
        
        final = normalize_llm_content(accumulated)
    except Exception as e:
        # Fallback: just concatenate departments cleanly
        final = f"**Based on your question:** {state['user_request']}\n\n"
        for dept, output in department_outputs.items():
            final += f"### {dept.capitalize()} Findings\n\n{normalize_llm_content(output)}\n\n"

    events.append({
        "event": "final_output",
        "data": final,
        "department": None,
        "agent": "CEO",
        "timestamp": datetime.utcnow().isoformat(),
    })

    return {"final_output": final, "agent_events": events}
