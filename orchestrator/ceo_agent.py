"""
CEO Agent — the top-level orchestrator of the AI Company.

Responsibilities:
1. Classify user intent → which departments are needed
2. Decompose multi-part tasks into subtasks with dependencies
3. Handle clarification if request is ambiguous
4. Route to the correct department subgraphs
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from orchestrator.state import OrchestratorState


# ─── Pydantic schema for structured CEO output ────────────────────────────────

class SubTask(BaseModel):
    department: Literal["research", "content", "code", "document"]
    task: str = Field(description="Specific task for this department")
    depends_on: Optional[str] = Field(
        None, description="Name of department this task depends on, or null"
    )


class TaskPlanOutput(BaseModel):
    departments: List[Literal["research", "content", "code", "document"]] = Field(
        default_factory=list,
        description="List of departments needed, in execution order. Empty list if clarification_needed is True."
    )
    sequence: Optional[Literal["sequential", "parallel"]] = Field(
        default="sequential",
        description="'sequential' if tasks depend on each other, 'parallel' if independent. Always set to 'sequential' or 'parallel', never null."
    )
    subtasks: List[SubTask] = Field(default_factory=list, description="Specific sub-tasks for each department")
    reasoning: str = Field(default="", description="Brief explanation of why these departments were chosen")
    clarification_needed: bool = Field(
        default=False, description="True if the request is too ambiguous to route"
    )
    clarification_question: Optional[str] = Field(
        None, description="Question to ask user if clarification_needed is True"
    )


CEO_SYSTEM_PROMPT = """You are the CEO of an AI company with three departments:

🔬 RESEARCH DEPARTMENT
- Web research, fact-finding, market analysis, summarization.
- Use when: user wants information, analysis, or research on any topic on the web.

📄 DOCUMENT DEPARTMENT
- Direct Q&A, data extraction, and search within the user's uploaded documents (PDFs, knowledge base).
- Use when: user asks questions about their uploaded files, PDFs, or private data.

✍️  CONTENT DEPARTMENT  
- Writing, blog posts, SEO, copywriting, editing, social media content
- Use when: user wants content created, written, or optimized

💻 CODE DEPARTMENT
- Code generation, debugging, documentation, technical problem solving
- Use when: user wants code written, debugged, or explained

Your job is to analyze the user's request and output a structured JSON task plan.

CRITICAL RULES (ALWAYS follow these):
- The "sequence" field MUST always be either "sequential" or "parallel" — NEVER null or empty.
- If clarification_needed is True, still set sequence="sequential" and departments=[] and subtasks=[].
- Pick ONLY the departments actually needed (1-3)
- If research is needed BEFORE content (e.g. "research X then write about it"), sequence="sequential" and content's depends_on="research"
- If tasks are independent, sequence="parallel"
- If the request is too vague (single word like "hello", "hi", "help me"), set clarification_needed=True, sequence="sequential", departments=[], subtasks=[]
- NEVER ask for clarification if the user asks about an uploaded document, PDF, or knowledge base. Route these to the Research department immediately so it can search the documents.
- Your reasoning should be 1-2 sentences max

Examples:
- "Write a blog post about AI trends" → departments=["content"], sequence="parallel"
- "Research competitors and write a comparison" → departments=["research", "content"], sequence="sequential"
- "Fix this Python bug" → departments=["code"], sequence="parallel"
- "hello" → clarification_needed=True, sequence="sequential", departments=[]
- "what is in my uploaded PDF" → departments=["document"], sequence="sequential"
"""


def emit_event(state: OrchestratorState, event: str, data: str = None, **kwargs) -> List[Dict]:
    """Helper to add a streaming event to the state."""
    events = list(state.get("agent_events", []))
    events.append({
        "event": event,
        "data": data,
        "department": kwargs.get("department"),
        "agent": kwargs.get("agent"),
        "timestamp": datetime.utcnow().isoformat(),
    })
    return events


async def ceo_router_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    CEO node: analyzes request, produces task plan, routes to departments.
    """
    from shared.llm import ceo_llm

    llm = ceo_llm(state.get("api_keys"), selected_model=state.get("selected_model"))
    structured_llm = llm.with_structured_output(TaskPlanOutput)

    events = emit_event(state, "ceo_planning", "CEO is analyzing your request...")

    try:
        # Build message list: prior turns + current request
        from langchain_core.messages import AIMessage
        messages = [SystemMessage(content=CEO_SYSTEM_PROMPT)]

        for turn in state.get("chat_history", []):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=f"User request: {state['user_request']}"))

        plan: TaskPlanOutput = await structured_llm.ainvoke(messages)

        task_plan = {
            "departments": plan.departments,
            "sequence": plan.sequence,
            "subtasks": [st.model_dump() for st in plan.subtasks],
            "reasoning": plan.reasoning,
        }

        events = emit_event(
            {**state, "agent_events": events},
            "ceo_plan_ready",
            data=f"Routing to: {', '.join(plan.departments)}",
        )

        return {
            "task_plan": task_plan,
            "active_departments": plan.departments,
            "completed_departments": [],
            "department_outputs": {},
            "agent_events": events,
            "clarification_needed": plan.clarification_needed,
            "clarification_question": plan.clarification_question,
        }

    except Exception as e:
        events = emit_event(
            {**state, "agent_events": events},
            "error",
            data=f"CEO routing failed: {str(e)}",
        )
        return {
            "task_plan": {"departments": ["research"], "sequence": "sequential", "subtasks": [], "reasoning": "fallback"},
            "active_departments": ["research"],
            "completed_departments": [],
            "department_outputs": {},
            "agent_events": events,
            "clarification_needed": False,
            "error": str(e),
        }


def ceo_route_departments(state: OrchestratorState) -> List[str]:
    """
    Conditional edge function: determines which department subgraph(s) to run next.
    Returns a list of node names to fan-out to (LangGraph Send API).
    """
    if state.get("clarification_needed"):
        return ["clarification_node"]

    departments = state.get("active_departments", [])
    sequence = state.get("task_plan", {}).get("sequence", "sequential")

    if not departments:
        return ["aggregator_node"]

    # For sequential: run first department; for parallel: run all
    if sequence == "sequential":
        # Find first department without completed dependency
        completed = set(state.get("completed_departments", []))
        subtasks = state.get("task_plan", {}).get("subtasks", [])

        for subtask in subtasks:
            dept = subtask["department"]
            depends = subtask.get("depends_on")
            if dept in completed:
                continue
            if depends is None or depends in completed:
                return [f"{dept}_department_node"]

        return ["aggregator_node"]
    else:
        # Parallel: send to all departments at once
        return [f"{d}_department_node" for d in departments]


async def clarification_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    Emits a clarification request. In production this would use LangGraph interrupt().
    For now returns final_output with the question.
    """
    question = state.get("clarification_question", "Could you provide more details about your request?")
    events = emit_event(state, "clarification_needed", data=question)
    return {
        "final_output": f"❓ **Clarification needed:** {question}",
        "agent_events": events,
    }
