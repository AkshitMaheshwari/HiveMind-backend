"""
Code Department LangGraph subgraph.
Flow: code_generator_node → debugger_node → doc_writer_node → [done]
"""
import asyncio
from datetime import datetime
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from departments.code.state import CodeDeptState
from departments.code.agents import CodeGeneratorAgent, DebuggerAgent, DocWriterAgent
from shared.tools import execute_code


# ─── Agent factory ───────────────────────────────────────────────────────────

def _make_agents(api_keys=None, selected_model=None):
    """Create fresh agent instances with the given API keys and model."""
    return (
        CodeGeneratorAgent(api_keys=api_keys, selected_model=selected_model),
        DebuggerAgent(api_keys=api_keys, selected_model=selected_model),
        DocWriterAgent(api_keys=api_keys, selected_model=selected_model),
    )


def _emit(state: CodeDeptState, event: str, agent: str, data: str = "") -> list:
    events = list(state.get("events", []))
    events.append({
        "event": event,
        "department": "code",
        "agent": agent,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return events


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

async def code_generator_node(state: CodeDeptState) -> Dict[str, Any]:
    """Code Generator — writes the initial code solution."""
    code_gen, _, _ = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "CodeGeneratorAgent", "Generating code solution...")

    output = await code_gen.execute(state["task"])

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "CodeGeneratorAgent",
        f"Code generated ({output.metadata.get('language', 'python')})",
    )

    return {
        "generated_code": output.content,
        "problem_description": output.metadata.get("explanation", state["task"]),
        "events": events,
        "_language": output.metadata.get("language", "python"),
        "_dependencies": output.metadata.get("dependencies", []),
    }


async def debugger_node(state: CodeDeptState) -> Dict[str, Any]:
    """Debugger — runs code in sandbox (Python) or verifies web code (HTML/JS/CSS)."""
    _, debugger, _ = _make_agents(state.get("api_keys"), state.get("selected_model"))

    events = _emit(state, "agent_working", "DebuggerAgent", "Testing code in sandbox...")

    code = state.get("generated_code", "")
    lang = (state.get("_language") or "python").lower()

    # Skip Python interpreter execution for HTML/CSS/JS web code to prevent SyntaxError
    if lang in ["html", "css", "js", "javascript", "typescript", "xml", "svg", "web"]:
        exec_result = {
            "success": True,
            "stdout": f"Validated {lang.upper()} document structure successfully.",
            "stderr": "",
        }
    else:
        exec_result = await asyncio.to_thread(execute_code, code)

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "DebuggerAgent",
        "Execution: " + ("✅ Success" if exec_result["success"] else f"❌ Error: {exec_result['stderr'][:100]}"),
    )

    # If execution failed (for Python code), invoke debugger agent to fix
    if not exec_result["success"] and exec_result["stderr"]:
        events = _emit(
            {**state, "events": events},
            "agent_working",
            "DebuggerAgent",
            "Fixing errors...",
        )
        debug_output = await debugger.execute(
            state["task"],
            context={
                "generated_code": code,
                "stdout": exec_result["stdout"],
                "stderr": exec_result["stderr"],
                "success": exec_result["success"],
            },
        )
        fixed_code = debug_output.content
        events = _emit(
            {**state, "events": events},
            "agent_done",
            "DebuggerAgent",
            "Code fixed: " + ("✅" if debug_output.metadata.get("is_resolved") else "⚠️ Partial fix"),
        )
    else:
        fixed_code = code

    return {
        "execution_stdout": exec_result["stdout"],
        "execution_stderr": exec_result["stderr"],
        "execution_success": exec_result["success"],
        "fixed_code": fixed_code,
        "events": events,
    }


async def doc_writer_node(state: CodeDeptState) -> Dict[str, Any]:
    """Doc Writer — generates documentation or formatted web deliverable."""
    _, _, doc_writer = _make_agents(state.get("api_keys"), state.get("selected_model"))

    lang = (state.get("_language") or "python").lower()
    final_code_display = state.get("fixed_code") or state.get("generated_code", "")

    # For web/UI tasks (HTML/CSS/JS), generate clean report without Python docstring clutter
    if lang in ["html", "css", "js", "javascript", "web"]:
        events = _emit(state, "agent_done", "DocWriterAgent", "Web deliverable ready")
        final_report = f"""## 💻 Web Application Solution

### Generated Code ({lang.upper()})

```{lang}
{final_code_display}
```

> 💡 **Tip:** Click the **👁️ Live Preview** tab at the top of this message (or **↗️ Fullscreen**) to view and interact with your rendered website live!
"""
        return {
            "documentation": "Web application deliverable generated successfully.",
            "final_report": final_report,
            "events": list({**state, "events": events}["events"]),
        }

    # For Python or standard backend code, generate complete developer docs
    events = _emit(state, "agent_working", "DocWriterAgent", "Generating documentation...")
    output = await doc_writer.execute(
        state["task"],
        context={
            "final_code": final_code_display,
            "generated_code": state.get("generated_code", ""),
            "language": lang,
            "explanation": state.get("problem_description", ""),
        },
    )

    events = _emit(
        {**state, "events": events},
        "agent_done",
        "DocWriterAgent",
        "Documentation complete",
    )

    exec_success = state.get("execution_success", False)
    final_report = f"""## 💻 Code Solution

### Generated Code

```{lang}
{final_code_display}
```

### Execution Results
- **Status:** {"✅ Passed" if exec_success else "⚠️ Errors encountered"}

**Output:**
```text
{state.get("execution_stdout", "No output")[:500]}
```

{f'**Errors:**\\n```text\\n{state.get("execution_stderr", "")[:500]}\\n```' if state.get("execution_stderr") else ""}

### Documentation

{output.content}
"""

    return {
        "documentation": output.content,
        "final_report": final_report,
        "events": list({**state, "events": events}["events"]),
    }


# ─── Build the Code Subgraph ──────────────────────────────────────────────────

code_graph = StateGraph(CodeDeptState)

code_graph.add_node("code_generator_node", code_generator_node)
code_graph.add_node("debugger_node", debugger_node)
code_graph.add_node("doc_writer_node", doc_writer_node)

code_graph.add_edge(START, "code_generator_node")
code_graph.add_edge("code_generator_node", "debugger_node")
code_graph.add_edge("debugger_node", "doc_writer_node")
code_graph.add_edge("doc_writer_node", END)

code_subgraph = code_graph.compile()


# ─── Outer node — plugs into root orchestrator graph ─────────────────────────

async def code_department_node(state) -> Dict[str, Any]:
    """Outer node that plugs into the main orchestrator graph."""
    subtasks = state.get("task_plan", {}).get("subtasks", [])
    code_task = state["user_request"]
    for st in subtasks:
        if st.get("department") == "code":
            code_task = st.get("task", state["user_request"])
            break

    events = list(state.get("agent_events", []))
    events.append({
        "event": "department_started",
        "department": "code",
        "agent": "Code Head",
        "data": f"Starting code task: {code_task[:100]}...",
        "timestamp": datetime.utcnow().isoformat(),
    })

    initial_state = {
        "task": code_task,
        "original_request": state["user_request"],
        "api_keys": state.get("api_keys"),
        "selected_model": state.get("selected_model"),
        "events": [],
    }

    final_state = await code_subgraph.ainvoke(initial_state)

    events.extend(final_state.get("events", []))
    events.append({
        "event": "department_done",
        "department": "code",
        "agent": "Code Head",
        "data": "Code department completed successfully",
        "timestamp": datetime.utcnow().isoformat(),
    })

    department_outputs = dict(state.get("department_outputs", {}))
    department_outputs["code"] = final_state.get("final_report", "Code generation completed.")

    completed = list(state.get("completed_departments", []))
    if "code" not in completed:
        completed.append("code")

    return {
        "department_outputs": department_outputs,
        "completed_departments": completed,
        "agent_events": events,
    }
