"""
Chat API routes — POST /api/chat to start a task, GET /api/task/{id} to poll.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from api.websocket.stream import manager

router = APIRouter()

# In-memory task store (replace with Redis/DB for production)
task_store: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"


class ChatResponse(BaseModel):
    task_id: str
    status: str
    message: str


async def run_task_async(task_id: str, user_request: str, conversation_id: str):
    """
    Runs the orchestrator in a background thread and streams events via WebSocket.
    """
    import sys
    from pathlib import Path
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    task_store[task_id]["status"] = "running"

    try:
        # Run in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        def _run():
            from orchestrator.graph import compiled_graph
            initial_state = {
                "user_request": user_request,
                "conversation_id": conversation_id,
                "task_plan": None,
                "active_departments": [],
                "completed_departments": [],
                "department_outputs": {},
                "agent_events": [],
                "final_output": "",
                "clarification_needed": False,
                "clarification_question": None,
                "error": None,
            }
            return compiled_graph.invoke(initial_state)

        final_state = await loop.run_in_executor(None, _run)

        # Stream all events to connected WebSocket clients
        events = final_state.get("agent_events", [])
        await manager.send_events(task_id, events)

        # Send final output event
        await manager.broadcast(task_id, {
            "event": "final_output",
            "data": final_state.get("final_output", "Task completed."),
            "department": None,
            "agent": "CEO",
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Update task store
        task_store[task_id].update({
            "status": "done",
            "final_output": final_state.get("final_output", ""),
            "events": events,
            "completed_at": datetime.utcnow().isoformat(),
        })

        # Send done signal
        await manager.broadcast(task_id, {
            "event": "task_done",
            "data": "Task completed successfully",
            "timestamp": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        error_msg = str(e)
        task_store[task_id].update({"status": "error", "error": error_msg})
        await manager.broadcast(task_id, {
            "event": "error",
            "data": error_msg,
            "timestamp": datetime.utcnow().isoformat(),
        })


@router.post("/chat", response_model=ChatResponse)
async def start_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Start a new orchestrated task. Returns a task_id for WebSocket connection."""
    task_id = str(uuid.uuid4())

    task_store[task_id] = {
        "task_id": task_id,
        "user_request": request.message,
        "conversation_id": request.conversation_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "final_output": None,
        "events": [],
    }

    background_tasks.add_task(
        run_task_async,
        task_id,
        request.message,
        request.conversation_id,
    )

    return ChatResponse(
        task_id=task_id,
        status="queued",
        message=f"Task started. Connect to WebSocket: /ws/{task_id}",
    )


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Poll task status and get final output when done."""
    if task_id not in task_store:
        return {"error": "Task not found"}
    return task_store[task_id]


@router.get("/tasks")
async def list_tasks():
    """List all tasks in the current session."""
    return [
        {
            "task_id": t["task_id"],
            "user_request": t["user_request"][:80],
            "status": t["status"],
            "created_at": t["created_at"],
        }
        for t in task_store.values()
    ]
