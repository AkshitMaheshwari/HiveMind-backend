"""
Chat API routes — POST /api/chat to start a task, GET /api/task/{id} to poll.
Integrated with Supabase PostgreSQL Database Service for persistent storage.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from api.websocket.stream import manager
from db.supabase_client import db_service
from api.auth import get_optional_user, require_authenticated_user, require_admin_user

router = APIRouter()


class ApiKeys(BaseModel):
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    api_keys: Optional[ApiKeys] = None
    selected_model: Optional[str] = None  # e.g. "gemini-2.0-flash", "llama-3.3-70b-versatile"


class ChatResponse(BaseModel):
    task_id: str
    status: str
    message: str


async def run_task_async(
    task_id: str,
    user_request: str,
    conversation_id: str,
    user_id: str,
    api_keys: Optional[Dict[str, str]] = None,
    selected_model: Optional[str] = None,
):
    """
    Runs the orchestrator in a background thread, updates DB, and streams events via WebSocket.
    """
    import sys
    from pathlib import Path
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    await db_service.update_task(task_id, {"status": "running"})

    # ── Fetch conversation history for memory context ──────────────
    chat_history = []
    if user_id and conversation_id and conversation_id != "default":
        try:
            chat_history = await db_service.get_conversation_history(
                conversation_id=conversation_id,
                user_id=user_id,
                limit=10,
            )
        except Exception as hist_exc:
            logger.warning("Could not fetch conversation history: %s", hist_exc)

    try:
        # Run in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        def _run():
            from orchestrator.graph import compiled_graph
            initial_state = {
                "user_request": user_request,
                "conversation_id": conversation_id,
                "chat_history": chat_history,
                "api_keys": api_keys,
                "selected_model": selected_model,
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

        # Stream all events to connected WebSocket clients and save to DB
        events = final_state.get("agent_events", [])
        for ev in events:
            await db_service.save_event(
                task_id=task_id,
                event_type=ev.get("event", "event"),
                department=ev.get("department"),
                agent=ev.get("agent"),
                data=ev.get("data"),
            )

        await manager.send_events(task_id, events)

        final_output = final_state.get("final_output", "Task completed.")

        # Stream the final output token-by-token for a typewriter effect
        words = final_output.split(" ")
        chunk_size = 4  # send N words at a time
        accumulated = ""
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            accumulated += ("" if i == 0 else " ") + chunk
            await manager.broadcast(task_id, {
                "event": "partial_output",
                "data": accumulated,
                "timestamp": datetime.utcnow().isoformat(),
            })
            await asyncio.sleep(0.02)  # ~50 chunks/second

        # Update DB task record
        await db_service.update_task(task_id, {
            "status": "done",
            "final_output": final_output,
            "task_plan": final_state.get("task_plan"),
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
        await db_service.update_task(task_id, {"status": "error", "error": error_msg})
        await manager.broadcast(task_id, {
            "event": "error",
            "data": error_msg,
            "timestamp": datetime.utcnow().isoformat(),
        })


@router.get("/models")
async def get_models():
    """Return the full model registry for all supported providers."""
    from shared.llm import MODEL_REGISTRY
    return MODEL_REGISTRY


@router.post("/chat", response_model=ChatResponse)
async def start_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_authenticated_user),
):
    """Start a new orchestrated task linked to user_id. Returns a task_id for WebSocket connection."""
    task_id = str(uuid.uuid4())
    user_id = user.get("id") if user else None

    api_keys_dict = request.api_keys.model_dump(exclude_none=True) if request.api_keys else None

    await db_service.create_task(
        task_id=task_id,
        user_request=request.message,
        conversation_id=request.conversation_id,
        user_id=user_id,
    )

    background_tasks.add_task(
        run_task_async,
        task_id,
        request.message,
        request.conversation_id,
        user_id,
        api_keys_dict,
        request.selected_model,
    )

    return ChatResponse(
        task_id=task_id,
        status="queued",
        message=f"Task started. Connect to WebSocket: /ws/{task_id}",
    )


@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    user: Dict[str, Any] = Depends(require_authenticated_user),
):
    """Fetch task status, metadata, and event history from DB for the authenticated user."""
    user_id = user.get("id") if user else None
    task_data = await db_service.get_task(task_id, user_id=user_id)
    if not task_data:
        return {"error": "Task not found"}
    return task_data


@router.get("/tasks")
async def list_tasks(
    conversation_id: str = None,
    user: Dict[str, Any] = Depends(require_authenticated_user),
):
    """List recent tasks belonging to the current authenticated user."""
    user_id = user.get("id") if user else None
    tasks = await db_service.list_tasks(limit=50, conversation_id=conversation_id, user_id=user_id)
    return [
        {
            "task_id": t.get("id") or t.get("task_id"),
            "user_request": (t.get("user_request") or "")[:80],
            "status": t.get("status"),
            "created_at": t.get("created_at"),
        }
        for t in tasks
    ]


@router.get("/admin/tasks")
async def list_admin_tasks(admin_user: Dict[str, Any] = Depends(require_admin_user)):
    """Admin endpoint: List all tasks across all users in the system."""
    tasks = await db_service.list_admin_all_tasks(limit=100)
    return tasks


@router.get("/conversations")
async def list_conversations(
    user: Dict[str, Any] = Depends(require_authenticated_user),
):
    """List conversation threads for the current user (for sidebar)."""
    user_id = user.get("id") if user else None
    if not user_id:
        return []
    return await db_service.list_conversations(user_id=user_id)


@router.get("/conversation/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    user: Dict[str, Any] = Depends(require_authenticated_user),
):
    """
    Returns all tasks in a conversation thread as an ordered list of
    {user_request, final_output, status, created_at} objects.
    Used to reload a full chat thread when clicking a conversation in the sidebar.
    """
    user_id = user.get("id") if user else None
    if not user_id:
        return []
    tasks = await db_service.list_tasks(
        limit=100,
        conversation_id=conversation_id,
        user_id=user_id,
    )
    # Return in chronological order
    tasks_sorted = sorted(tasks, key=lambda t: t.get("created_at", ""))
    return [
        {
            "task_id": t.get("id") or t.get("task_id"),
            "user_request": t.get("user_request", ""),
            "final_output": t.get("final_output", ""),
            "status": t.get("status"),
            "created_at": t.get("created_at"),
        }
        for t in tasks_sorted
    ]


@router.delete("/task/{task_id}")
async def delete_task(
    task_id: str,
    user: Dict[str, Any] = Depends(require_authenticated_user),
):
    """
    Hard-delete a task and all its events from the database.
    Only the task owner can delete their own task.
    """
    user_id = user.get("id") if user else None
    success = await db_service.delete_task(task_id, user_id=user_id)
    if success:
        return {"success": True, "message": f"Task {task_id} deleted successfully."}
    return {"success": False, "message": "Task not found or you do not have permission to delete it."}
