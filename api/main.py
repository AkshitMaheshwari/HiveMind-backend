"""
FastAPI application entry point.
Serves the REST API, WebSocket endpoint, and static frontend.
"""
import logging
import sys
from pathlib import Path

# Ensure backend root is in sys.path for all imports
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

logger = logging.getLogger(__name__)

from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes.chat import router as chat_router
from api.routes.upload import router as upload_router
from api.websocket.stream import manager
from api.auth import verify_token
from db.supabase_client import db_service


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Universal Multi-Agent Orchestrator",
    description="An AI Company with CEO, Research, Content, and Code departments powered by LangGraph",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    """
    Application startup: bootstrap the tool registry and initialise the
    Qdrant vector collection (idempotent — safe to run on every start).
    """
    try:
        from shared.tools.registry_bootstrap import bootstrap
        bootstrap()
    except Exception as exc:
        # Log loudly — a broken registry at startup is a hard failure.
        logger.critical(
            "Tool registry bootstrap FAILED — agents may not have access to tools: %s",
            exc,
            exc_info=True,
        )
        raise

    try:
        from rag.vector_store import ensure_collection
        ensure_collection()
        logger.info("Qdrant vector collection initialised successfully.")
    except Exception as exc:
        # Non-fatal: RAG won't work, but other agents still can.
        logger.warning(
            "Qdrant initialisation failed — RAG features will be unavailable: %s", exc
        )

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API Routes ───────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(upload_router, prefix="/api", tags=["Upload"])


# ─── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str, token: Optional[str] = Query(None)):
    """
    WebSocket endpoint for real-time agent event streaming.
    Client connects to /ws/{task_id}?token={bearer_token} after calling POST /api/chat.
    """
    user = await verify_token(token) if token else None
    user_id = user.get("id") if user else None

    # Enforce task ownership when Supabase is connected
    if db_service.is_connected:
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required")
            return
        task = await db_service.get_task(task_id, user_id=user_id)
        if not task:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized or task not found")
            return

    await manager.connect(task_id, websocket)
    try:
        # Keep connection alive; events are pushed by background task
        while True:
            # Receive any client messages (e.g. ping)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text('{"event": "pong"}')
    except WebSocketDisconnect:
        manager.disconnect(task_id, websocket)


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Universal Multi-Agent Orchestrator",
        "version": "1.0.0",
    }


# ─── Serve Frontend (optional — only if frontend dir exists) ─────────────────
_FRONTEND_DIR = _BACKEND_ROOT.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
