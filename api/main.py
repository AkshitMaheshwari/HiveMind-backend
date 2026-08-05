"""
FastAPI application entry point.
Serves the REST API, WebSocket endpoint, and static frontend.
"""
import sys
from pathlib import Path

# Ensure backend root is in sys.path for all imports
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes.chat import router as chat_router
from api.websocket.stream import manager


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Universal Multi-Agent Orchestrator",
    description="An AI Company with CEO, Research, Content, and Code departments powered by LangGraph",
    version="1.0.0",
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


# ─── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time agent event streaming.
    Client connects to /ws/{task_id} after calling POST /api/chat.
    """
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
