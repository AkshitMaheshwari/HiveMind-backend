"""
WebSocket connection manager and task streaming handler.
Each task gets a unique task_id. The client connects to /ws/{task_id}
and receives real-time agent events as JSON.
"""
import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages active WebSocket connections keyed by task_id."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        if task_id not in self._connections:
            self._connections[task_id] = set()
        self._connections[task_id].add(websocket)

    def disconnect(self, task_id: str, websocket: WebSocket):
        if task_id in self._connections:
            self._connections[task_id].discard(websocket)
            if not self._connections[task_id]:
                del self._connections[task_id]

    async def broadcast(self, task_id: str, message: dict):
        """Send a JSON message to all clients subscribed to this task_id."""
        if task_id not in self._connections:
            return
        dead = set()
        for ws in self._connections[task_id]:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections[task_id].discard(ws)

    async def send_events(self, task_id: str, events: list):
        """Replay a list of events to connected clients."""
        for event in events:
            await self.broadcast(task_id, event)
            await asyncio.sleep(0.05)  # Small delay for smooth streaming feel

    def has_connections(self, task_id: str) -> bool:
        return bool(self._connections.get(task_id))


# Singleton instance
manager = ConnectionManager()
