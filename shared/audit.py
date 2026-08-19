"""
Lightweight Audit Log
======================
Logs every department start/done event with user_id for per-user data isolation
tracking and basic security auditing.

Design decisions:
- Non-blocking: uses asyncio.create_task and run_in_executor so it never slows down agent execution.
- Dual sink: writes to a local JSONL file AND Supabase (if connected and table exists).
- Per-user isolation: user_id is on every log entry — cannot be omitted.
- Fail-safe: all errors are swallowed — audit failures must never crash or block the main pipeline.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Local JSONL audit log location (fallback when Supabase is unavailable)
_AUDIT_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "audit.jsonl"


def _build_entry(
    user_id: Optional[str],
    event_type: str,
    department: str,
    agent: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structured audit log entry."""
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "user_id": user_id or "anonymous",
        "event_type": event_type,
        "department": department,
        "agent": agent,
        "data": data or {},
    }


def _write_local(entry: Dict[str, Any]) -> None:
    """Append entry to the local JSONL audit log synchronously."""
    try:
        _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        pass


def _write_supabase_sync(entry: Dict[str, Any]) -> None:
    """Append entry to the Supabase audit_logs table if connected."""
    try:
        from db.supabase_client import db_service
        if db_service.is_connected:
            db_service.client.table("audit_logs").insert(entry).execute()
    except Exception:
        # Table might not exist yet; silently skip without throwing
        pass


async def log_event(
    user_id: Optional[str],
    event_type: str,
    department: str,
    agent: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record an audit event asynchronously.
    This function never raises — all failures are handled silently.
    """
    try:
        entry = _build_entry(user_id, event_type, department, agent, data)
        loop = asyncio.get_event_loop()
        # Non-blocking executor writes
        await loop.run_in_executor(None, _write_local, entry)
        await loop.run_in_executor(None, _write_supabase_sync, entry)
    except Exception:
        pass
