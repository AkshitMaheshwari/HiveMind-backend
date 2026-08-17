"""
Lightweight Audit Log
======================
Logs every department start/done event with user_id for per-user data isolation
tracking and basic security auditing.

Design decisions:
- Non-blocking: uses asyncio.create_task so it never slows down agent execution.
- Dual sink: writes to a local JSONL file AND Supabase (if connected).
- Per-user isolation: user_id is on every log entry — cannot be omitted.
- Fail-safe: all errors are swallowed with a warning — audit failures must
  never crash the main pipeline.

Usage::

    from shared.audit import log_event

    # Fire-and-forget in an async context
    asyncio.create_task(log_event(
        user_id=state.get("user_id"),
        event_type="department_started",
        department="analytics",
        agent="DataProfilerAgent",
        data={"task": task[:100]},
    ))
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
        logger.warning("audit: local write failed: %s", exc)


async def _write_supabase(entry: Dict[str, Any]) -> None:
    """Append entry to the Supabase audit_logs table if connected."""
    try:
        from db.supabase_client import db_service
        if db_service.is_connected:
            await db_service.client.table("audit_logs").insert(entry).execute()
    except Exception as exc:
        logger.warning("audit: Supabase write failed: %s", exc)


async def log_event(
    user_id: Optional[str],
    event_type: str,
    department: str,
    agent: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record an audit event asynchronously.

    Parameters:
        user_id: The authenticated user who triggered this event.
                 Per-user data isolation is enforced by always including this.
        event_type: e.g. "department_started", "department_done", "tool_called".
        department: The department generating this event, e.g. "analytics".
        agent: The specific agent, e.g. "DataProfilerAgent".
        data: Optional extra key-value payload (task preview, result sizes, etc.).

    This function never raises — all failures are logged as warnings only.
    """
    try:
        entry = _build_entry(user_id, event_type, department, agent, data)

        # Write to local file in a thread (non-blocking)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write_local, entry)

        # Write to Supabase (best-effort)
        await _write_supabase(entry)

    except Exception as exc:
        logger.warning("audit: log_event failed silently: %s", exc)
