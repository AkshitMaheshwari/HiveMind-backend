"""
Supabase PostgreSQL Database Client for Universal Multi-Agent Orchestrator.
Supports live Supabase database storage with fallback to in-memory store if keys are not set.
"""
import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("db_service")
logger.setLevel(logging.INFO)


class SupabaseDatabaseService:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.client = None
        self._in_memory_tasks: Dict[str, Any] = {}
        self._in_memory_events: Dict[str, List[Dict[str, Any]]] = {}

        if self.supabase_url and self.supabase_key:
            try:
                from supabase import create_client, Client
                self.client: Optional[Client] = create_client(self.supabase_url, self.supabase_key)
                logger.info("✅ Supabase client initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Supabase client: {e}. Falling back to in-memory store.")
                self.client = None
        else:
            logger.info("ℹ️ SUPABASE_URL / SUPABASE_KEY missing in .env. Using in-memory fallback store.")

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    async def create_task(self, task_id: str, user_request: str, conversation_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
        """Creates a new task entry in the database."""
        now = datetime.utcnow().isoformat()
        task_data = {
            "id": task_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "user_request": user_request,
            "status": "queued",
            "task_plan": None,
            "final_output": None,
            "error": None,
            "created_at": now,
            "completed_at": None,
        }

        if self.client:
            try:
                response = self.client.table("tasks").insert(task_data).execute()
                if response.data:
                    return response.data[0]
            except Exception as e:
                logger.error(f"Error inserting task to Supabase: {e}")

        # Fallback in-memory
        self._in_memory_tasks[task_id] = task_data
        self._in_memory_events[task_id] = []
        return task_data

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates status, final output, task_plan, or error for an existing task."""
        if "status" in updates and updates["status"] in ("done", "error") and "completed_at" not in updates:
            updates["completed_at"] = datetime.utcnow().isoformat()

        if self.client:
            try:
                # Add .neq("status", "deleted") so background tasks don't resurrect deleted tasks
                response = self.client.table("tasks").update(updates).eq("id", task_id).neq("status", "deleted").execute()
                if response.data:
                    return response.data[0]
                else:
                    return None
            except Exception as e:
                logger.error(f"Error updating task in Supabase: {e}")

        # Fallback in-memory
        if task_id in self._in_memory_tasks:
            if self._in_memory_tasks[task_id].get("status") == "deleted":
                return None # Do not resurrect soft-deleted task
            self._in_memory_tasks[task_id].update(updates)
            return self._in_memory_tasks[task_id]
        return None

    async def save_event(
        self,
        task_id: str,
        event_type: str,
        department: Optional[str] = None,
        agent: Optional[str] = None,
        data: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves a streaming execution event for a task."""
        now = datetime.utcnow().isoformat()
        event_data = {
            "task_id": task_id,
            "event_type": event_type,
            "department": department,
            "agent": agent,
            "data": data,
            "timestamp": now,
        }

        if self.client:
            try:
                self.client.table("task_events").insert(event_data).execute()
            except Exception as e:
                logger.error(f"Error saving event to Supabase: {e}")

        # Fallback in-memory
        if task_id not in self._in_memory_events:
            self._in_memory_events[task_id] = []
        self._in_memory_events[task_id].append(event_data)
        return event_data

    async def get_task(self, task_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves task details and its associated events, enforcing user ownership."""
        if not user_id and self.is_connected:
            logger.warning(f"get_task called without user_id for task {task_id}")
            return None

        task_obj = None

        if self.client:
            try:
                query = self.client.table("tasks").select("*").eq("id", task_id)
                if user_id:
                    query = query.eq("user_id", user_id)
                response = query.execute()
                if response.data:
                    task_obj = response.data[0]
                    # Fetch events strictly for this task
                    events_resp = self.client.table("task_events").select("*").eq("task_id", task_id).order("id").execute()
                    task_obj["events"] = events_resp.data or []
                    return task_obj
                else:
                    return None
            except Exception as e:
                logger.error(f"Error fetching task from Supabase: {e}")

        # Fallback in-memory
        if task_id in self._in_memory_tasks:
            task_obj = dict(self._in_memory_tasks[task_id])
            if user_id and task_obj.get("user_id") and task_obj["user_id"] != user_id:
                return None
            task_obj["events"] = self._in_memory_events.get(task_id, [])
            return task_obj

        return None

    async def list_tasks(self, limit: int = 50, conversation_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists recent tasks belonging strictly to the specified user."""
        if not user_id and self.is_connected:
            logger.warning("list_tasks requested without user_id filter")
            return []

        if self.client:
            try:
                query = self.client.table("tasks").select("*").order("created_at", desc=True).limit(limit)
                query = query.neq("status", "deleted")  # Exclude soft-deleted tasks
                if conversation_id:
                    query = query.eq("conversation_id", conversation_id)
                if user_id:
                    query = query.eq("user_id", user_id)
                response = query.execute()
                if response.data:
                    return response.data
                return []
            except Exception as e:
                logger.error(f"Error listing tasks from Supabase: {e}")

        # Fallback in-memory
        tasks = list(self._in_memory_tasks.values())
        if conversation_id:
            tasks = [t for t in tasks if t.get("conversation_id") == conversation_id]
        if user_id:
            tasks = [t for t in tasks if t.get("user_id") == user_id]
        
        # Exclude soft-deleted tasks
        tasks = [t for t in tasks if t.get("status") != "deleted"]
            
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return tasks[:limit]

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetches user profile including role (user / admin)."""
        if self.client:
            try:
                resp = self.client.table("profiles").select("*").eq("id", user_id).execute()
                if resp.data:
                    return resp.data[0]
            except Exception as e:
                logger.error(f"Error fetching profile: {e}")
        return {"id": user_id, "role": "user", "email": "user@example.com"}

    async def delete_task(self, task_id: str, user_id: Optional[str] = None) -> bool:
        """
        Deletes a task and its events from the database.
        If the database RLS policies block hard-deletes, it performs a soft-delete (status='deleted').
        """
        if self.client:
            try:
                # Verify ownership before deleting
                query = self.client.table("tasks").select("id").eq("id", task_id)
                if user_id:
                    query = query.eq("user_id", user_id)
                check = query.execute()
                if not check.data:
                    logger.warning(f"delete_task: task {task_id} not found or not owned by user {user_id}")
                    return False

                # Delete associated events first
                try:
                    self.client.table("task_events").delete().eq("task_id", task_id).execute()
                except Exception as e:
                    logger.warning(f"Error deleting events for task {task_id}: {e}")

                # Try hard-delete the task
                resp = self.client.table("tasks").delete().eq("id", task_id).execute()
                
                # If RLS blocks the DELETE (common when DELETE policy is missing), fallback to Soft Delete
                if len(resp.data) == 0:
                    logger.info(f"Hard delete blocked by RLS for task {task_id}. Falling back to soft-delete.")
                    update_resp = self.client.table("tasks").update({"status": "deleted"}).eq("id", task_id).execute()
                    if len(update_resp.data) == 0:
                        return False
                        
                logger.info(f"Task {task_id} deleted successfully.")
                return True
            except Exception as e:
                logger.error(f"Error deleting task {task_id} from Supabase: {e}")
                return False

        # Fallback in-memory
        if task_id in self._in_memory_tasks:
            task = self._in_memory_tasks[task_id]
            if user_id and task.get("user_id") and task["user_id"] != user_id:
                return False
            del self._in_memory_tasks[task_id]
            self._in_memory_events.pop(task_id, None)
            return True
        return False

    async def get_conversation_history(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Returns the last `limit` completed turns in a conversation as
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}] pairs.
        Used to inject memory into the orchestrator on every new message.
        """
        tasks: List[Dict[str, Any]] = []

        if self.client:
            try:
                resp = (
                    self.client.table("tasks")
                    .select("user_request, final_output, created_at")
                    .eq("conversation_id", conversation_id)
                    .eq("user_id", user_id)
                    .eq("status", "done")
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                tasks = resp.data or []
            except Exception as e:
                logger.error(f"Error fetching conversation history: {e}")
        else:
            tasks = [
                t for t in self._in_memory_tasks.values()
                if t.get("conversation_id") == conversation_id
                and t.get("user_id") == user_id
                and t.get("status") == "done"
            ]
            tasks = sorted(tasks, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

        # Reverse so oldest turn is first (chronological order for LLM)
        turns: List[Dict[str, str]] = []
        for t in reversed(tasks):
            if t.get("user_request"):
                turns.append({"role": "user", "content": t["user_request"]})
            if t.get("final_output"):
                turns.append({"role": "assistant", "content": t["final_output"]})
        return turns

    async def list_conversations(
        self, user_id: str, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Returns distinct conversation threads for a user, each with:
        - conversation_id
        - first_message (the earliest user_request in that thread)
        - last_updated (latest created_at)
        - message_count
        Used by the sidebar to show conversation threads instead of individual tasks.
        """
        if self.client:
            try:
                resp = (
                    self.client.table("tasks")
                    .select("conversation_id, user_request, created_at")
                    .eq("user_id", user_id)
                    .neq("status", "deleted")
                    .order("created_at", desc=False)
                    .execute()
                )
                rows = resp.data or []
            except Exception as e:
                logger.error(f"Error listing conversations: {e}")
                rows = []
        else:
            rows = [
                t for t in self._in_memory_tasks.values()
                if t.get("user_id") == user_id and t.get("status") != "deleted"
            ]

        # Group by conversation_id
        from collections import defaultdict
        grouped: Dict[str, list] = defaultdict(list)
        for row in rows:
            cid = row.get("conversation_id", "default")
            grouped[cid].append(row)

        conversations = []
        for cid, items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x.get("created_at", ""))
            conversations.append({
                "conversation_id": cid,
                "first_message": (items_sorted[0].get("user_request") or "Untitled Chat")[:120],
                "last_updated": items_sorted[-1].get("created_at", ""),
                "message_count": len(items_sorted),
            })

        # Sort by last_updated desc, trim to limit
        conversations.sort(key=lambda x: x["last_updated"], reverse=True)
        return conversations[:limit]

    async def list_admin_all_tasks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Lists all system tasks across all users (Admin view)."""
        if self.client:
            try:
                resp = self.client.table("tasks").select("*, profiles(email, role)").order("created_at", desc=True).limit(limit).execute()
                if resp.data:
                    return resp.data
            except Exception as e:
                logger.error(f"Error fetching admin tasks: {e}")

        return list(self._in_memory_tasks.values())[:limit]


# Global database service instance
db_service = SupabaseDatabaseService()
