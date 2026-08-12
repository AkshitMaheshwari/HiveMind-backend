import asyncio
from db.supabase_client import db_service

async def main():
    tasks = await db_service.list_admin_all_tasks(limit=5)
    for t in tasks:
        print(f"Task: {t.get('id')} | Status: {t.get('status')} | Error: {t.get('error')}")
        print(f"  Events count: {len(t.get('events', []))}")

asyncio.run(main())
