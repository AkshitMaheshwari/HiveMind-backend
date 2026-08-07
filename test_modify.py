import asyncio
from db.supabase_client import db_service

async def test():
    resp = db_service.client.table("tasks").select("*").execute()
    tasks = resp.data
    for t in tasks:
        if t["status"] == "running":
            print(f"Running task found: {t['id']}, user: {t['user_id']}")
            # Try updating it to error
            res = db_service.client.table("tasks").update({"status": "error", "error": "Force killed"}).eq("id", t["id"]).execute()
            print("Update result:", res)
            
            # Try deleting it
            res_del = db_service.client.table("tasks").delete().eq("id", t["id"]).execute()
            print("Delete result:", res_del)

if __name__ == "__main__":
    asyncio.run(test())
